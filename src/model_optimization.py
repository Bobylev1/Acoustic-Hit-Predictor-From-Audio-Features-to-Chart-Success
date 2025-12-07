"""
Model Optimization Module
Кросс-валидация, гиперпараметрическая оптимизация и финальная оценка модели
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import os
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import threading

warnings.filterwarnings('ignore')

# Проверка доступности дополнительных библиотек
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def load_and_prepare_data(dataset_path=None):
    """
    Загрузка и подготовка данных
    
    Args:
        dataset_path: Путь к датасету
        
    Returns:
        X_train, X_test, y_train, y_test, df_features
    """
    print("="*70)
    print("ЗАГРУЗКА ДАННЫХ")
    print("="*70)
    
    if dataset_path is None:
        dataset_path = os.path.join('dataset', 'dataset.csv')
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join('src', 'dataset', 'dataset.csv')
    
    df = pd.read_csv(dataset_path, index_col=0)
    print(f"Загружено записей: {len(df)}")
    print(f"Количество колонок: {len(df.columns)}")
    
    # Создание признаков
    print("\n" + "="*70)
    print("СОЗДАНИЕ ПРИЗНАКОВ")
    print("="*70)
    
    df_features = df.copy()
    
    # Обязательные признаки
    df_features['duration_min'] = df_features['duration_ms'] / 60000
    df_features['energy_dance_ratio'] = df_features['energy'] / (df_features['danceability'] + 1e-6)
    df_features['acoustic_energy_balance'] = df_features['acousticness'] * (1 - df_features['energy'])
    df_features['tempo_energy_product'] = df_features['tempo'] * df_features['energy']
    df_features['valence_energy_interaction'] = df_features['valence'] * df_features['energy']
    
    # Логарифмические трансформации
    df_features['log_duration_ms'] = np.log1p(df_features['duration_ms'])
    df_features['log_instrumentalness'] = np.log1p(df_features['instrumentalness'] * 1e6) / np.log(1e6 + 1)
    df_features['log_speechiness'] = np.log1p(df_features['speechiness'] * 1e6) / np.log(1e6 + 1)
    df_features['log_loudness'] = np.log1p(df_features['loudness'] + 60)
    
    # Бинаризация
    df_features['is_high_energy'] = (df_features['energy'] > 0.7).astype(int)
    df_features['is_major_mode'] = df_features['mode'].astype(int)
    df_features['is_high_danceability'] = (df_features['danceability'] > 0.7).astype(int)
    df_features['is_high_valence'] = (df_features['valence'] > 0.7).astype(int)
    
    # Всегда создаем is_explicit (по умолчанию 0, если нет данных)
    if 'explicit' in df_features.columns:
        df_features['is_explicit'] = df_features['explicit'].astype(int)
    else:
        df_features['is_explicit'] = 0
    
    # Полиномиальные взаимодействия
    key_features_for_poly = ['danceability', 'valence', 'energy', 'tempo']
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    poly_features = poly.fit_transform(df_features[key_features_for_poly])
    poly_feature_names = poly.get_feature_names_out(key_features_for_poly)
    
    for feature_name in poly_feature_names:
        if ' ' in feature_name:
            clean_name = feature_name.replace(' ', '_')
            if clean_name not in df_features.columns:
                idx = list(poly_feature_names).index(feature_name)
                df_features[clean_name] = poly_features[:, idx]
    
    print(f"✓ Создано признаков: {len(df_features.columns) - len(df.columns)}")
    print(f"✓ Итого признаков: {len(df_features.columns)}")
    
    # Подготовка данных для обучения
    numeric_features = df_features.select_dtypes(include=[np.number]).columns.tolist()
    
    if 'popularity' in numeric_features:
        numeric_features.remove('popularity')
    if 'track_id' in numeric_features:
        numeric_features.remove('track_id')
    
    categorical_to_exclude = ['key', 'mode', 'time_signature']
    numeric_features = [f for f in numeric_features if f not in categorical_to_exclude]
    
    X = df_features[numeric_features].copy()
    y = df_features['popularity'].copy()
    
    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.median())
    
    # Разделение на train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("\n" + "="*70)
    print("ПОДГОТОВКА ДАННЫХ")
    print("="*70)
    print(f"Количество признаков: {len(numeric_features)}")
    print(f"Train размер: {X_train.shape[0]:,} записей")
    print(f"Test размер: {X_test.shape[0]:,} записей")
    
    return X_train, X_test, y_train, y_test, df_features


def perform_cross_validation(X_train, y_train, n_splits=5, n_jobs=-1):
    """
    Кросс-валидация модели Random Forest
    
    Args:
        X_train: Обучающие признаки
        y_train: Целевая переменная
        n_splits: Количество фолдов
        n_jobs: Количество параллельных процессов
        
    Returns:
        Словарь с результатами кросс-валидации
    """
    print("\n" + "="*70)
    print(f"K-FOLD CROSS-VALIDATION (k={n_splits})")
    print("="*70)
    
    # Базовая модель
    base_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=n_jobs
    )
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Массивы для хранения метрик
    mae_scores = []
    rmse_scores = []
    r2_scores = []
    
    print("\nПрогресс валидации:")
    for fold_idx, (train_idx, val_idx) in enumerate(tqdm(kfold.split(X_train), total=n_splits, desc="Фолды"), 1):
        X_train_fold = X_train.iloc[train_idx]
        X_val_fold = X_train.iloc[val_idx]
        y_train_fold = y_train.iloc[train_idx]
        y_val_fold = y_train.iloc[val_idx]
        
        # Обучение модели
        base_model.fit(X_train_fold, y_train_fold)
        
        # Предсказания
        y_pred_fold = base_model.predict(X_val_fold)
        
        # Метрики
        mae = mean_absolute_error(y_val_fold, y_pred_fold)
        rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred_fold))
        r2 = r2_score(y_val_fold, y_pred_fold)
        
        mae_scores.append(mae)
        rmse_scores.append(rmse)
        r2_scores.append(r2)
    
    # Конвертируем в numpy arrays
    mae_scores = np.array(mae_scores)
    rmse_scores = np.array(rmse_scores)
    r2_scores = np.array(r2_scores)
    
    # Вывод результатов
    print("\n" + "="*70)
    print("РЕЗУЛЬТАТЫ КРОСС-ВАЛИДАЦИИ")
    print("="*70)
    print(f"\nMAE:  {mae_scores.mean():.4f} ± {mae_scores.std():.4f}")
    print(f"RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")
    print(f"R²:   {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")
    
    # Анализ стабильности
    r2_std = r2_scores.std()
    if r2_std < 0.01:
        stability = "Очень стабильная"
    elif r2_std < 0.02:
        stability = "Стабильная"
    elif r2_std < 0.03:
        stability = "Умеренно стабильная"
    else:
        stability = "Нестабильная"
    
    print("\n" + "="*70)
    print("ВЫВОД: СТАБИЛЬНОСТЬ МОДЕЛИ")
    print("="*70)
    print(f"Модель: {stability}")
    print(f"Коэффициент вариации R²: {(r2_scores.std() / r2_scores.mean() * 100):.2f}%")
    
    return {
        'mae_scores': mae_scores,
        'rmse_scores': rmse_scores,
        'r2_scores': r2_scores,
        'stability': stability
    }


def optimize_hyperparameters_manual(X_train, y_train, n_jobs=-1):
    """
    Ручная гиперпараметрическая оптимизация с прогресс-баром
    
    Args:
        X_train: Обучающие признаки
        y_train: Целевая переменная
        n_jobs: Количество параллельных процессов
        
    Returns:
        Лучшая модель и её параметры
    """
    print("\n" + "="*70)
    print("ГИПЕРПАРАМЕТРИЧЕСКАЯ ОПТИМИЗАЦИЯ (РУЧНАЯ)")
    print("="*70)
    
    # Определяем комбинации параметров
    param_combinations = [
        {'n_estimators': 50, 'max_depth': None, 'min_samples_split': 5},
        {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 5},
        {'n_estimators': 150, 'max_depth': None, 'min_samples_split': 5},
        {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 5},
        {'n_estimators': 150, 'max_depth': None, 'min_samples_split': 10},
        {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 10},
        {'n_estimators': 100, 'max_depth': 25, 'min_samples_split': 5},
        {'n_estimators': 150, 'max_depth': 25, 'min_samples_split': 5},
        {'n_estimators': 200, 'max_depth': 25, 'min_samples_split': 5},
        {'n_estimators': 100, 'max_depth': 30, 'min_samples_split': 5},
        {'n_estimators': 150, 'max_depth': 30, 'min_samples_split': 5},
        {'n_estimators': 200, 'max_depth': 30, 'min_samples_split': 5},
        {'n_estimators': 150, 'max_depth': 20, 'min_samples_split': 5},
        {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 5},
        {'n_estimators': 150, 'max_depth': 15, 'min_samples_split': 5},
        {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5},
        {'n_estimators': 100, 'max_depth': 20, 'min_samples_split': 10},
        {'n_estimators': 150, 'max_depth': 20, 'min_samples_split': 10},
    ]
    
    print(f"Проверяем {len(param_combinations)} комбинаций")
    print(f"Используем 3-фолдовую CV\n")
    
    results = []
    best_score = float('inf')
    best_params = None
    best_model = None
    
    total_combinations = len(param_combinations)
    
    for i, params in enumerate(tqdm(param_combinations, desc="Поиск параметров"), 1):
        # Создаём модель с текущими параметрами
        model = RandomForestRegressor(
            **params,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=42,
            n_jobs=n_jobs,
            verbose=0
        )
        
        # 3-fold CV для ускорения
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        mae_scores = []
        
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            mae_scores.append(mae)
        
        mean_mae = np.mean(mae_scores)
        std_mae = np.std(mae_scores)
        
        results.append({
            'params': params,
            'MAE': mean_mae,
            'Std': std_mae
        })
        
        if mean_mae < best_score:
            best_score = mean_mae
            best_params = params
            best_model = model
    
    # Сортируем результаты по MAE
    results_df = pd.DataFrame(results).sort_values('MAE')
    
    print("\n" + "="*70)
    print("РЕЗУЛЬТАТЫ ПОИСКА")
    print("="*70)
    print(f"\nЛучшие параметры:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"Лучший MAE (CV): {best_score:.4f}")
    
    print(f"\nТоп-5 комбинаций:")
    for i, (_, row) in enumerate(results_df.head(5).iterrows(), 1):
        print(f"\n{i}. MAE: {row['MAE']:.4f} ± {row['Std']:.4f}")
        for param, value in row['params'].items():
            print(f"   {param}: {value}")
    
    return best_model, best_params, results_df


def optimize_hyperparameters_randomized(X_train, y_train, n_iter=50, cv=5, n_jobs=-1):
    """
    Гиперпараметрическая оптимизация с использованием RandomizedSearchCV
    
    Args:
        X_train: Обучающие признаки
        y_train: Целевая переменная
        n_iter: Количество итераций поиска
        cv: Количество фолдов для кросс-валидации
        n_jobs: Количество параллельных процессов
        
    Returns:
        Обученная модель с лучшими параметрами
    """
    print("\n" + "="*70)
    print("СОЗДАНИЕ И ЗАПУСК RandomizedSearchCV")
    print("="*70)
    
    # Параметры для оптимизации
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [15, 20, 25, 30, None],
        'min_samples_split': [5, 10, 15, 20],
        'min_samples_leaf': [2, 4, 5, 8],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Создаем scorer для MAE (меньше = лучше)
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    
    # RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42, n_jobs=1, verbose=0),
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=mae_scorer,
        cv=cv,
        n_jobs=n_jobs,
        random_state=42,
        verbose=2  # Выводит прогресс в консоль
    )
    
    total_fits = n_iter * cv
    print(f"Всего итераций: {n_iter}, фолдов CV: {cv}, всего обучений: {total_fits}")
    print("Начало поиска оптимальных параметров...")
    
    start_time = time.time()
    random_search.fit(X_train, y_train)
    search_time = time.time() - start_time
    
    print(f"\nПоиск завершен за {search_time:.2f} секунд ({search_time/60:.1f} минут)")
    print("\nОптимальные параметры:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"Лучший MAE (CV): {abs(random_search.best_score_):.4f}")
    print("\n" + "="*70)
    
    # Используем уже обученную лучшую модель
    final_model = random_search.best_estimator_
    final_model.set_params(n_jobs=n_jobs)
    
    return final_model, random_search.best_params_


def evaluate_final_model(model, X_train, X_test, y_train, y_test, best_params, save_path='../models'):
    """
    Финальная оценка модели на train и test наборах
    
    Args:
        model: Обученная модель
        X_train, X_test: Признаки
        y_train, y_test: Целевая переменная
        best_params: Лучшие параметры модели
        save_path: Путь для сохранения модели
        
    Returns:
        Словарь с метриками
    """
    print("\n" + "="*70)
    print("ПЕРЕОБУЧЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ")
    print("="*70)
    print("Обучение на полном train set...")
    
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    print(f"Обучение завершено за {train_time:.2f} секунд")
    print(f"Параметры модели: {best_params}")
    
    # Предсказания
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Метрики
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("\n" + "="*70)
    print("ФИНАЛЬНЫЕ МЕТРИКИ")
    print("="*70)
    print(f"\nTrain Set:")
    print(f"  MAE:  {train_mae:.4f}")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  R²:   {train_r2:.4f}")
    
    print(f"\nTest Set:")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  R²:   {test_r2:.4f}")
    
    # Сохранение модели
    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path, 'final_random_forest_model.pkl')
    joblib.dump(model, model_path)
    print(f"\n✓ Модель сохранена: {model_path}")
    
    return {
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'model_path': model_path
    }


def run_full_optimization(dataset_path=None, use_randomized_search=True, n_jobs=-1):
    """
    Запуск полного процесса оптимизации модели
    
    Args:
        dataset_path: Путь к датасету
        use_randomized_search: Использовать RandomizedSearchCV (True) или ручной поиск (False)
        n_jobs: Количество параллельных процессов
        
    Returns:
        Словарь со всеми результатами
    """
    print("="*70)
    print("ЗАПУСК ПОЛНОЙ ОПТИМИЗАЦИИ МОДЕЛИ")
    print("="*70)
    print(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Метод оптимизации: {'RandomizedSearchCV' if use_randomized_search else 'Ручной поиск'}")
    print("="*70)
    
    # 1. Загрузка и подготовка данных
    X_train, X_test, y_train, y_test, df_features = load_and_prepare_data(dataset_path)
    
    # 2. Кросс-валидация (опционально)
    print("\n" + "="*70)
    print("ШАГ 1: КРОСС-ВАЛИДАЦИЯ")
    print("="*70)
    cv_results = perform_cross_validation(X_train, y_train, n_splits=5, n_jobs=n_jobs)
    
    # 3. Гиперпараметрическая оптимизация
    print("\n" + "="*70)
    print("ШАГ 2: ГИПЕРПАРАМЕТРИЧЕСКАЯ ОПТИМИЗАЦИЯ")
    print("="*70)
    
    if use_randomized_search:
        best_model, best_params = optimize_hyperparameters_randomized(
            X_train, y_train, n_iter=50, cv=5, n_jobs=n_jobs
        )
    else:
        best_model, best_params, results_df = optimize_hyperparameters_manual(
            X_train, y_train, n_jobs=n_jobs
        )
    
    # 4. Финальная оценка
    print("\n" + "="*70)
    print("ШАГ 3: ФИНАЛЬНАЯ ОЦЕНКА")
    print("="*70)
    
    eval_results = evaluate_final_model(
        best_model, X_train, X_test, y_train, y_test, best_params
    )
    
    print("\n" + "="*70)
    print("✅ ОПТИМИЗАЦИЯ ЗАВЕРШЕНА УСПЕШНО")
    print("="*70)
    
    return {
        'cv_results': cv_results,
        'best_params': best_params,
        'eval_results': eval_results,
        'model': best_model
    }


if __name__ == '__main__':
    # Запуск оптимизации
    results = run_full_optimization(
        dataset_path=None,  # Автоматически найдет датасет
        use_randomized_search=True,  # True для RandomizedSearchCV, False для ручного поиска
        n_jobs=-1  # Использовать все доступные ядра
    )
    
    print("\n" + "="*70)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("="*70)
    print(f"Стабильность модели: {results['cv_results']['stability']}")
    print(f"Лучшие параметры: {results['best_params']}")
    print(f"Test MAE: {results['eval_results']['test_mae']:.4f}")
    print(f"Test R²: {results['eval_results']['test_r2']:.4f}")
    print(f"Модель сохранена: {results['eval_results']['model_path']}")
