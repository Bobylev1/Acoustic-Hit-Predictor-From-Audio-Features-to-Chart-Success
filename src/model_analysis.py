"""
Model Analysis and Visualization Module
Кросс-валидация, интерпретация модели и визуализация результатов
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.model_selection import KFold, cross_validate, ParameterGrid, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

# Настройка стиля графиков
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Проверка доступности SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# PartialDependenceDisplay импортируется отложенно в функции


def perform_cross_validation_analysis(model, X_train, y_train, figures_path='../reports/figures'):
    """
    Задача 1: Кросс-валидация с визуализацией
    
    Args:
        model: Модель для валидации
        X_train: Обучающие признаки
        y_train: Целевая переменная
        figures_path: Путь для сохранения графиков
        
    Returns:
        Словарь с результатами кросс-валидации
    """
    print("\n" + "="*70)
    print("ЗАДАЧА 1: КРОСС-ВАЛИДАЦИЯ (k=5)")
    print("="*70)
    
    os.makedirs(figures_path, exist_ok=True)
    
    # K-Fold Cross-Validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    print("Выполняется кросс-валидация на 5 фолдах...")
    print("Это займет ~1-2 минуты...\n")
    
    # Вычисление всех метрик на одних и тех же фолдах
    cv_results_raw = cross_validate(
        estimator=model,
        X=X_train,
        y=y_train,
        cv=kfold,
        scoring={
            'MAE': make_scorer(mean_absolute_error, greater_is_better=False),
            'RMSE': make_scorer(lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred)), greater_is_better=False),
            'R2': 'r2'
        },
        n_jobs=1,
        return_train_score=False,
        verbose=0
    )
    
    print("✓ Кросс-валидация завершена")
    
    # Обработка результатов
    cv_results = {}
    for metric_name in ['MAE', 'RMSE', 'R2']:
        scores = cv_results_raw[f'test_{metric_name}']
        
        if metric_name in ['MAE', 'RMSE']:
            scores = np.abs(scores)
        
        cv_results[metric_name] = scores
        
        print(f"\n{metric_name}:")
        print(f"  Среднее: {scores.mean():.4f}")
        print(f"  Стандартное отклонение: {scores.std():.4f}")
        print(f"  Минимум: {scores.min():.4f}, Максимум: {scores.max():.4f}")
        print(f"  Все значения: {['%.4f' % s for s in scores]}")
    
    # Визуализация кросс-валидации
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (metric, ax) in enumerate(zip(['MAE', 'RMSE', 'R2'], axes)):
        scores = cv_results[metric]
        ax.boxplot([scores], labels=[metric])
        ax.set_ylabel(metric, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric}: {scores.mean():.4f} ± {scores.std():.4f}',
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('K-Fold Cross-Validation Results (k=5)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, '01_cross_validation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Детали по фолдам
    print("\n" + "="*70)
    print("ДЕТАЛИ ПО ФОЛДАМ:")
    print("="*70)
    
    fold_data = []
    for i in range(kfold.n_splits):
        fold_data.append({
            'Фолд': i+1,
            'MAE': f"{cv_results['MAE'][i]:.4f}",
            'RMSE': f"{cv_results['RMSE'][i]:.4f}",
            'R²': f"{cv_results['R2'][i]:.4f}"
        })
    df_folds = pd.DataFrame(fold_data)
    print(df_folds.to_string(index=False))
    
    # Стабильность модели
    r2_scores = cv_results['R2']
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
    print(f"Средний R²: {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")
    print(f"Коэффициент вариации R²: {(r2_scores.std() / r2_scores.mean() * 100):.2f}%")
    
    # Анализ согласованности метрик
    correlation_matrix = np.corrcoef([cv_results['MAE'], cv_results['RMSE'], cv_results['R2']])
    
    print("\n" + "="*70)
    print("АНАЛИЗ СОГЛАСОВАННОСТИ МЕТРИК:")
    print("="*70)
    print("Корреляция между метриками по фолдам:")
    print(f"MAE ↔ RMSE: {correlation_matrix[0,1]:.4f}")
    print(f"MAE ↔ R²:   {correlation_matrix[0,2]:.4f}")
    print(f"RMSE ↔ R²:  {correlation_matrix[1,2]:.4f}")
    
    return {
        'cv_results': cv_results,
        'stability': stability,
        'fold_data': df_folds
    }


def perform_hyperparameter_tuning(X_train, y_train, figures_path='../reports/figures'):
    """
    Задача 2: Гиперпараметрическая оптимизация с прогресс-баром
    
    Args:
        X_train: Обучающие признаки
        y_train: Целевая переменная
        figures_path: Путь для сохранения графиков
        
    Returns:
        Обученная модель с лучшими параметрами и результаты оптимизации
    """
    print("\n" + "="*70)
    print("ЗАДАЧА 2: ГИПЕРПАРАМЕТРИЧЕСКАЯ ОПТИМИЗАЦИЯ")
    print("="*70)
    
    os.makedirs(figures_path, exist_ok=True)
    
    # Упрощённая сетка параметров для быстроты
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [15, 20, None],
        'min_samples_split': [5, 10]
    }
    
    total_combinations = len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split'])
    print(f"Проверяем {total_combinations} комбинаций")
    print("Используем 3-фолдовую CV\n")
    
    all_params = list(ParameterGrid(param_grid))
    results = []
    best_score = float('inf')
    best_params = None
    
    # Прогресс-бар
    total_fits = len(all_params) * 3
    print(f"Всего обучений: {total_fits}")
    pbar = tqdm(total=len(all_params), desc="Комбинации параметров")
    
    for params in all_params:
        # Модель с текущими параметрами
        model = RandomForestRegressor(
            **params,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        # 3-фолдовая кросс-валидация
        scores = cross_val_score(
            model, X_train, y_train,
            cv=3,
            scoring='neg_mean_absolute_error',
            n_jobs=1
        )
        
        mae = abs(scores.mean())
        
        if mae < best_score:
            best_score = mae
            best_params = params
        
        results.append({
            **params,
            'MAE': mae,
            'Std': scores.std()
        })

        pbar.update(1)
        pbar.set_postfix({
            'Лучший': f"{best_score:.4f}",
            'Текущий': f"{mae:.4f}",
            'Осталось': f"{len(all_params) - (results.__len__())}"
        })
    
    pbar.close()
    
    # Анализ результатов
    results_df = pd.DataFrame(results).sort_values('MAE')
    
    print("\n" + "="*70)
    print("РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ")
    print("="*70)
    print(f"\nЛучшие параметры:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"Лучший MAE (CV): {best_score:.4f}")
    
    print(f"\nТоп-5 комбинаций:")
    for i, (_, row) in enumerate(results_df.head().iterrows(), 1):
        print(f"\n{i}. MAE: {row['MAE']:.4f} ± {row['Std']:.4f}")
        for param in param_grid.keys():
            print(f"   {param}: {row[param]}")
    
    # Обучение финальной модели
    print("\n" + "="*70)
    print("ОБУЧЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ")
    print("="*70)
    
    final_model = RandomForestRegressor(
        **best_params,
        min_samples_leaf=4,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    print("Обучение на полном train set...")
    final_model.fit(X_train, y_train)
    
    # Визуализация результатов оптимизации
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # График 1: Все результаты
    axes[0].scatter(range(len(results_df)), results_df['MAE'], alpha=0.6, s=50)
    axes[0].axhline(y=best_score, color='red', linestyle='--', linewidth=2,
                    label=f'Лучший: {best_score:.4f}')
    axes[0].set_xlabel('Номер комбинации', fontweight='bold')
    axes[0].set_ylabel('MAE', fontweight='bold')
    axes[0].set_title('Все комбинации параметров', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # График 2: Топ-10
    top_n = min(10, len(results_df))
    top_results = results_df.head(top_n).copy()
    
    y_pos = np.arange(top_n)
    axes[1].barh(y_pos, top_results['MAE'][::-1], color='steelblue', alpha=0.7)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([f"#{i+1}" for i in range(top_n)][::-1])
    axes[1].set_xlabel('MAE', fontweight='bold')
    axes[1].set_title(f'Топ-{top_n} комбинаций', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Результаты гиперпараметрической оптимизации', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, '02_hyperparameter_optimization.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return final_model, best_params, results_df


def analyze_model_interpretation(model, X_train, X_test, feature_names, 
                                 figures_path='../reports/figures'):
    """
    Задача 4: Интерпретация модели
    
    Args:
        model: Обученная модель
        X_train: Обучающие признаки
        X_test: Тестовые признаки
        feature_names: Названия признаков
        figures_path: Путь для сохранения графиков
        
    Returns:
        Словарь с результатами интерпретации
    """
    print("\n" + "="*70)
    print("ЗАДАЧА 4: ИНТЕРПРЕТАЦИЯ МОДЕЛИ")
    print("="*70)
    
    os.makedirs(figures_path, exist_ok=True)
    
    # 1. Feature Importance
    print("\n" + "-"*70)
    print("1. FEATURE IMPORTANCE")
    print("-"*70)
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nТоп-15 признаков по важности:")
    print("-"*70)
    for i, (_, row) in enumerate(feature_importance.head(15).iterrows(), 1):
        print(f"{i:2d}. {row['feature']:35s}: {row['importance']:.6f}")
    
    # Визуализация Feature Importance
    plt.figure(figsize=(12, 10))
    top_features = feature_importance.head(20)
    
    y_pos = np.arange(len(top_features))
    plt.barh(y_pos, top_features['importance'], color='steelblue', alpha=0.7)
    plt.yticks(y_pos, top_features['feature'], fontsize=9)
    plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
    plt.ylabel('Признак', fontsize=12, fontweight='bold')
    plt.title('Топ-20 признаков по важности (Random Forest)', 
              fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    
    # Добавляем значения на графики
    for i, (_, row) in enumerate(top_features.iterrows()):
        plt.text(row['importance'] + 0.001, i, f'{row["importance"]:.4f}',
                 va='center', fontweight='bold', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, '04_feature_importance.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. SHAP Values (если доступен)
    if SHAP_AVAILABLE:
        print("\n" + "-"*70)
        print("2. SHAP VALUES")
        print("-"*70)
        print("Вычисление SHAP values...")
        
        explainer = shap.TreeExplainer(model)
        
        # Используем небольшую выборку для ускорения (100-200 примеров достаточно)
        sample_size = min(200, len(X_test))
        X_test_sample = X_test.sample(n=sample_size, random_state=42)
        
        # Вычисление с прогресс баром (батчами)
        batch_size = 20
        n_batches = (sample_size + batch_size - 1) // batch_size
        shap_values_list = []
        
        print(f"Обработка {sample_size} примеров батчами по {batch_size}...")
        for i in tqdm(range(0, sample_size, batch_size), 
                     desc="SHAP values", 
                     total=n_batches):
            batch = X_test_sample.iloc[i:i+batch_size]
            batch_shap = explainer.shap_values(batch, check_additivity=False)
            shap_values_list.append(batch_shap)
        
        # Объединяем результаты
        shap_values = np.vstack(shap_values_list)
        
        print(f"✓ SHAP values вычислены для {sample_size} примеров")
        
        # SHAP Summary Plot (Bar)
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_test_sample, plot_type="bar", show=False)
        plt.title('SHAP Feature Importance (Top 20)', 
                  fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_path, '04_shap_importance.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # SHAP Summary Plot (Dot)
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_test_sample, show=False)
        plt.title('SHAP Summary Plot', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_path, '04_shap_summary.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # SHAP Waterfall для одного примера
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=X_test_sample.iloc[0],
                feature_names=feature_names
            ),
            show=False
        )
        plt.title('SHAP Waterfall Plot (Пример 1)', 
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_path, '04_shap_waterfall.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\n✓ SHAP графики сохранены")
    else:
        print("\n⚠ SHAP не установлен (pip install shap)")
        print("Пропускаем анализ SHAP values")
    
    # 3. Partial Dependence Plots
    print("\n" + "-"*70)
    print("3. PARTIAL DEPENDENCE PLOTS")
    print("-"*70)
    
    # Локальный импорт для избежания проблем при загрузке модуля
    try:
        from sklearn.inspection import PartialDependenceDisplay
        
        top_5_features = feature_importance.head(5)['feature'].tolist()
        
        # Используем выборку для ускорения
        sample_size_pdp = min(5000, len(X_train))
        X_train_sample = X_train.sample(n=sample_size_pdp, random_state=42)
        print(f"Используется выборка: {sample_size_pdp} записей из {len(X_train)}")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        print("Построение Partial Dependence Plots...")
        for idx, feature in enumerate(tqdm(top_5_features, desc="PDP графики")):
            if idx < len(axes):
                PartialDependenceDisplay.from_estimator(
                    model, X_train_sample, [feature],
                    ax=axes[idx],
                    grid_resolution=50
                )
                axes[idx].set_title(f'PDP: {feature}', fontsize=11, fontweight='bold')
    
        # Удаляем лишний subplot
        if len(top_5_features) < len(axes):
            fig.delaxes(axes[-1])
        
        plt.suptitle('Partial Dependence Plots (Топ-5 признаков)', 
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_path, '04_partial_dependence.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\n✓ Partial Dependence Plots сохранены")
    
    except ImportError:
        print("\n⚠ sklearn.inspection.PartialDependenceDisplay недоступен")
        print("Пропускаем Partial Dependence Plots")
    
    # 4. Итоговый анализ
    print("\n" + "="*70)
    print("ВЫВОД: ИНТЕРПРЕТАЦИЯ МОДЕЛИ")
    print("="*70)
    print("\n1. Feature Importance показывает вклад признаков в модель")
    print("2. SHAP values объясняют предсказания для конкретных примеров")
    print("3. PDP показывают зависимость предсказаний от значений признаков")
    
    return {
        'feature_importance': feature_importance,
        'top_features': top_5_features
    }


def run_full_analysis(model, X_train, X_test, y_train, y_test, feature_names, save_model=False):
    """
    Запуск полного анализа модели (Задачи 1, 2, 4)
    
    Args:
        model: Базовая модель для кросс-валидации
        X_train, X_test: Признаки
        y_train, y_test: Целевая переменная
        feature_names: Названия признаков
        save_model: Сохранять ли новую модель (по умолчанию False, чтобы не перезаписывать)
        
    Returns:
        Словарь со всеми результатами
    """
    print("="*70)
    print("ЗАПУСК ПОЛНОГО АНАЛИЗА МОДЕЛИ")
    print("="*70)
    
    # Определяем абсолютный путь к reports/figures
    if os.path.basename(os.getcwd()) == 'src':
        # Если запускаем из src/
        figures_path = '../reports/figures'
    else:
        # Если запускаем из корня проекта
        figures_path = 'reports/figures'
    
    figures_path = os.path.abspath(figures_path)
    os.makedirs(figures_path, exist_ok=True)
    print(f"Графики будут сохранены в: {figures_path}\n")
    
    # Задача 1: Кросс-валидация
    cv_results = perform_cross_validation_analysis(model, X_train, y_train, figures_path)
    
    # Задача 2: Оптимизация
    final_model, best_params, optimization_results = perform_hyperparameter_tuning(
        X_train, y_train, figures_path
    )
    
    # Финальная оценка
    y_train_pred = final_model.predict(X_train)
    y_test_pred = final_model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("\n" + "="*70)
    print("ФИНАЛЬНЫЕ МЕТРИКИ")
    print("="*70)
    print(f"Train MAE: {train_mae:.4f}, Train R²: {train_r2:.4f}")
    print(f"Test MAE:  {test_mae:.4f}, Test R²:  {test_r2:.4f}")
    
    # Задача 4: Интерпретация
    interpretation_results = analyze_model_interpretation(
        final_model, X_train, X_test, feature_names, figures_path
    )
    
    # Сохранение модели (опционально)
    if save_model:
        if os.path.basename(os.getcwd()) == 'src':
            model_path = '../models/final_random_forest_model.pkl'
        else:
            model_path = 'models/final_random_forest_model.pkl'
        
        model_path = os.path.abspath(model_path)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(final_model, model_path)
        print(f"\n✓ Модель сохранена: {model_path}")
    else:
        model_path = None
        print(f"\n⚠ Модель НЕ сохранена (используй save_model=True для сохранения)")
    
    print("\n" + "="*70)
    print("✅ АНАЛИЗ ЗАВЕРШЁН")
    print("="*70)
    print(f"Все графики сохранены в: {figures_path}")
    
    return {
        'cv_results': cv_results,
        'best_params': best_params,
        'optimization_results': optimization_results,
        'interpretation': interpretation_results,
        'metrics': {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2
        },
        'model_path': model_path
    }


if __name__ == '__main__':
    # Пример использования (требуется предварительная загрузка данных)
    from model_optimization import load_and_prepare_data
    
    print("Загрузка данных...")
    X_train, X_test, y_train, y_test, df_features = load_and_prepare_data()
    
    # Получение имен признаков
    numeric_features = df_features.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['popularity', 'track_id', 'key', 'mode', 'time_signature']
    feature_names = [f for f in numeric_features if f not in exclude_cols]
    
    # Базовая модель
    base_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # Запуск полного анализа
    results = run_full_analysis(base_model, X_train, X_test, y_train, y_test, feature_names)
    
    print("\n" + "="*70)
    print("ИТОГИ")
    print("="*70)
    print(f"Стабильность: {results['cv_results']['stability']}")
    print(f"Лучшие параметры: {results['best_params']}")
    print(f"Test MAE: {results['metrics']['test_mae']:.4f}")
    print(f"Test R²: {results['metrics']['test_r2']:.4f}")
    print(f"Модель: {results['model_path']}")
    print(f"Графики: {os.path.abspath('../reports/figures')}")
