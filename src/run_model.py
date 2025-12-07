"""
Run Model Module
Загрузка обученной модели и выполнение предсказаний
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def download_model_from_hf(local_path='models/final_random_forest_model.pkl'):
    """
    Загрузка модели с Hugging Face
    
    Args:
        local_path: Локальный путь для сохранения модели
    """
    try:
        from huggingface_hub import hf_hub_download
        
        print("⬇️  Модель не найдена локально")
        print("📥 Загрузка модели с Hugging Face...")
        
        # Создаём папку models, если её нет
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Загружаем модель
        downloaded_path = hf_hub_download(
            repo_id="mmobi/Forest_Regressor_v1",
            filename="final_random_forest_model.pkl",
            cache_dir=None,
            local_dir=os.path.dirname(local_path),
            local_dir_use_symlinks=False
        )
        
        print(f"✅ Модель загружена с Hugging Face")
        return local_path
        
    except ImportError:
        raise ImportError(
            "Для загрузки модели установите: pip install huggingface_hub"
        )
    except Exception as e:
        raise RuntimeError(
            f"Ошибка загрузки модели с Hugging Face: {e}\n"
            f"Альтернатива: запустите python run_optimization.py"
        )


def load_model(model_path='../models/final_random_forest_model.pkl'):
    """
    Загрузка обученной модели (локально или с Hugging Face)
    
    Args:
        model_path: Путь к файлу модели
        
    Returns:
        Загруженная модель
    """
    # Пробуем альтернативные пути
    alt_paths = [
        model_path,
        'models/final_random_forest_model.pkl',
        os.path.join('..', 'models', 'final_random_forest_model.pkl'),
        os.path.join('src', '..', 'models', 'final_random_forest_model.pkl')
    ]
    
    found_path = None
    for alt_path in alt_paths:
        if os.path.exists(alt_path):
            found_path = alt_path
            break
    
    # Если не найдена - загружаем с HF
    if found_path is None:
        found_path = download_model_from_hf('models/final_random_forest_model.pkl')
    
    model = joblib.load(found_path)
    print(f"✅ Модель загружена: {found_path}")
    return model


def prepare_features(df):
    """
    Подготовка признаков для предсказания
    Повторяет логику из model_optimization.py
    
    Args:
        df: DataFrame с исходными признаками
        
    Returns:
        DataFrame с подготовленными признаками
    """
    df_features = df.copy()
    
    # Базовые признаки
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
    
    # Выбираем нужные признаки
    numeric_features = df_features.select_dtypes(include=[np.number]).columns.tolist()
    
    # Исключаем целевую переменную и ненужные колонки
    exclude_cols = ['popularity', 'track_id', 'key', 'mode', 'time_signature']
    numeric_features = [f for f in numeric_features if f not in exclude_cols]
    
    X = df_features[numeric_features].copy()
    
    # Заполнение пропусков
    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.median())
    
    return X


def predict(model, X):
    """
    Выполнение предсказаний
    
    Args:
        model: Обученная модель
        X: Признаки для предсказания
        
    Returns:
        Предсказанные значения популярности
    """
    predictions = model.predict(X)
    # Ограничиваем диапазон 0-100
    predictions = np.clip(predictions, 0, 100)
    return predictions


def predict_from_csv(csv_path, model_path='../models/final_random_forest_model.pkl'):
    """
    Предсказание популярности треков из CSV файла
    
    Args:
        csv_path: Путь к CSV файлу с признаками треков
        model_path: Путь к файлу модели
        
    Returns:
        DataFrame с предсказаниями
    """
    print("="*70)
    print("ПРЕДСКАЗАНИЕ ПОПУЛЯРНОСТИ ТРЕКОВ")
    print("="*70)
    
    # Загрузка модели
    model = load_model(model_path)
    
    # Загрузка данных
    print(f"\nЗагрузка данных из: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✓ Загружено треков: {len(df)}")
    
    # Подготовка признаков
    print("\nПодготовка признаков...")
    X = prepare_features(df)
    print(f"✓ Подготовлено признаков: {X.shape[1]}")
    
    # Предсказание
    print("\nВыполнение предсказаний...")
    predictions = predict(model, X)
    
    # Добавляем предсказания в DataFrame
    df['predicted_popularity'] = predictions
    
    # Если есть реальная популярность, считаем ошибку
    if 'popularity' in df.columns:
        df['error'] = abs(df['popularity'] - df['predicted_popularity'])
        mean_error = df['error'].mean()
        print(f"\n✓ Средняя абсолютная ошибка: {mean_error:.2f}")
    
    print(f"✓ Предсказания выполнены для {len(df)} треков")
    print("\n" + "="*70)
    
    return df


def predict_single_track(track_features, model_path='../models/final_random_forest_model.pkl'):
    """
    Предсказание популярности для одного трека
    
    Args:
        track_features: Словарь с характеристиками трека
        model_path: Путь к файлу модели
        
    Returns:
        Предсказанная популярность
    """
    # Создаем DataFrame из словаря
    df = pd.DataFrame([track_features])
    
    # Загрузка модели
    model = load_model(model_path)
    
    # Подготовка признаков
    X = prepare_features(df)
    
    # Предсказание
    prediction = predict(model, X)[0]
    
    return prediction


def show_top_predictions(df, n=10):
    """
    Показать топ-N треков по предсказанной популярности
    
    Args:
        df: DataFrame с предсказаниями
        n: Количество треков для показа
    """
    print(f"\nТОП-{n} ТРЕКОВ ПО ПРЕДСКАЗАННОЙ ПОПУЛЯРНОСТИ:")
    print("="*70)
    
    top_tracks = df.nlargest(n, 'predicted_popularity')
    
    for idx, (i, row) in enumerate(top_tracks.iterrows(), 1):
        print(f"\n{idx}. Предсказанная популярность: {row['predicted_popularity']:.1f}")
        
        # Показываем название трека, если есть
        if 'track_name' in df.columns:
            print(f"   Трек: {row['track_name']}")
        if 'artist_name' in df.columns:
            print(f"   Исполнитель: {row['artist_name']}")
        
        # Показываем реальную популярность, если есть
        if 'popularity' in df.columns:
            print(f"   Реальная популярность: {row['popularity']:.1f}")
            print(f"   Ошибка: {row['error']:.1f}")
        
        # Ключевые характеристики
        print(f"   Energy: {row['energy']:.2f}, Danceability: {row['danceability']:.2f}, "
              f"Valence: {row['valence']:.2f}")


def main():
    """
    Основная функция для демонстрации работы
    """
    print("="*70)
    print("ACOUSTIC HIT PREDICTOR - ЗАПУСК МОДЕЛИ")
    print("="*70)
    
    # Пример 1: Предсказание для датасета
    dataset_path = os.path.join('src', 'dataset', 'dataset.csv')
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join('dataset', 'dataset.csv')
    
    if os.path.exists(dataset_path):
        print("\n1. Загрузка полного датасета...")
        df = pd.read_csv(dataset_path, index_col=0)
        
        # Берем случайную выборку для демонстрации
        sample_df = df.sample(n=min(100, len(df)), random_state=42)
        
        print(f"   Взята выборка: {len(sample_df)} треков")
        
        # Предсказания
        model = load_model()
        X = prepare_features(sample_df)
        predictions = predict(model, X)
        
        sample_df['predicted_popularity'] = predictions
        if 'popularity' in sample_df.columns:
            sample_df['error'] = abs(sample_df['popularity'] - sample_df['predicted_popularity'])
            print(f"   Средняя ошибка: {sample_df['error'].mean():.2f}")
        
        # Показываем топ-10
        show_top_predictions(sample_df, n=10)
    
    # Пример 2: Предсказание для одного трека
    print("\n\n2. Пример предсказания для одного трека:")
    print("="*70)
    
    example_track = {
        'duration_ms': 200000,
        'danceability': 0.7,
        'energy': 0.8,
        'loudness': -5.0,
        'speechiness': 0.05,
        'acousticness': 0.1,
        'instrumentalness': 0.0,
        'liveness': 0.1,
        'valence': 0.6,
        'tempo': 120.0,
        'mode': 1,
        'key': 5,
        'time_signature': 4
    }
    
    print("\nХарактеристики трека:")
    for key, value in example_track.items():
        print(f"  {key}: {value}")
    
    predicted_popularity = predict_single_track(example_track)
    print(f"\nПредсказанная популярность: {predicted_popularity:.1f}/100")
    
    print("\n" + "="*70)
    print("✅ ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
    print("="*70)


if __name__ == '__main__':
    main()
