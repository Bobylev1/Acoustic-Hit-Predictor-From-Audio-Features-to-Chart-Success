"""
Скрипт для запуска анализа модели (Задачи 1, 2, 4)
"""

import sys
import os

# Добавляем src в путь
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model_optimization import load_and_prepare_data
from model_analysis import run_full_analysis
from sklearn.ensemble import RandomForestRegressor
import numpy as np


if __name__ == '__main__':
    print("="*70)
    print("ЗАПУСК АНАЛИЗА МОДЕЛИ")
    print("="*70)
    print("Будут выполнены:")
    print("  - Задача 1: Кросс-валидация")
    print("  - Задача 2: Гиперпараметрическая оптимизация")
    print("  - Задача 4: Интерпретация модели")
    print("="*70)
    
    # 1. Загрузка данных
    print("\nЗагрузка данных...")
    X_train, X_test, y_train, y_test, df_features = load_and_prepare_data()
    
    # Получение имен признаков
    numeric_features = df_features.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['popularity', 'track_id', 'key', 'mode', 'time_signature']
    feature_names = [f for f in numeric_features if f not in exclude_cols]
    
    # 2. Базовая модель для кросс-валидации
    base_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # 3. Запуск полного анализа (без сохранения модели, чтобы не перезаписать основную)
    print("\nЗапуск анализа...\n")
    results = run_full_analysis(
        base_model, X_train, X_test, y_train, y_test, feature_names,
        save_model=False  # Не перезаписываем модель, только создаем графики
    )
    
    # 4. Итоги
    print("\n" + "="*70)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ АНАЛИЗА")
    print("="*70)
    print(f"Стабильность модели: {results['cv_results']['stability']}")
    print(f"Лучшие параметры из анализа: {results['best_params']}")
    print(f"Test MAE (анализируемой модели): {results['metrics']['test_mae']:.4f}")
    print(f"Test R² (анализируемой модели): {results['metrics']['test_r2']:.4f}")
    
    if results['model_path']:
        print(f"\nМодель сохранена: {results['model_path']}")
    else:
        print(f"\n⚠ Модель НЕ была сохранена (только анализ)")
        print(f"💡 Основная модель осталась в: models/final_random_forest_model.pkl")
    
    print(f"\nГрафики сохранены: {os.path.abspath('reports/figures')}")
    print("\nАнализ завершён!")
