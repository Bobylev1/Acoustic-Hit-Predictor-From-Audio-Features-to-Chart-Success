"""
Скрипт для запуска оптимизации модели
"""

import sys
import os

# Добавляем src в путь
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model_optimization import run_full_optimization


if __name__ == '__main__':
    print("Запуск оптимизации модели...\n")
    
    # Запуск полной оптимизации
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
    print("\nОптимизация завершена!")
