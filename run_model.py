"""
Скрипт для запуска обученной модели
"""

import sys
import os

# Добавляем src в путь
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from run_model import main


if __name__ == '__main__':
    main()
