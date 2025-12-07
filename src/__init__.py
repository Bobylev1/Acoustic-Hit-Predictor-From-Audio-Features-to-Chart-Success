"""
Acoustic Hit Predictor - Source Package
Вспомогательные модули для ML проекта
"""

from .data_preprocessing import (
    load_data,
    check_missing_values,
    remove_duplicates,
    handle_outliers,
    scale_features,
    split_data,
    get_numeric_columns,
    get_categorical_columns
)

from .feature_engineering import (
    create_duration_features,
    create_energy_features,
    create_mood_features,
    create_acoustic_features,
    create_tempo_features,
    create_loudness_features,
    create_all_features,
    get_feature_correlations
)

from .visualization import (
    plot_distribution,
    plot_multiple_distributions,
    plot_correlation_matrix,
    plot_target_correlation,
    plot_scatter_with_target,
    plot_pairplot,
    plot_feature_importance,
    plot_model_comparison,
    plot_predictions_vs_actual,
    plot_residuals
)

from .models import (
    calculate_metrics,
    train_linear_models,
    train_tree_models,
    train_boosting_models,
    train_neural_network,
    train_all_models,
    create_results_table,
    perform_cross_validation,
    get_feature_importance
)

__version__ = '1.0.0'
__author__ = 'mmobik'

__all__ = [
    # Data preprocessing
    'load_data',
    'check_missing_values',
    'remove_duplicates',
    'handle_outliers',
    'scale_features',
    'split_data',
    'get_numeric_columns',
    'get_categorical_columns',
    
    # Feature engineering
    'create_duration_features',
    'create_energy_features',
    'create_mood_features',
    'create_acoustic_features',
    'create_tempo_features',
    'create_loudness_features',
    'create_all_features',
    'get_feature_correlations',
    
    # Visualization
    'plot_distribution',
    'plot_multiple_distributions',
    'plot_correlation_matrix',
    'plot_target_correlation',
    'plot_scatter_with_target',
    'plot_pairplot',
    'plot_feature_importance',
    'plot_model_comparison',
    'plot_predictions_vs_actual',
    'plot_residuals',
    
    # Models
    'calculate_metrics',
    'train_linear_models',
    'train_tree_models',
    'train_boosting_models',
    'train_neural_network',
    'train_all_models',
    'create_results_table',
    'perform_cross_validation',
    'get_feature_importance',
]
