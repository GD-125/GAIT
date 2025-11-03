# explainability/__init__.py
"""
Explainability Module for Gait Detection System

Provides SHAP-based model interpretability and feature importance analysis.

Example usage:
    from explainability import GaitSHAPExplainer
    from models import CNN_BiLSTM_GaitDetector, get_device
    
    # Initialize model and explainer
    device = get_device()
    model = CNN_BiLSTM_GaitDetector(input_features=38, seq_length=128)
    explainer = GaitSHAPExplainer(model=model, device=device)
    
    # Create explainer with background data
    explainer.create_explainer(background_data=train_data[:100])
    
    # Explain test samples
    shap_values = explainer.explain_samples(test_data=test_data[:20])
    
    # Generate comprehensive report
    explainer.generate_explanation_report(test_data=test_data[:20])
    
    # Get feature importance
    importance = explainer.get_feature_importance()
"""

from .shap_explainer import GaitSHAPExplainer

__all__ = [
    'GaitSHAPExplainer'
]

__version__ = '1.0.0'
