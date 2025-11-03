# src/dashboard/lime_explainer_dash.py
"""
LIME Explainer optimized for dashboard display
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class DashboardLIMEExplainer:
    """
    LIME explainer optimized for dashboard use.
    Fast and simple explanations for real-time display.
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: torch.device):
        """
        Initialize LIME explainer.
        
        Args:
            model: Trained PyTorch model
            device: Device model is on
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
        # Generate feature names
        self.feature_names = self._generate_feature_names()
    
    def _generate_feature_names(self) -> List[str]:
        """Generate descriptive feature names."""
        feature_names = []
        
        # Accelerometer features (18)
        for location in ['right_foot', 'right_shin', 'right_thigh', 
                        'left_foot', 'left_shin', 'left_thigh']:
            for axis in ['x', 'y', 'z']:
                feature_names.append(f'accel_{location}_{axis}')
        
        # Gyroscope features (18)
        for location in ['right_foot', 'right_shin', 'right_thigh',
                        'left_foot', 'left_shin', 'left_thigh']:
            for axis in ['x', 'y', 'z']:
                feature_names.append(f'gyro_{location}_{axis}')
        
        # EMG features (2)
        feature_names.extend(['EMG_right', 'EMG_left'])
        
        return feature_names
    
    def _predict_fn(self, samples: np.ndarray) -> np.ndarray:
        """
        Prediction function for LIME.
        
        Args:
            samples: Perturbed samples (num_samples, seq_length, features)
            
        Returns:
            Predictions
        """
        self.model.eval()
        
        with torch.no_grad():
            samples_tensor = torch.FloatTensor(samples).to(self.device)
            outputs = self.model(samples_tensor).cpu().numpy()
            return outputs.flatten()
    
    def _perturb_sample(self,
                       sample: np.ndarray,
                       num_samples: int = 500) -> tuple:
        """
        Create perturbed versions of a sample.
        
        Args:
            sample: Original sample (seq_length, features)
            num_samples: Number of perturbations
            
        Returns:
            Perturbed samples and binary masks
        """
        seq_length, num_features = sample.shape
        
        # Create perturbation masks
        masks = np.random.binomial(1, 0.5, size=(num_samples, num_features))
        
        # Calculate feature-wise standard deviations
        feature_stds = np.std(sample, axis=0) * 0.1
        feature_stds = np.where(feature_stds == 0, 0.01, feature_stds)
        
        # Generate perturbed samples
        perturbed_samples = np.zeros((num_samples, seq_length, num_features))
        
        for i in range(num_samples):
            perturbed = sample.copy()
            
            for feat_idx in range(num_features):
                if masks[i, feat_idx] == 0:  # Perturb this feature
                    noise = np.random.normal(0, feature_stds[feat_idx], seq_length)
                    perturbed[:, feat_idx] += noise
            
            perturbed_samples[i] = perturbed
        
        return perturbed_samples, masks
    
    def explain_single(self, 
                      sample: np.ndarray,
                      sample_idx: int = 0) -> Dict:
        """
        Explain a single prediction.
        
        Args:
            sample: Sample to explain (seq_length, features)
            sample_idx: Index for tracking
            
        Returns:
            Dictionary with explanation
        """
        # Get original prediction
        with torch.no_grad():
            original_tensor = torch.FloatTensor(sample).unsqueeze(0).to(self.device)
            original_pred = float(self.model(original_tensor).cpu().numpy()[0, 0])
        
        # Generate perturbed samples
        perturbed_samples, masks = self._perturb_sample(sample, num_samples=500)
        
        # Get predictions for perturbed samples
        perturbed_preds = self._predict_fn(perturbed_samples)
        
        # Fit linear model
        distances = np.sum(1 - masks, axis=1)
        weights = np.exp(-distances / (masks.shape[1] * 0.25))
        
        scaler = StandardScaler()
        masks_scaled = scaler.fit_transform(masks)
        
        ridge = Ridge(alpha=1.0)
        ridge.fit(masks_scaled, perturbed_preds, sample_weight=weights)
        
        # Get feature importance
        feature_importance = ridge.coef_
        base_value = ridge.intercept_
        r2_score = ridge.score(masks_scaled, perturbed_preds, sample_weight=weights)
        
        # Create explanation
        explanation = {
            'sample_idx': sample_idx,
            'prediction': original_pred,
            'predicted_class': 'Gait' if original_pred > 0.5 else 'Non-Gait',
            'base_value': float(base_value),
            'feature_importance': feature_importance,
            'feature_names': self.feature_names[:len(feature_importance)],
            'r2_score': float(r2_score)
        }
        
        return explanation
    
    def explain_batch(self,
                     samples: np.ndarray,
                     num_samples: int = 5) -> Dict:
        """
        Explain multiple samples and aggregate.
        
        Args:
            samples: Batch of samples (num_samples, seq_length, features)
            num_samples: Number to explain
            
        Returns:
            Dictionary with individual and global explanations
        """
        num_to_explain = min(num_samples, len(samples))
        
        sample_explanations = []
        all_importance = []
        
        for i in range(num_to_explain):
            explanation = self.explain_single(samples[i], sample_idx=i)
            sample_explanations.append(explanation)
            all_importance.append(explanation['feature_importance'])
        
        # Calculate global importance
        global_importance = np.mean(np.abs(all_importance), axis=0)
        
        # Create importance dictionary
        importance_dict = {}
        for i, importance in enumerate(global_importance):
            if i < len(self.feature_names):
                importance_dict[self.feature_names[i]] = float(importance)
        
        # Sort by importance
        importance_dict = dict(sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        return {
            'samples': sample_explanations,
            'global_importance': importance_dict,
            'num_explained': num_to_explain
        }
