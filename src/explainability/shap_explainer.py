# explainability/shap_explainer_fixed.py
"""
Fixed SHAP Explainability Module for Gait Detection System
Resolves indexing and visualization bugs
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import shap
from typing import List, Optional, Tuple, Dict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class GaitSHAPExplainer:
    """
    SHAP-based explainability for gait detection models.
    Explains which features contribute most to gait vs non-gait predictions.
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 feature_names: Optional[List[str]] = None):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained PyTorch model
            device: Device model is on
            feature_names: List of feature names for visualization
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
        # Generate default feature names if not provided
        if feature_names is None:
            self.feature_names = self._generate_default_feature_names()
        else:
            self.feature_names = feature_names
        
        self.explainer = None
        self.shap_values = None
        self.base_values = None
    
    def _generate_default_feature_names(self, num_features: int = 38) -> List[str]:
        """Generate default feature names based on sensor configuration."""
        feature_names = []
        
        # Accelerometer features
        for location in ['right_foot', 'right_shin', 'right_thigh', 
                        'left_foot', 'left_shin', 'left_thigh']:
            for axis in ['x', 'y', 'z']:
                feature_names.append(f'accel_{location}_{axis}')
        
        # Gyroscope features
        for location in ['right_foot', 'right_shin', 'right_thigh',
                        'left_foot', 'left_shin', 'left_thigh']:
            for axis in ['x', 'y', 'z']:
                feature_names.append(f'gyro_{location}_{axis}')
        
        # EMG features
        feature_names.extend(['EMG_right', 'EMG_left'])
        
        return feature_names
    
    def create_explainer(self,
                        background_data: np.ndarray,
                        max_background_samples: int = 100):
        """
        Create SHAP Deep Explainer using background data.
        
        Args:
            background_data: Background dataset for SHAP (num_samples, seq_length, features)
            max_background_samples: Maximum samples to use for background
        """
        print(f"\n{'='*60}")
        print("CREATING SHAP EXPLAINER")
        print(f"{'='*60}")
        
        # Sample background data if too large
        if len(background_data) > max_background_samples:
            indices = np.random.choice(len(background_data), 
                                      max_background_samples, 
                                      replace=False)
            background_data = background_data[indices]
        
        # Further reduce for stability
        background_data = background_data[:min(50, len(background_data))]
        
        print(f"Background samples: {len(background_data)}")
        print(f"Data shape: {background_data.shape}")
        
        # Convert to tensor
        background_tensor = torch.FloatTensor(background_data).to(self.device)
        
        try:
            # Create explainer
            self.explainer = shap.DeepExplainer(self.model, background_tensor)
            print(f"✓ SHAP explainer created successfully!")
        except Exception as e:
            print(f"⚠ Warning: Could not create SHAP explainer: {str(e)}")
            print(f"  This may be due to model complexity. Continuing without SHAP...")
            self.explainer = None
        
        print(f"{'='*60}\n")
    
    def explain_samples(self,
                       test_data: np.ndarray,
                       num_samples: Optional[int] = None) -> np.ndarray:
        """
        Generate SHAP values for test samples.
        
        Args:
            test_data: Test data to explain (num_samples, seq_length, features)
            num_samples: Number of samples to explain (None = all)
            
        Returns:
            SHAP values array or None if explainer failed
        """
        if self.explainer is None:
            print("⚠ SHAP explainer not available. Skipping explanations.")
            return None
        
        print(f"\n{'='*60}")
        print("COMPUTING SHAP VALUES")
        print(f"{'='*60}")
        
        # Sample data if needed
        if num_samples is not None and len(test_data) > num_samples:
            indices = np.random.choice(len(test_data), num_samples, replace=False)
            test_data = test_data[indices]
        
        # Limit to 10 samples for stability
        test_data = test_data[:min(10, len(test_data))]
        
        print(f"Explaining {len(test_data)} samples...")
        print(f"This may take a few minutes...")
        
        # Convert to tensor
        test_tensor = torch.FloatTensor(test_data).to(self.device)
        
        try:
            # Compute SHAP values (disable additivity check for complex models)
            try:
                self.shap_values = self.explainer.shap_values(test_tensor, check_additivity=False)
            except TypeError:
                # Older SHAP versions don't have check_additivity parameter
                self.shap_values = self.explainer.shap_values(test_tensor)
            
            # Handle different SHAP output formats
            if isinstance(self.shap_values, list):
                self.shap_values = self.shap_values[0]
            
            print(f"✓ SHAP values computed!")
            print(f"Shape: {self.shap_values.shape}")
            print(f"{'='*60}\n")
            
            return self.shap_values
            
        except Exception as e:
            print(f"⚠ Error computing SHAP values: {str(e)}")
            print(f"  Continuing without SHAP explanations...")
            self.shap_values = None
            return None
    
    def get_feature_importance(self,
                              shap_values: Optional[np.ndarray] = None,
                              aggregate_temporal: bool = True) -> Dict:
        """
        Calculate feature importance from SHAP values.
        
        Args:
            shap_values: SHAP values (if None, uses stored values)
            aggregate_temporal: Average across time dimension
            
        Returns:
            Dictionary with feature importance scores
        """
        if shap_values is None:
            shap_values = self.shap_values
        
        if shap_values is None:
            return {}
        
        try:
            # Average across samples and time dimension
            if aggregate_temporal and len(shap_values.shape) == 3:
                # Shape: (samples, time, features) -> (features,)
                importance = np.mean(np.abs(shap_values), axis=(0, 1))
            else:
                # Shape: (samples, features) -> (features,)
                importance = np.mean(np.abs(shap_values), axis=0)
            
            # Flatten if needed
            if len(importance.shape) > 1:
                importance = importance.flatten()
            
            # Create dictionary
            importance_dict = {}
            
            # Determine number of features to use
            num_features_to_use = min(len(importance), len(self.feature_names))
            
            for i in range(num_features_to_use):
                feature_name = self.feature_names[i % len(self.feature_names)]
                importance_dict[f"{feature_name}_{i//len(self.feature_names)}" if i >= len(self.feature_names) else feature_name] = float(importance[i])
            
            # Sort by importance
            importance_dict = dict(sorted(importance_dict.items(), 
                                         key=lambda x: x[1], 
                                         reverse=True))
            
            return importance_dict
            
        except Exception as e:
            print(f"⚠ Error calculating feature importance: {str(e)}")
            return {}
    
    def plot_feature_importance(self,
                               top_k: int = 20,
                               save_path: Optional[str] = None):
        """
        Plot top-k most important features.
        
        Args:
            top_k: Number of top features to plot
            save_path: Path to save plot (optional)
        """
        try:
            importance_dict = self.get_feature_importance()
            
            if not importance_dict or len(importance_dict) == 0:
                print("⚠ No feature importance data available for plotting")
                return
            
            # Get top-k features
            top_features = list(importance_dict.items())[:top_k]
            
            if len(top_features) == 0:
                print("⚠ No features to plot")
                return
            
            features, importance = zip(*top_features)
            
            # Create plot
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(features)), importance, color='steelblue')
            plt.yticks(range(len(features)), features, fontsize=8)
            plt.xlabel('Mean |SHAP Value|', fontsize=12)
            plt.ylabel('Features', fontsize=12)
            plt.title(f'Top {len(features)} Most Important Features for Gait Detection', 
                     fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✓ Feature importance plot saved to {save_path}")
            
            plt.close()
            
        except Exception as e:
            print(f"⚠ Error plotting feature importance: {str(e)}")
            plt.close()
    
    def plot_shap_summary(self,
                      test_data: np.ndarray,
                      max_display: int = 20,
                      save_path: Optional[str] = None):
        """
        Plot SHAP summary plot (manual beeswarm version).

        Args:
            test_data: Test data used for explanation
            max_display: Maximum features to display
            save_path: Path to save plot (optional)
        """
        if self.shap_values is None:
            print("⚠ No SHAP values available for summary plot")
            return

        try:
            # Prepare SHAP values and test data (avg over time if 3D)
            if self.shap_values.ndim == 3:
                shap_values_2d = np.mean(self.shap_values, axis=1)
                test_data_2d = np.mean(test_data[:len(self.shap_values)], axis=1)
            else:
                shap_values_2d = self.shap_values
                test_data_2d = test_data[:len(self.shap_values)]

            # Ensure feature counts match
            num_features = min(shap_values_2d.shape[1], len(self.feature_names))
            shap_values_2d = shap_values_2d[:, :num_features]
            test_data_2d = test_data_2d[:, :num_features]
            feature_names_used = self.feature_names[:num_features]

            # Compute mean absolute SHAP values
            mean_abs_shap = np.mean(np.abs(shap_values_2d), axis=0)
            top_indices = np.argsort(mean_abs_shap)[-max_display:]

            # Create manual beeswarm plot
            plt.figure(figsize=(10, 8))
            y_pos = np.arange(len(top_indices))

            for i, feat_idx in enumerate(top_indices):
                feat_idx = int(feat_idx)  # Ensure scalar index

                feat_shap = shap_values_2d[:, feat_idx].flatten()
                feat_values = test_data_2d[:, feat_idx].flatten()

                if feat_shap.shape != feat_values.shape:
                    print(f"⚠ Shape mismatch for feature index {feat_idx}")
                    continue

                # Normalize for color mapping
                if feat_values.max() != feat_values.min():
                    feat_norm = (feat_values - feat_values.min()) / (feat_values.max() - feat_values.min())
                else:
                    feat_norm = np.ones_like(feat_values) * 0.5

                y_jitter = np.random.normal(loc=y_pos[i], scale=0.1, size=feat_shap.shape[0])

                scatter = plt.scatter(
                    feat_shap, y_jitter,
                    c=feat_norm, cmap='coolwarm',
                    alpha=0.6, s=20, edgecolors='none'
                )

            # Clean label rendering
            cleaned_indices = [int(i) if np.isscalar(i) else int(i.item()) for i in top_indices]
            plt.yticks(y_pos, [feature_names_used[i] for i in cleaned_indices])
            plt.xlabel('SHAP Value', fontsize=12)
            plt.ylabel('Features', fontsize=12)
            plt.title('SHAP Feature Importance Summary', fontsize=14, fontweight='bold')
            plt.colorbar(scatter, label='Feature Value (normalized)')
            plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
            plt.tight_layout()

            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✓ SHAP summary plot saved to {save_path}")

            plt.close()

        except Exception as e:
            print(f"⚠ Error plotting SHAP summary: {str(e)}")
            import traceback
            traceback.print_exc()
            plt.close()

    
    def plot_single_prediction(self,
                           sample_idx: int,
                           test_data: np.ndarray,
                           save_path: Optional[str] = None):
        """
        Plot explanation for a single prediction using SHAP values.

        Args:
            sample_idx: Index of sample to explain
            test_data: Test data array
            save_path: Path to save plot (optional)
        """
        def to_scalar_index(idx):
            """Robustly convert index to scalar int"""
            if isinstance(idx, np.ndarray):
                if idx.size == 1:
                    return int(idx.item())
                raise ValueError(f"Expected scalar index, got array of shape {idx.shape}")
            return int(idx)

        if self.shap_values is None:
            print(f"⚠ No SHAP values available for sample {sample_idx}")
            return

        if sample_idx >= len(self.shap_values):
            print(f"⚠ Sample index {sample_idx} out of range")
            return

        try:
            # Get SHAP values and input data
            if self.shap_values.ndim == 3:
                sample_shap = np.mean(self.shap_values[sample_idx], axis=0)
                sample_data_input = np.mean(test_data[sample_idx], axis=0)
            else:
                sample_shap = self.shap_values[sample_idx]
                sample_data_input = test_data[sample_idx].flatten()

            num_features = min(len(sample_shap), len(self.feature_names))
            sample_shap = sample_shap[:num_features]
            sample_data_input = sample_data_input[:num_features]
            feature_names_used = self.feature_names[:num_features]

            # Get model prediction
            with torch.no_grad():
                sample_tensor = torch.FloatTensor(test_data[sample_idx:sample_idx+1]).to(self.device)
                prediction = self.model(sample_tensor).cpu().numpy()[0, 0]

            # Get SHAP base value
            expected_val = self.explainer.expected_value
            if isinstance(expected_val, (list, np.ndarray)):
                expected_val = float(expected_val[0]) if len(expected_val) > 0 else 0.0
            else:
                expected_val = float(expected_val)

            print(f"\nSample {sample_idx}:")
            print(f"  Prediction: {prediction:.4f} ({'Gait' if prediction > 0.5 else 'Non-Gait'})")
            print(f"  Base value: {expected_val:.4f}")

            # Print top 3 contributing features
            abs_shap = np.abs(sample_shap)
            top_idx = np.argsort(abs_shap)[-3:][::-1]

            print("  Top 3 contributing features:")
            for idx in top_idx:
                idx_int = to_scalar_index(idx)
                print(f"    - {feature_names_used[idx_int]}: {sample_shap[idx_int]:.4f}")

            # Plot waterfall (top 10 features)
            plt.figure(figsize=(12, 6))

            top_n = min(10, len(sample_shap))
            top_indices = np.argsort(abs_shap)[-top_n:]
            sorted_indices = top_indices[np.argsort(sample_shap[top_indices])]

            shap_sorted = sample_shap[sorted_indices]
            features_sorted = [feature_names_used[to_scalar_index(i)] for i in sorted_indices]

            colors = ['red' if x < 0 else 'green' for x in shap_sorted]
            y_pos = np.arange(len(features_sorted))

            plt.barh(y_pos, shap_sorted, color=colors, alpha=0.7)
            plt.yticks(y_pos, features_sorted)
            plt.xlabel('SHAP Value (Impact on Prediction)', fontsize=11)
            plt.title(f'Top {top_n} Features for Sample {sample_idx}\n' +
                    f'Prediction: {prediction:.3f} ({"Gait" if prediction > 0.5 else "Non-Gait"})',
                    fontsize=13, fontweight='bold')
            plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            plt.tight_layout()

            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"  ✓ Waterfall plot saved to {save_path}")

            plt.close()

        except Exception as e:
            print(f"⚠ Error explaining sample {sample_idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            plt.close()

    '''
    def generate_explanation_report(self,
                                   test_data: np.ndarray,
                                   output_dir: str = 'results/shap_outputs'):
        """
        Generate comprehensive SHAP explanation report with multiple visualizations.
        
        Args:
            test_data: Test data for explanation
            output_dir: Directory to save outputs
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print("GENERATING SHAP EXPLANATION REPORT")
        print(f"{'='*60}\n")
        
        if self.shap_values is None:
            print("⚠ SHAP values not available. Skipping SHAP report generation.")
            print(f"{'='*60}\n")
            return
        
        try:
            # 1. Feature importance bar plot
            print("[1/3] Creating feature importance plot...")
            self.plot_feature_importance(
                top_k=20,
                save_path=output_path / 'feature_importance.png'
            )
            
            # 2. Summary plot
            print("[2/3] Creating summary plot...")
            self.plot_shap_summary(
                test_data=test_data,
                max_display=20,
                save_path=output_path / 'shap_summary.png'
            )
            
            # 3. Individual prediction explanations
            print("[3/3] Creating waterfall plots for sample predictions...")
            num_plots = min(3, len(test_data))
            for i in range(num_plots):
                self.plot_single_prediction(
                    sample_idx=i,
                    test_data=test_data,
                    save_path=output_path / f'waterfall_sample_{i}.png'
                )
            
            # Save feature importance to text file
            importance_dict = self.get_feature_importance()
            with open(output_path / 'feature_importance.txt', 'w') as f:
                f.write("Feature Importance Rankings\n")
                f.write("="*60 + "\n\n")
                for rank, (feature, importance) in enumerate(importance_dict.items(), 1):
                    f.write(f"{rank:3d}. {feature:40s} {importance:.6f}\n")
            
            print(f"\n{'='*60}")
            print(f"✓ SHAP report generated successfully!")
            print(f"Output directory: {output_path}")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"⚠ Error generating SHAP report: {str(e)}")
            print(f"  Some visualizations may not be available.")
            print(f"{'='*60}\n")

    '''
    
    def save_shap_values(self, filepath: str):
        """Save computed SHAP values to disk."""
        if self.shap_values is None:
            print("⚠ No SHAP values to save")
            return
        
        try:
            np.save(filepath, self.shap_values)
            print(f"✓ SHAP values saved to {filepath}")
        except Exception as e:
            print(f"⚠ Error saving SHAP values: {str(e)}")
    
    def load_shap_values(self, filepath: str):
        """Load SHAP values from disk."""
        try:
            self.shap_values = np.load(filepath)
            print(f"✓ SHAP values loaded from {filepath}")
        except Exception as e:
            print(f"⚠ Error loading SHAP values: {str(e)}")
            self.shap_values = None
            