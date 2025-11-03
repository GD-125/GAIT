# explainability/unified_explainer.py
"""
Unified Explainability Module for Gait Detection
Provides easy interface to switch between SHAP and LIME
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional
from pathlib import Path

# Import both explainers
try:
    from shap_explainer_fixed import GaitSHAPExplainer
    SHAP_AVAILABLE = True
except ImportError:
    print("⚠ SHAP explainer not available")
    SHAP_AVAILABLE = False

try:
    from lime_explainer import GaitLIMEExplainer
    LIME_AVAILABLE = True
except ImportError:
    print("⚠ LIME explainer not available")
    LIME_AVAILABLE = False


class UnifiedGaitExplainer:
    """
    Unified explainability interface supporting both SHAP and LIME.
    Automatically falls back to LIME if SHAP fails.
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 method: str = 'auto',
                 feature_names: Optional[List[str]] = None):
        """
        Initialize unified explainer.
        
        Args:
            model: Trained PyTorch model
            device: Device model is on
            method: 'shap', 'lime', or 'auto' (tries SHAP first, falls back to LIME)
            feature_names: List of feature names
        """
        self.model = model
        self.device = device
        self.feature_names = feature_names
        self.method = method.lower()
        
        self.explainer = None
        self.active_method = None
        
        print(f"\n{'='*60}")
        print("INITIALIZING UNIFIED EXPLAINER")
        print(f"{'='*60}")
        print(f"Requested method: {self.method.upper()}")
        
        if self.method == 'shap':
            self._initialize_shap()
        elif self.method == 'lime':
            self._initialize_lime()
        elif self.method == 'auto':
            # Try SHAP first, fall back to LIME
            if self._initialize_shap():
                print("✓ Using SHAP explainer")
            else:
                print("→ Falling back to LIME explainer")
                self._initialize_lime()
        else:
            raise ValueError(f"Unknown method: {method}. Use 'shap', 'lime', or 'auto'")
        
        print(f"Active method: {self.active_method.upper()}")
        print(f"{'='*60}\n")
    
    def _initialize_shap(self) -> bool:
        """Initialize SHAP explainer. Returns True if successful."""
        if not SHAP_AVAILABLE:
            print("⚠ SHAP not available")
            return False
        
        try:
            self.explainer = GaitSHAPExplainer(
                model=self.model,
                device=self.device,
                feature_names=self.feature_names
            )
            self.active_method = 'shap'
            return True
        except Exception as e:
            print(f"⚠ Failed to initialize SHAP: {str(e)}")
            return False
    
    def _initialize_lime(self) -> bool:
        """Initialize LIME explainer. Returns True if successful."""
        if not LIME_AVAILABLE:
            print("⚠ LIME not available")
            return False
        
        try:
            self.explainer = GaitLIMEExplainer(
                model=self.model,
                device=self.device,
                feature_names=self.feature_names
            )
            self.active_method = 'lime'
            return True
        except Exception as e:
            print(f"⚠ Failed to initialize LIME: {str(e)}")
            return False
    
    def create_explainer(self, background_data: np.ndarray, **kwargs):
        """
        Create/initialize the explainer with background data.
        
        Args:
            background_data: Background dataset
            **kwargs: Method-specific parameters
        """
        if self.active_method == 'shap':
            self.explainer.create_explainer(
                background_data=background_data,
                max_background_samples=kwargs.get('max_background_samples', 100)
            )
        elif self.active_method == 'lime':
            # LIME doesn't need explicit background data initialization
            print("✓ LIME explainer ready (no background data needed)")
    
    def explain_samples(self,
                       test_data: np.ndarray,
                       num_samples: Optional[int] = None,
                       **kwargs) -> any:
        """
        Generate explanations for test samples.
        
        Args:
            test_data: Test data to explain
            num_samples: Number of samples to explain
            **kwargs: Method-specific parameters
            
        Returns:
            Explanations (format depends on method)
        """
        if self.active_method == 'shap':
            return self.explainer.explain_samples(
                test_data=test_data,
                num_samples=num_samples
            )
        elif self.active_method == 'lime':
            return self.explainer.explain_samples(
                test_data=test_data,
                num_samples_to_explain=num_samples or 10,
                num_perturbations=kwargs.get('num_perturbations', 1000)
            )
    
    def get_feature_importance(self) -> dict:
        """Get global feature importance rankings."""
        return self.explainer.get_feature_importance()
    
    def plot_feature_importance(self, top_k: int = 20, save_path: Optional[str] = None):
        """Plot feature importance."""
        self.explainer.plot_feature_importance(top_k=top_k, save_path=save_path)
    
    def generate_report(self,
                       test_data: np.ndarray,
                       output_dir: str = 'results/explainability',
                       **kwargs):
        """
        Generate comprehensive explanation report.
        
        Args:
            test_data: Test data
            output_dir: Output directory
            **kwargs: Method-specific parameters
        """
        # Modify output directory to include method name
        output_path = Path(output_dir) / self.active_method
        
        if self.active_method == 'shap':
            self.explainer.generate_explanation_report(
                test_data=test_data,
                output_dir=str(output_path)
            )
        elif self.active_method == 'lime':
            self.explainer.generate_explanation_report(
                test_data=test_data,
                num_samples_to_explain=kwargs.get('num_samples', 10),
                output_dir=str(output_path)
            )
    
    def compare_methods(self,
                       background_data: np.ndarray,
                       test_data: np.ndarray,
                       output_dir: str = 'results/explainability/comparison'):
        """
        Generate reports using both SHAP and LIME for comparison.
        
        Args:
            background_data: Background data for SHAP
            test_data: Test data to explain
            output_dir: Output directory
        """
        print(f"\n{'='*60}")
        print("COMPARING SHAP AND LIME")
        print(f"{'='*60}\n")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # Try SHAP
        if SHAP_AVAILABLE:
            print("[1/2] Generating SHAP explanations...")
            try:
                shap_explainer = GaitSHAPExplainer(
                    model=self.model,
                    device=self.device,
                    feature_names=self.feature_names
                )
                shap_explainer.create_explainer(background_data)
                shap_explainer.explain_samples(test_data, num_samples=10)
                shap_explainer.generate_explanation_report(
                    test_data=test_data,
                    output_dir=str(output_path / 'shap')
                )
                results['shap'] = shap_explainer.get_feature_importance()
                print("✓ SHAP completed")
            except Exception as e:
                print(f"⚠ SHAP failed: {str(e)}")
                results['shap'] = None
        
        # Try LIME
        if LIME_AVAILABLE:
            print("[2/2] Generating LIME explanations...")
            try:
                lime_explainer = GaitLIMEExplainer(
                    model=self.model,
                    device=self.device,
                    feature_names=self.feature_names
                )
                lime_explainer.generate_explanation_report(
                    test_data=test_data,
                    num_samples_to_explain=10,
                    output_dir=str(output_path / 'lime')
                )
                results['lime'] = lime_explainer.get_feature_importance()
                print("✓ LIME completed")
            except Exception as e:
                print(f"⚠ LIME failed: {str(e)}")
                results['lime'] = None
        
        # Create comparison report
        if results.get('shap') and results.get('lime'):
            self._create_comparison_report(results, output_path)
        
        print(f"\n{'='*60}")
        print(f"✓ Comparison complete!")
        print(f"Results saved to: {output_path}")
        print(f"{'='*60}\n")
        
        return results
    
    def _create_comparison_report(self, results: dict, output_path: Path):
        """Create a comparison report between SHAP and LIME."""
        import matplotlib.pyplot as plt
        
        shap_importance = results['shap']
        lime_importance = results['lime']
        
        # Get common features
        common_features = set(shap_importance.keys()) & set(lime_importance.keys())
        
        if not common_features:
            return
        
        # Create comparison plot
        top_n = 15
        
        # Get top features from SHAP
        shap_top = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        shap_features, shap_values = zip(*shap_top)
        
        # Get corresponding LIME values
        lime_values = [lime_importance.get(f, 0) for f in shap_features]
        
        # Normalize for comparison
        shap_norm = np.array(shap_values) / max(shap_values)
        lime_norm = np.array(lime_values) / max(lime_values) if max(lime_values) > 0 else np.zeros_like(lime_values)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(shap_features))
        width = 0.35
        
        ax.barh(x - width/2, shap_norm, width, label='SHAP', color='steelblue', alpha=0.8)
        ax.barh(x + width/2, lime_norm, width, label='LIME', color='coral', alpha=0.8)
        
        ax.set_yticks(x)
        ax.set_yticklabels(shap_features, fontsize=9)
        ax.set_xlabel('Normalized Importance', fontsize=11)
        ax.set_title('Feature Importance Comparison: SHAP vs LIME', 
                    fontsize=13, fontweight='bold')
        ax.legend()
        ax.invert_yaxis()
        plt.tight_layout()
        
        plt.savefig(output_path / 'shap_vs_lime_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Comparison plot saved to {output_path / 'shap_vs_lime_comparison.png'}")
        
        # Save text comparison
        with open(output_path / 'comparison_report.txt', 'w') as f:
            f.write("="*60 + "\n")
            f.write("SHAP vs LIME COMPARISON REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write("TOP 15 FEATURES COMPARISON\n")
            f.write("-"*60 + "\n")
            f.write(f"{'Feature':<40} {'SHAP':>10} {'LIME':>10}\n")
            f.write("-"*60 + "\n")
            
            for feature in shap_features:
                shap_val = shap_importance.get(feature, 0)
                lime_val = lime_importance.get(feature, 0)
                f.write(f"{feature:<40} {shap_val:>10.6f} {lime_val:>10.6f}\n")


# Example usage and integration script
def explain_model(model: nn.Module,
                 device: torch.device,
                 background_data: np.ndarray,
                 test_data: np.ndarray,
                 method: str = 'auto',
                 output_dir: str = 'results/explainability',
                 feature_names: Optional[List[str]] = None):
    """
    Convenient function to explain a model.
    
    Args:
        model: Trained model
        device: Device
        background_data: Background/training data for SHAP
        test_data: Test data to explain
        method: 'shap', 'lime', or 'auto'
        output_dir: Output directory
        feature_names: Feature names
        
    Returns:
        Explainer instance
    """
    # Initialize explainer
    explainer = UnifiedGaitExplainer(
        model=model,
        device=device,
        method=method,
        feature_names=feature_names
    )
    
    # Create explainer (for SHAP)
    explainer.create_explainer(background_data)
    
    # Generate explanations
    explainer.explain_samples(test_data, num_samples=10)
    
    # Generate report
    explainer.generate_report(test_data, output_dir=output_dir)
    
    return explainer


if __name__ == "__main__":
    # Example usage
    print("Unified Explainer Example")
    print("="*60)
    print("\nUsage Options:")
    print("\n1. Auto mode (tries SHAP, falls back to LIME):")
    print("   explainer = UnifiedGaitExplainer(model, device, method='auto')")
    print("\n2. Force SHAP:")
    print("   explainer = UnifiedGaitExplainer(model, device, method='shap')")
    print("\n3. Force LIME:")
    print("   explainer = UnifiedGaitExplainer(model, device, method='lime')")
    print("\n4. Compare both methods:")
    print("   explainer.compare_methods(background_data, test_data)")
    print("\n5. Quick explain function:")
    print("   explain_model(model, device, bg_data, test_data, method='auto')")
    print("="*60)