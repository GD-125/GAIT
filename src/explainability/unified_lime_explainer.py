"""
Unified LIME Explainer for Gait Analysis System

Provides LIME (Local Interpretable Model-agnostic Explanations) for:
1. Gait Detection (Binary Classification)
2. Disease Classification (Multi-class)
3. Severity Estimation (Regression)

LIME helps users understand:
- Which features influenced the prediction
- How features contributed to the decision
- Why the model made a specific prediction
"""

import numpy as np
import torch
import torch.nn as nn
from lime import lime_tabular
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json


class UnifiedLIMEExplainer:
    """Unified LIME explainer for all model stages."""

    def __init__(self,
                 gait_model: Optional[nn.Module] = None,
                 disease_model: Optional[nn.Module] = None,
                 severity_model: Optional[nn.Module] = None,
                 device: torch.device = None):
        """
        Initialize LIME explainer.

        Args:
            gait_model: Gait detection model
            disease_model: Disease classification model
            severity_model: Severity estimation model
            device: PyTorch device
        """
        self.gait_model = gait_model
        self.disease_model = disease_model
        self.severity_model = severity_model
        self.device = device if device is not None else torch.device('cpu')

        # Move models to device and set to eval mode
        if self.gait_model is not None:
            self.gait_model.to(self.device).eval()
        if self.disease_model is not None:
            self.disease_model.to(self.device).eval()
        if self.severity_model is not None:
            self.severity_model.to(self.device).eval()

        # Feature names for sensors
        self.feature_names = self._create_feature_names()

        # Disease names
        self.disease_names = [
            "Parkinson's Disease",
            "Huntington's Disease",
            "Cerebral Palsy",
            "Multiple Sclerosis",
            "Ataxia",
            "Normal/Healthy Gait"
        ]

    def _create_feature_names(self) -> List[str]:
        """Create human-readable feature names."""
        locations = ['R_Foot', 'R_Shin', 'R_Thigh', 'L_Foot', 'L_Shin', 'L_Thigh']
        sensors = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']

        features = []
        for loc in locations:
            for sensor in sensors:
                features.append(f"{loc}_{sensor}")

        features.extend(['EMG_Right', 'EMG_Left'])
        return features

    def _flatten_sequence(self, X: np.ndarray) -> np.ndarray:
        """
        Flatten sequence data for LIME.

        Args:
            X: Input data (n_samples, seq_length, features) or (seq_length, features)

        Returns:
            Flattened data (n_samples, seq_length * features) or (seq_length * features,)
        """
        if X.ndim == 3:
            return X.reshape(X.shape[0], -1)
        elif X.ndim == 2:
            return X.reshape(-1)
        else:
            return X

    def _unflatten_sequence(self, X_flat: np.ndarray, seq_length: int, num_features: int) -> np.ndarray:
        """
        Unflatten data back to sequence format.

        Args:
            X_flat: Flattened data
            seq_length: Sequence length
            num_features: Number of features

        Returns:
            Sequence data
        """
        if X_flat.ndim == 2:
            return X_flat.reshape(X_flat.shape[0], seq_length, num_features)
        else:
            return X_flat.reshape(seq_length, num_features)

    # ==================== Gait Detection Explainability ====================

    def explain_gait_detection(self,
                               sample: np.ndarray,
                               training_data: np.ndarray,
                               num_features: int = 20,
                               num_samples: int = 5000) -> Dict:
        """
        Explain gait detection prediction using LIME.

        Args:
            sample: Single sample to explain (seq_length, features)
            training_data: Background training data for LIME (n_samples, seq_length, features)
            num_features: Number of top features to show
            num_samples: Number of samples for LIME

        Returns:
            Dictionary with explanation data
        """
        if self.gait_model is None:
            raise ValueError("Gait model not provided")

        # Flatten data
        sample_flat = self._flatten_sequence(sample)
        training_flat = self._flatten_sequence(training_data)

        # Create prediction function
        def predict_fn(X_flat):
            X_seq = self._unflatten_sequence(X_flat, sample.shape[0], sample.shape[1])
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            with torch.no_grad():
                outputs = self.gait_model(X_tensor)
            return outputs.cpu().numpy().reshape(-1, 1)

        # Create LIME explainer
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=training_flat,
            mode='regression',
            feature_names=[f"T{t//38}_{self.feature_names[t%38]}" for t in range(len(sample_flat))],
            discretize_continuous=False
        )

        # Get explanation
        exp = explainer.explain_instance(
            data_row=sample_flat,
            predict_fn=predict_fn,
            num_features=num_features,
            num_samples=num_samples
        )

        # Get prediction
        prediction = predict_fn(sample_flat.reshape(1, -1))[0, 0]

        # Extract feature importance
        feature_importance = dict(exp.as_list())

        return {
            'prediction': prediction,
            'prediction_label': 'Gait' if prediction > 0.5 else 'Non-Gait',
            'confidence': prediction if prediction > 0.5 else 1 - prediction,
            'feature_importance': feature_importance,
            'explanation_object': exp
        }

    # ==================== Disease Classification Explainability ====================

    def explain_disease_classification(self,
                                       sample: np.ndarray,
                                       training_data: np.ndarray,
                                       num_features: int = 20,
                                       num_samples: int = 5000) -> Dict:
        """
        Explain disease classification prediction using LIME.

        Args:
            sample: Single sample to explain
            training_data: Background training data
            num_features: Number of top features to show
            num_samples: Number of samples for LIME

        Returns:
            Dictionary with explanation data
        """
        if self.disease_model is None:
            raise ValueError("Disease model not provided")

        # Flatten data
        sample_flat = self._flatten_sequence(sample)
        training_flat = self._flatten_sequence(training_data)

        # Create prediction function
        def predict_fn(X_flat):
            X_seq = self._unflatten_sequence(X_flat, sample.shape[0], sample.shape[1])
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            with torch.no_grad():
                outputs = self.disease_model(X_tensor)
            return outputs.cpu().numpy()

        # Create LIME explainer
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=training_flat,
            mode='classification',
            feature_names=[f"T{t//38}_{self.feature_names[t%38]}" for t in range(len(sample_flat))],
            class_names=self.disease_names,
            discretize_continuous=False
        )

        # Get explanation
        exp = explainer.explain_instance(
            data_row=sample_flat,
            predict_fn=predict_fn,
            num_features=num_features,
            num_samples=num_samples,
            top_labels=len(self.disease_names)
        )

        # Get prediction
        probabilities = predict_fn(sample_flat.reshape(1, -1))[0]
        predicted_class = np.argmax(probabilities)

        # Get feature importance for predicted class
        feature_importance = dict(exp.as_list(label=predicted_class))

        return {
            'predicted_class': int(predicted_class),
            'predicted_disease': self.disease_names[predicted_class],
            'probabilities': {
                self.disease_names[i]: float(probabilities[i])
                for i in range(len(self.disease_names))
            },
            'confidence': float(probabilities[predicted_class]),
            'feature_importance': feature_importance,
            'explanation_object': exp
        }

    # ==================== Severity Estimation Explainability ====================

    def explain_severity_estimation(self,
                                    sample: np.ndarray,
                                    training_data: np.ndarray,
                                    num_features: int = 20,
                                    num_samples: int = 5000) -> Dict:
        """
        Explain severity estimation prediction using LIME.

        Args:
            sample: Single sample to explain
            training_data: Background training data
            num_features: Number of top features to show
            num_samples: Number of samples for LIME

        Returns:
            Dictionary with explanation data
        """
        if self.severity_model is None:
            raise ValueError("Severity model not provided")

        # Flatten data
        sample_flat = self._flatten_sequence(sample)
        training_flat = self._flatten_sequence(training_data)

        # Create prediction function
        def predict_fn(X_flat):
            X_seq = self._unflatten_sequence(X_flat, sample.shape[0], sample.shape[1])
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            with torch.no_grad():
                outputs = self.severity_model(X_tensor)
            return (outputs.cpu().numpy() * 100).reshape(-1, 1)  # Convert to percentage

        # Create LIME explainer
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=training_flat,
            mode='regression',
            feature_names=[f"T{t//38}_{self.feature_names[t%38]}" for t in range(len(sample_flat))],
            discretize_continuous=False
        )

        # Get explanation
        exp = explainer.explain_instance(
            data_row=sample_flat,
            predict_fn=predict_fn,
            num_features=num_features,
            num_samples=num_samples
        )

        # Get prediction
        severity_percentage = predict_fn(sample_flat.reshape(1, -1))[0, 0]

        # Extract feature importance
        feature_importance = dict(exp.as_list())

        # Determine severity level
        if severity_percentage < 40:
            severity_level = "Mild"
        elif severity_percentage < 70:
            severity_level = "Moderate"
        else:
            severity_level = "Severe"

        return {
            'severity_percentage': float(severity_percentage),
            'severity_level': severity_level,
            'feature_importance': feature_importance,
            'explanation_object': exp
        }

    # ==================== Visualization Functions ====================

    def plot_feature_importance(self,
                                feature_importance: Dict,
                                title: str,
                                save_path: Optional[str] = None,
                                top_n: int = 15):
        """
        Plot feature importance from LIME explanation.

        Args:
            feature_importance: Dictionary of feature: importance
            title: Plot title
            save_path: Path to save plot
            top_n: Number of top features to show
        """
        # Sort by absolute importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_n]

        features, importances = zip(*sorted_features)

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))

        colors = ['green' if imp > 0 else 'red' for imp in importances]
        y_pos = np.arange(len(features))

        ax.barh(y_pos, importances, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=9)
        ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

        # Add grid
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")

        plt.close()

    def generate_explanation_report(self,
                                    sample: np.ndarray,
                                    training_data: np.ndarray,
                                    output_dir: str,
                                    sample_name: str = "sample"):
        """
        Generate comprehensive LIME explanation report for all stages.

        Args:
            sample: Sample to explain
            training_data: Background data
            output_dir: Output directory
            sample_name: Name for the sample
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*80)
        print("GENERATING LIME EXPLANATION REPORT")
        print("="*80)

        report = {}

        # 1. Gait Detection Explanation
        if self.gait_model is not None:
            print("\n[1/3] Explaining Gait Detection...")
            gait_exp = self.explain_gait_detection(sample, training_data)
            report['gait_detection'] = {
                'prediction': gait_exp['prediction_label'],
                'confidence': f"{gait_exp['confidence']:.2%}",
                'top_features': list(gait_exp['feature_importance'].items())[:10]
            }

            # Plot
            self.plot_feature_importance(
                gait_exp['feature_importance'],
                f"LIME Explanation: Gait Detection\nPrediction: {gait_exp['prediction_label']} (Confidence: {gait_exp['confidence']:.2%})",
                save_path=str(output_path / f"{sample_name}_gait_detection_lime.png")
            )

        # 2. Disease Classification Explanation
        if self.disease_model is not None:
            print("[2/3] Explaining Disease Classification...")
            disease_exp = self.explain_disease_classification(sample, training_data)
            report['disease_classification'] = {
                'predicted_disease': disease_exp['predicted_disease'],
                'confidence': f"{disease_exp['confidence']:.2%}",
                'all_probabilities': {k: f"{v:.2%}" for k, v in disease_exp['probabilities'].items()},
                'top_features': list(disease_exp['feature_importance'].items())[:10]
            }

            # Plot
            self.plot_feature_importance(
                disease_exp['feature_importance'],
                f"LIME Explanation: Disease Classification\nPredicted: {disease_exp['predicted_disease']} (Confidence: {disease_exp['confidence']:.2%})",
                save_path=str(output_path / f"{sample_name}_disease_classification_lime.png")
            )

        # 3. Severity Estimation Explanation
        if self.severity_model is not None:
            print("[3/3] Explaining Severity Estimation...")
            severity_exp = self.explain_severity_estimation(sample, training_data)
            report['severity_estimation'] = {
                'severity_percentage': f"{severity_exp['severity_percentage']:.1f}%",
                'severity_level': severity_exp['severity_level'],
                'top_features': list(severity_exp['feature_importance'].items())[:10]
            }

            # Plot
            self.plot_feature_importance(
                severity_exp['feature_importance'],
                f"LIME Explanation: Severity Estimation\nSeverity: {severity_exp['severity_percentage']:.1f}% ({severity_exp['severity_level']})",
                save_path=str(output_path / f"{sample_name}_severity_estimation_lime.png")
            )

        # Save report as JSON
        report_path = output_path / f"{sample_name}_lime_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)

        print(f"\n✓ LIME explanation report saved to {output_dir}")
        print("="*80 + "\n")

        return report


if __name__ == "__main__":
    print("="*80)
    print("UNIFIED LIME EXPLAINER - Test Mode")
    print("="*80)

    # This would normally use actual trained models
    print("\nNote: This is a template. Use with trained models for actual explanations.")
    print("\nExample usage:")
    print("""
    from src.models.cnn_bilstm import CNN_BiLSTM_GaitDetector
    from src.models.disease_classifier_new import DiseaseClassifier, SeverityEstimator
    from src.utils.device_manager import get_device

    # Load models
    device = get_device()
    gait_model = CNN_BiLSTM_GaitDetector(input_features=38, seq_length=128)
    disease_model = DiseaseClassifier(input_features=38, seq_length=128)
    severity_model = SeverityEstimator(input_features=38, seq_length=128)

    # Load checkpoints...

    # Create explainer
    explainer = UnifiedLIMEExplainer(
        gait_model=gait_model,
        disease_model=disease_model,
        severity_model=severity_model,
        device=device
    )

    # Generate explanations
    explainer.generate_explanation_report(
        sample=test_sample,
        training_data=X_train,
        output_dir='results/explainability/lime',
        sample_name='patient_001'
    )
    """)
    print("="*80)
