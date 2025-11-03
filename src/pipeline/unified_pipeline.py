"""
Unified Gait Analysis and Neurological Disease Classification Pipeline

Complete multi-stage pipeline:
1. Gait Detection (Gait vs Non-Gait)
2. Disease Classification (if gait detected)
3. Severity Estimation
4. LIME Explanations for all stages

This pipeline integrates all components into a seamless workflow.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union, List
import json
from datetime import datetime


class UnifiedGaitAnalysisPipeline:
    """
    Complete pipeline for gait analysis and disease classification.

    Workflow:
        Input → Preprocessing → Gait Detection
                                    ↓ (if gait)
                            Disease Classification → Severity Estimation
                                    ↓
                            LIME Explanations
    """

    def __init__(self,
                 gait_model_path: str,
                 disease_model_path: Optional[str] = None,
                 severity_model_path: Optional[str] = None,
                 device: Optional[torch.device] = None,
                 enable_explanations: bool = True):
        """
        Initialize unified pipeline.

        Args:
            gait_model_path: Path to trained gait detection model
            disease_model_path: Path to trained disease classification model
            severity_model_path: Path to trained severity estimation model
            device: PyTorch device
            enable_explanations: Whether to generate LIME explanations
        """
        self.device = device if device is not None else torch.device('cpu')
        self.enable_explanations = enable_explanations

        # Load models
        print("\n" + "="*80)
        print("INITIALIZING UNIFIED GAIT ANALYSIS PIPELINE")
        print("="*80)

        self.gait_model = self._load_gait_model(gait_model_path)
        self.disease_model = self._load_disease_model(disease_model_path) if disease_model_path else None
        self.severity_model = self._load_severity_model(severity_model_path) if severity_model_path else None

        # Initialize explainer if needed
        self.explainer = None
        if enable_explanations:
            try:
                from explainability.unified_lime_explainer import UnifiedLIMEExplainer
                self.explainer = UnifiedLIMEExplainer(
                    gait_model=self.gait_model,
                    disease_model=self.disease_model,
                    severity_model=self.severity_model,
                    device=self.device
                )
                print("✓ LIME Explainer initialized")
            except Exception as e:
                print(f"⚠ Could not initialize LIME explainer: {e}")

        print("="*80 + "\n")

    def _load_gait_model(self, model_path: str) -> nn.Module:
        """Load gait detection model."""
        from models.cnn_bilstm import CNN_BiLSTM_GaitDetector

        print(f"Loading gait detection model from {model_path}...")
        model = CNN_BiLSTM_GaitDetector(input_features=38, seq_length=128)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device).eval()

        print(f"✓ Gait model loaded (Accuracy: {checkpoint.get('accuracy', 'N/A')})")
        return model

    def _load_disease_model(self, model_path: str) -> Optional[nn.Module]:
        """Load disease classification model."""
        if not model_path or not Path(model_path).exists():
            print("⚠ Disease model not provided or not found")
            return None

        from models.disease_classifier_new import DiseaseClassifier

        print(f"Loading disease classification model from {model_path}...")
        model = DiseaseClassifier(input_features=38, seq_length=128, num_diseases=6)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device).eval()

        print(f"✓ Disease model loaded (Accuracy: {checkpoint.get('accuracy', 'N/A')})")
        return model

    def _load_severity_model(self, model_path: str) -> Optional[nn.Module]:
        """Load severity estimation model."""
        if not model_path or not Path(model_path).exists():
            print("⚠ Severity model not provided or not found")
            return None

        from models.disease_classifier_new import SeverityEstimator

        print(f"Loading severity estimation model from {model_path}...")
        model = SeverityEstimator(input_features=38, seq_length=128)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device).eval()

        print("✓ Severity model loaded")
        return model

    def analyze_sample(self,
                      sample: np.ndarray,
                      generate_explanations: bool = True,
                      training_data: Optional[np.ndarray] = None,
                      output_dir: Optional[str] = None) -> Dict:
        """
        Analyze a single sample through the complete pipeline.

        Args:
            sample: Input sample (seq_length, features) or (batch, seq_length, features)
            generate_explanations: Whether to generate LIME explanations
            training_data: Background data for LIME (required if generating explanations)
            output_dir: Directory to save explanation plots

        Returns:
            Dictionary with complete analysis results
        """
        # Ensure sample has correct shape
        if sample.ndim == 2:
            sample = sample[np.newaxis, :]  # Add batch dimension

        # Convert to tensor
        sample_tensor = torch.FloatTensor(sample).to(self.device)

        results = {
            'timestamp': datetime.now().isoformat(),
            'input_shape': sample.shape
        }

        # ==================== Stage 1: Gait Detection ====================
        print("\n" + "-"*80)
        print("[STAGE 1] GAIT DETECTION")
        print("-"*80)

        with torch.no_grad():
            gait_output = self.gait_model(sample_tensor)
            gait_prob = gait_output[0].item()

        is_gait = gait_prob > 0.5
        results['gait_detection'] = {
            'is_gait': bool(is_gait),
            'probability': float(gait_prob),
            'confidence': float(gait_prob if is_gait else 1 - gait_prob),
            'label': 'Gait' if is_gait else 'Non-Gait'
        }

        print(f"  Result: {results['gait_detection']['label']}")
        print(f"  Confidence: {results['gait_detection']['confidence']:.2%}")

        # If not gait, stop here
        if not is_gait:
            print("\n⚠ Non-gait detected. Pipeline terminated.")
            return results

        # ==================== Stage 2: Disease Classification ====================
        if self.disease_model is not None:
            print("\n" + "-"*80)
            print("[STAGE 2] DISEASE CLASSIFICATION")
            print("-"*80)

            with torch.no_grad():
                disease_probs = self.disease_model(sample_tensor)[0]
                predicted_class = torch.argmax(disease_probs).item()
                confidence = disease_probs[predicted_class].item()

            disease_names = [
                "Parkinson's Disease",
                "Huntington's Disease",
                "Cerebral Palsy",
                "Multiple Sclerosis",
                "Ataxia",
                "Normal/Healthy Gait"
            ]

            results['disease_classification'] = {
                'predicted_class': int(predicted_class),
                'predicted_disease': disease_names[predicted_class],
                'confidence': float(confidence),
                'all_probabilities': {
                    disease_names[i]: float(disease_probs[i].item())
                    for i in range(len(disease_names))
                }
            }

            print(f"  Predicted Disease: {results['disease_classification']['predicted_disease']}")
            print(f"  Confidence: {results['disease_classification']['confidence']:.2%}")
            print(f"\n  All Probabilities:")
            for disease, prob in results['disease_classification']['all_probabilities'].items():
                print(f"    - {disease}: {prob:.2%}")

        # ==================== Stage 3: Severity Estimation ====================
        if self.severity_model is not None:
            print("\n" + "-"*80)
            print("[STAGE 3] SEVERITY ESTIMATION")
            print("-"*80)

            with torch.no_grad():
                severity = self.severity_model(sample_tensor)[0].item() * 100  # Convert to percentage

            # Determine severity level
            if severity < 40:
                severity_level = "Mild"
            elif severity < 70:
                severity_level = "Moderate"
            else:
                severity_level = "Severe"

            results['severity_estimation'] = {
                'severity_percentage': float(severity),
                'severity_level': severity_level
            }

            print(f"  Severity: {results['severity_estimation']['severity_percentage']:.1f}%")
            print(f"  Level: {results['severity_estimation']['severity_level']}")

        # ==================== Stage 4: LIME Explanations ====================
        if generate_explanations and self.explainer is not None and training_data is not None:
            print("\n" + "-"*80)
            print("[STAGE 4] GENERATING LIME EXPLANATIONS")
            print("-"*80)

            try:
                explanation_report = self.explainer.generate_explanation_report(
                    sample=sample[0],  # Remove batch dimension
                    training_data=training_data,
                    output_dir=output_dir if output_dir else 'results/explainability/lime',
                    sample_name='sample_analysis'
                )
                results['explanations'] = explanation_report
            except Exception as e:
                print(f"⚠ Error generating explanations: {e}")
                results['explanations'] = None

        return results

    def analyze_batch(self,
                     samples: np.ndarray,
                     batch_size: int = 32,
                     show_progress: bool = True) -> List[Dict]:
        """
        Analyze multiple samples through the pipeline.

        Args:
            samples: Input samples (n_samples, seq_length, features)
            batch_size: Batch size for processing
            show_progress: Show progress bar

        Returns:
            List of results for each sample
        """
        results = []

        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(range(0, len(samples), batch_size), desc="Processing batches")
        else:
            iterator = range(0, len(samples), batch_size)

        for i in iterator:
            batch = samples[i:i+batch_size]

            for sample in batch:
                result = self.analyze_sample(
                    sample=sample,
                    generate_explanations=False  # Skip explanations for batch processing
                )
                results.append(result)

        return results

    def save_results(self, results: Union[Dict, List[Dict]], output_path: str):
        """
        Save analysis results to JSON file.

        Args:
            results: Single result or list of results
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"\n✓ Results saved to {output_path}")

    def print_summary(self, results: Dict):
        """Print a formatted summary of analysis results."""
        print("\n" + "="*80)
        print("ANALYSIS SUMMARY")
        print("="*80)

        # Gait Detection
        gait = results['gait_detection']
        print(f"\n1. GAIT DETECTION: {gait['label']}")
        print(f"   Confidence: {gait['confidence']:.2%}")

        # Disease Classification (if available)
        if 'disease_classification' in results:
            disease = results['disease_classification']
            print(f"\n2. DISEASE CLASSIFICATION: {disease['predicted_disease']}")
            print(f"   Confidence: {disease['confidence']:.2%}")

        # Severity (if available)
        if 'severity_estimation' in results:
            severity = results['severity_estimation']
            print(f"\n3. SEVERITY ESTIMATION: {severity['severity_percentage']:.1f}%")
            print(f"   Level: {severity['severity_level']}")

        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    print("="*80)
    print("UNIFIED GAIT ANALYSIS PIPELINE - Test Mode")
    print("="*80)

    print("\nExample Usage:")
    print("""
    from src.utils.device_manager import get_device

    # Initialize pipeline
    device = get_device(prefer_gpu=True)
    pipeline = UnifiedGaitAnalysisPipeline(
        gait_model_path='checkpoints/gait_detection/best_model.pt',
        disease_model_path='checkpoints/disease_classification/best_model.pt',
        severity_model_path='checkpoints/severity_estimation/best_model.pt',
        device=device,
        enable_explanations=True
    )

    # Analyze single sample
    sample = np.random.randn(128, 38)  # Example data
    training_data = np.random.randn(100, 128, 38)  # For LIME

    results = pipeline.analyze_sample(
        sample=sample,
        generate_explanations=True,
        training_data=training_data,
        output_dir='results/explainability/lime'
    )

    # Print summary
    pipeline.print_summary(results)

    # Save results
    pipeline.save_results(results, 'results/analysis/sample_001.json')
    """)

    print("="*80)
