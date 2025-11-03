# src/dashboard/inference_handler.py
"""
Inference handler for dashboard - processes uploaded files
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import sys
import io

# Add parent directories to path
dashboard_dir = Path(__file__).parent
src_dir = dashboard_dir.parent
project_dir = src_dir.parent
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(project_dir))
sys.path.insert(0, str(dashboard_dir))

from models import CNN_BiLSTM_GaitDetector, get_device
from preprocessing import GaitDataLoader, DataCleaner, SignalFilter, DataNormalizer, DataSegmenter
from lime_explainer_dash import DashboardLIMEExplainer


class InferenceHandler:
    """Handle inference for uploaded files in dashboard."""
    
    def __init__(self,
                 model_path: str,
                 device: Optional[str] = None):
        """
        Initialize inference handler.
        
        Args:
            model_path: Path to trained model
            device: Device to use ('cpu', 'cuda', or None for auto)
        """
        # Set device
        if device:
            self.device = torch.device(device)
        else:
            self.device = get_device()
        
        # Load model
        self.model = CNN_BiLSTM_GaitDetector(input_features=38, seq_length=128)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        # ALWAYS create fresh normalizer for inference
        # (Training normalizer may have been fitted on data WITH label column)
        self.normalizer = DataNormalizer(method='zscore')
        self.normalizer_fitted = False
        
        # Initialize preprocessing components
        self.cleaner = DataCleaner()
        self.filter = SignalFilter(sampling_rate=100.0)
        self.segmenter = DataSegmenter(window_size=128, overlap=0.5)
        
        # LIME explainer
        self.lime_explainer = DashboardLIMEExplainer(
            model=self.model,
            device=self.device
        )
    
    def preprocess_data(self, 
                       df: pd.DataFrame,
                       sensor_columns: Dict) -> np.ndarray:
        """
        Preprocess uploaded dataframe.
        
        Args:
            df: Uploaded data as pandas DataFrame (must have exactly 38 columns)
            sensor_columns: Sensor column mappings
            
        Returns:
            Windowed features ready for inference
        """
        # Verify shape
        if len(df.columns) != 38:
            raise ValueError(f"Expected exactly 38 feature columns, got {len(df.columns)}")
        
        # Convert to numpy
        features = df.values
        
        print(f"  Input shape: {features.shape}")
        
        # Clean
        features, _ = self.cleaner.clean_data(features)
        print(f"  After cleaning: {features.shape}")
        
        # Filter
        if sensor_columns:
            features = self.filter.apply_sensor_specific_filters(features, sensor_columns)
            print(f"  After filtering: {features.shape}")
        
        # Normalize - ALWAYS fit fresh on this data
        print(f"  Fitting normalizer on {features.shape[0]} samples with {features.shape[1]} features...")
        features = self.normalizer.fit_transform(features)
        self.normalizer_fitted = True
        print(f"  After normalization: {features.shape}")
        
        # Segment into windows
        dummy_labels = np.zeros(len(features))
        windowed_features, _, _ = self.segmenter.segment_data(features, dummy_labels)
        print(f"  After windowing: {windowed_features.shape}")
        
        return windowed_features
    
    def predict(self, windowed_data: np.ndarray) -> np.ndarray:
        """
        Run inference on windowed data.
        
        Args:
            windowed_data: Preprocessed windows
            
        Returns:
            Predictions array
        """
        predictions = []
        batch_size = 32
        
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(windowed_data), batch_size):
                batch = windowed_data[i:i + batch_size]
                batch_tensor = torch.FloatTensor(batch).to(self.device)
                
                logits = self.model(batch_tensor)  # Model returns logits
                probabilities = torch.sigmoid(logits)  # Apply sigmoid to get probabilities
                preds = (probabilities.cpu().numpy() > 0.5).astype(int).flatten()
                predictions.extend(preds)
        
        return np.array(predictions)
    
    def predict_from_file(self,
                         uploaded_file,
                         include_lime: bool = True,
                         num_lime_samples: int = 5) -> Dict:
        """
        Complete prediction pipeline from uploaded file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            include_lime: Whether to include LIME explanations
            num_lime_samples: Number of samples to explain with LIME
            
        Returns:
            Dictionary with predictions and explanations
        """
        # Read uploaded file
        df = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)  # Reset file pointer
        
        # Remove label column if present (expected 38 features only)
        if 'label' in df.columns:
            df = df.drop('label', axis=1)
        
        # Ensure exactly 38 features
        if len(df.columns) != 38:
            raise ValueError(f"Expected 38 feature columns, but got {len(df.columns)}. "
                           f"Please ensure CSV has exactly 38 sensor features (no label column).")
        
        # Create temporary data loader to get sensor columns
        temp_loader = GaitDataLoader(".")
        sensor_columns = temp_loader.sensor_columns
        
        # Preprocess
        windowed_data = self.preprocess_data(df, sensor_columns)
        
        # Predict
        predictions = self.predict(windowed_data)
        
        # Calculate metrics
        num_gait = int(np.sum(predictions == 1))
        num_non_gait = int(np.sum(predictions == 0))
        
        results = {
            'predictions': predictions,
            'num_windows': len(predictions),
            'num_gait': num_gait,
            'num_non_gait': num_non_gait,
        }
        
        # Generate LIME explanations
        if include_lime and len(windowed_data) > 0:
            num_to_explain = min(num_lime_samples, len(windowed_data))
            
            lime_results = self.lime_explainer.explain_batch(
                windowed_data[:num_to_explain],
                num_samples=num_to_explain
            )
            
            results['lime_explanations'] = lime_results
        
        return results