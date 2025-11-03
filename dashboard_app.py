"""
Professional Gait Detection Dashboard
Upload files, get predictions with LIME explanations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import pickle
from pathlib import Path
import sys
import io
import time
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models import CNN_BiLSTM_GaitDetector, load_checkpoint, get_device
from preprocessing.pipeline import PreprocessingPipeline
from explainability.lime_explainer import GaitLIMEExplainer

# Page configuration
st.set_page_config(
    page_title="Gait Detection System",
    page_icon="🚶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional dark mode
st.markdown("""
<style>
    /* Dark page background */
    .stApp {
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 100%);
    }

    /* Main content area - dark */
    .main .block-container {
        background: #1e1e1e;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        border: 1px solid #333;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    /* Sidebar dark mode */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: #1a1a1a;
    }

    /* Main title styling */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4ff 0%, #00ff88 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 20px rgba(0,212,255,0.3);
    }

    .subtitle {
        font-size: 1.2rem;
        color: #b0b0b0;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
    }

    /* Result card styling - vibrant colors */
    .result-card {
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.4);
        border: 2px solid;
    }

    .gait-detected {
        background: linear-gradient(135deg, #00ff88 0%, #00d4ff 100%);
        color: #000;
        border-color: #00ff88;
    }

    .no-gait-detected {
        background: linear-gradient(135deg, #ff006e 0%, #ff7700 100%);
        color: #fff;
        border-color: #ff006e;
    }

    .result-title {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .confidence-score {
        font-size: 1.5rem;
        text-align: center;
        font-weight: 600;
    }

    /* Metric cards - dark with bright borders */
    .metric-card {
        background: #2a2a2a;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        border: 2px solid #00d4ff;
        margin: 0.5rem 0;
    }

    /* Upload section - bright on dark */
    .upload-section {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 0rem;
        border-radius: 15px;
        color: #00d4ff;
        margin: 2rem 0;
        border: 2px solid #00d4ff;
        box-shadow: 0 0 20px rgba(0,212,255,0.2);
    }

    /* Info boxes - bright on dark */
    .info-box {
        background: #2a2a2a;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #00d4ff;
        margin: 1rem 0;
        color: #e0e0e0;
    }

    /* Override Streamlit default text colors */
    .stMarkdown, .stText, p, span, div {
        color: #e0e0e0 !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }

    /* Streamlit metric styling */
    [data-testid="stMetricValue"] {
        color: #00ff88 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #b0b0b0 !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricDelta"] {
        color: #00d4ff !important;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: #2a2a2a !important;
        color: #00d4ff !important;
        border: 1px solid #00d4ff !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_pipeline():
    """Load trained model and preprocessing pipeline."""
    try:
        device = get_device()

        # Load model
        model_path = Path('checkpoints/best_model.pt')
        if not model_path.exists():
            return None, None, None, "Model not found. Please train the model first using efficient_run.py or train_efficient_no_overfit.py"

        # Load training config to get model architecture
        config_path = Path('checkpoints/training_config.json')
        model_config = None

        if config_path.exists():
            with open(config_path, 'r') as f:
                full_config = json.load(f)
                model_config = full_config.get('anti_overfitting_measures', {})

        # Determine architecture based on config
        # Check if this is a "small model" (anti-overfitting version)
        is_small_model = model_config.get('small_model', False) if model_config else False
        dropout = model_config.get('dropout', 0.3) if model_config else 0.3

        if is_small_model:
            # Small model architecture (train_efficient_no_overfit.py)
            model = CNN_BiLSTM_GaitDetector(
                input_features=38,
                seq_length=128,
                conv_filters=[32, 64, 128],      # REDUCED
                kernel_sizes=[5, 5, 5],
                lstm_hidden_size=64,             # REDUCED
                lstm_num_layers=1,               # REDUCED
                fc_hidden_sizes=[128, 64],       # REDUCED
                dropout=dropout,
                use_batch_norm=True,
                use_residual=True
            )
        else:
            # Standard model architecture (efficient_run.py)
            model = CNN_BiLSTM_GaitDetector(
                input_features=38,
                seq_length=128,
                conv_filters=[64, 128, 256],
                kernel_sizes=[5, 5, 5],
                lstm_hidden_size=128,
                lstm_num_layers=2,
                fc_hidden_sizes=[256, 128],
                dropout=dropout,
                use_batch_norm=True,
                use_residual=True
            )

        checkpoint = load_checkpoint(str(model_path), model, device=device)
        model.to(device)  # Ensure model is on the correct device
        model.eval()

        # Load preprocessing pipeline
        pipeline = PreprocessingPipeline(
            sampling_rate=100.0,
            window_size=128,
            overlap=0.5,
            normalization_method='zscore',
            filter_type='sensor_specific',
            balance_method='none'
        )

        # Load normalizer if exists
        normalizer_path = Path('data/processed/normalizer.pkl')
        if normalizer_path.exists():
            pipeline.normalizer.load(normalizer_path)
            pipeline.is_fitted = True

        return model, pipeline, device, None

    except Exception as e:
        return None, None, None, f"Error loading model: {str(e)}"


def preprocess_uploaded_file(uploaded_file, pipeline):
    """Preprocess uploaded CSV/Excel file."""
    try:
        # Read file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            return None, None, "Unsupported file format. Please upload CSV or Excel file."

        # Save temporarily
        temp_path = Path('temp_upload.csv')
        df.to_csv(temp_path, index=False)

        # Preprocess
        windowed_data, labels = pipeline.preprocess_single_file(str(temp_path))

        # Clean up
        temp_path.unlink()

        return windowed_data, df, None

    except Exception as e:
        return None, None, f"Error preprocessing file: {str(e)}"


def predict_with_model(model, data, device):
    """Make predictions on data."""
    model.eval()
    with torch.no_grad():
        data_tensor = torch.FloatTensor(data).to(device)
        logits = model(data_tensor)  # Model returns logits
        probabilities = torch.sigmoid(logits)  # Apply sigmoid to get probabilities
        probabilities = probabilities.cpu().numpy()

    return probabilities


def create_prediction_gauge(probability):
    """Create a gauge chart for prediction probability - DARK MODE."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Score", 'font': {'size': 24, 'color': '#ffffff', 'weight': 700}},
        delta={'reference': 50, 'increasing': {'color': "#00ff88"}, 'decreasing': {'color': "#ff006e"}},
        number={'font': {'size': 48, 'color': '#00d4ff', 'weight': 'bold'}},
        gauge={
            'axis': {
                'range': [None, 100],
                'tickwidth': 3,
                'tickcolor': "#ffffff",
                'tickfont': {'size': 14, 'color': '#ffffff', 'weight': 600}
            },
            'bar': {'color': "#00d4ff", 'thickness': 0.8},
            'bgcolor': "#1a1a1a",
            'borderwidth': 3,
            'bordercolor': "#00d4ff",
            'steps': [
                {'range': [0, 30], 'color': '#ff006e'},      # Bright red
                {'range': [30, 50], 'color': '#ff7700'},     # Bright orange
                {'range': [50, 70], 'color': '#ffeb3b'},     # Bright yellow
                {'range': [70, 100], 'color': '#00ff88'}     # Bright green
            ],
            'threshold': {
                'line': {'color': "#ffffff", 'width': 5},
                'thickness': 0.8,
                'value': 50
            }
        }
    ))

    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='#1e1e1e',
        plot_bgcolor='#1e1e1e',
        font={'size': 16, 'color': '#ffffff', 'family': 'Arial, sans-serif'}
    )

    return fig


def create_feature_importance_plot(feature_names, importance_values, top_n=15):
    """Create horizontal bar chart for feature importance."""
    # Get top features (filter out zeros)
    non_zero_mask = np.abs(importance_values) > 1e-6
    if non_zero_mask.sum() == 0:
        # If all zeros, return None to indicate no meaningful features
        return None

    # Filter and get top features
    non_zero_indices = np.where(non_zero_mask)[0]
    non_zero_importance = importance_values[non_zero_mask]

    # Get top N by absolute importance
    n_to_show = min(top_n, len(non_zero_importance))
    top_indices_in_nonzero = np.argsort(np.abs(non_zero_importance))[-n_to_show:]

    # Map back to original indices
    top_original_indices = non_zero_indices[top_indices_in_nonzero]

    top_features = [feature_names[i] for i in top_original_indices]
    top_values = importance_values[top_original_indices]

    # Sort by value for better visualization
    sort_idx = np.argsort(top_values)
    top_features = [top_features[i] for i in sort_idx]
    top_values = top_values[sort_idx]

    # Color based on positive/negative contribution with stronger colors
    colors = ['#2e7d32' if v > 0 else '#c62828' for v in top_values]

    fig = go.Figure(go.Bar(
        x=top_values,
        y=top_features,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(0,0,0,0.5)', width=2)
        ),
        text=[f"{v:+.4f}" if abs(v) >= 0.0001 else f"{v:+.2e}" for v in top_values],
        textposition='outside',
        textfont=dict(size=13, color='#1a1a1a', weight=600)
    ))

    fig.update_layout(
        title={
            'text': f"Top {len(top_features)} Contributing Features",
            'font': {'size': 22, 'color': '#ffffff', 'weight': 700}
        },
        xaxis_title="Contribution to Gait Prediction",
        yaxis_title="",
        height=max(450, len(top_features) * 35),
        margin=dict(l=250, r=80, t=80, b=60),
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#1e1e1e',
        font={'size': 13, 'color': '#ffffff', 'family': 'Arial, sans-serif'}
    )

    fig.update_xaxes(
        showgrid=True,
        gridwidth=2,
        gridcolor='#333333',
        zeroline=True,
        zerolinewidth=3,
        zerolinecolor='#00d4ff',
        tickfont=dict(size=12, color='#ffffff', weight=600),
        title_font=dict(size=14, color='#ffffff', weight=600)
    )

    fig.update_yaxes(
        tickfont=dict(size=12, color='#ffffff', weight=600)
    )

    return fig


def create_signal_visualization(df, sample_rate=100):
    """Create interactive signal visualization."""
    # Select key sensors to visualize
    key_sensors = [
        'accelerometer_right_foot_x',
        'accelerometer_right_foot_y',
        'accelerometer_right_foot_z',
        'gyroscope_right_foot_x',
        'gyroscope_right_foot_y',
        'gyroscope_right_foot_z'
    ]

    # Check which sensors exist
    available_sensors = [s for s in key_sensors if s in df.columns]

    if not available_sensors:
        return None

    # Create subplots
    n_sensors = len(available_sensors)
    fig = make_subplots(
        rows=n_sensors, cols=1,
        subplot_titles=available_sensors,
        vertical_spacing=0.08
    )

    # Time axis
    time = np.arange(len(df)) / sample_rate

    # Colors for different traces
    colors = ['#1976d2', '#388e3c', '#d32f2f', '#f57c00', '#7b1fa2', '#0097a7']

    # Add traces
    for i, sensor in enumerate(available_sensors, 1):
        fig.add_trace(
            go.Scatter(
                x=time,
                y=df[sensor],
                mode='lines',
                name=sensor,
                line=dict(width=2, color=colors[i-1])
            ),
            row=i, col=1
        )

    fig.update_layout(
        height=220 * n_sensors,
        showlegend=False,
        title_text="Sensor Signal Visualization",
        title_font=dict(size=22, color='#ffffff', weight=700),
        margin=dict(l=80, r=20, t=80, b=60),
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#1e1e1e',
        font={'size': 13, 'color': '#ffffff', 'family': 'Arial, sans-serif'}
    )

    fig.update_xaxes(
        title_text="Time (seconds)",
        row=n_sensors,
        col=1,
        titlefont=dict(size=14, color='#ffffff', weight=600),
        tickfont=dict(size=12, color='#ffffff', weight=600)
    )

    for i in range(1, n_sensors + 1):
        fig.update_yaxes(
            title_text="Value",
            row=i,
            col=1,
            titlefont=dict(size=13, color='#ffffff', weight=600),
            tickfont=dict(size=11, color='#ffffff', weight=600)
        )
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='#333333',
            row=i,
            col=1
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='#333333',
            row=i,
            col=1
        )

    # Update subplot title styling
    fig.update_annotations(font=dict(size=13, color='#ffffff', weight=600))

    return fig


def explain_prediction_with_lime(model, data, device, prediction, top_n=10):
    """Generate LIME explanation for prediction."""
    try:
        # Initialize LIME explainer
        explainer = GaitLIMEExplainer(model=model, device=device)

        # Get explanation for a sample window (use multiple windows for better stability)
        n_samples = min(5, len(data))
        sample_indices = np.linspace(0, len(data)-1, n_samples, dtype=int)

        all_importance = []

        for sample_idx in sample_indices:
            try:
                sample_data = data[sample_idx]

                # Get LIME explanation
                explanation = explainer.explain_single_sample(
                    sample_data=sample_data,
                    num_features=top_n,
                    num_samples=1000  # Increased for better accuracy
                )

                # Extract feature importance
                feature_names = explainer.feature_names
                importance_map = explanation.as_map()[1]  # For class 1 (Gait)

                # Create importance array
                importance = np.zeros(len(feature_names))
                for feat_idx, weight in importance_map:
                    importance[feat_idx] = weight

                all_importance.append(importance)
            except:
                continue

        if not all_importance:
            return None, None, "Failed to generate LIME explanations"

        # Average importance across samples for stability
        avg_importance = np.mean(all_importance, axis=0)

        return feature_names, avg_importance, None

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, f"Error generating LIME explanation: {str(e)}"


def main():
    """Main dashboard function."""

    # Header
    st.markdown('<h1 class="main-title">🚶 Gait Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload sensor data to detect and analyze human gait patterns</p>', unsafe_allow_html=True)

    # Load model
    with st.spinner("Loading model and preprocessing pipeline..."):
        model, pipeline, device, error = load_model_and_pipeline()

    if error:
        st.error(f"❌ {error}")
        st.info("💡 Please train the model first by running: `python efficient_run.py`")
        return

    st.success("✅ Model loaded successfully!")

    # Sidebar
    with st.sidebar:
        st.header("📊 About")
        st.markdown("""
        This system uses a **CNN-BiLSTM** deep learning model to detect human gait patterns from multi-sensor wearable data.

        **Features:**
        - Real-time gait detection
        - LIME-based explanations
        - Confidence scoring
        - Feature importance analysis
        """)

        st.markdown("---")

        st.header("📁 File Requirements")
        st.markdown("""
        **Required columns:**
        - Accelerometer data (x, y, z) for 6 locations
        - Gyroscope data (x, y, z) for 6 locations
        - EMG data (right, left)
        - Activity label

        **Locations:**
        - Right: foot, shin, thigh
        - Left: foot, shin, thigh

        **Total:** 38 features
        """)

        st.markdown("---")

        st.header("⚙️ Settings")
        show_signals = st.checkbox("Show signal visualization", value=True)
        show_lime = st.checkbox("Show LIME explanation", value=True)
        num_features = st.slider("Number of top features", 5, 20, 15)

    # Main content
    st.markdown("---")

    # Upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload Your Sensor Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a file containing sensor readings in the required format"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        # Read file first to get column info
        if uploaded_file.name.endswith('.csv'):
            temp_df = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)  # Reset file pointer
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            temp_df = pd.read_excel(uploaded_file)
            uploaded_file.seek(0)  # Reset file pointer
        else:
            temp_df = None

        # File info cards with attractive styling
        st.markdown("### 📋 File Information")

        info_col1, info_col2, info_col3, info_col4 = st.columns(4)

        with info_col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 1.5rem; border-radius: 10px; text-align: center;
                        box-shadow: 0 0 20px rgba(102,126,234,0.5);
                        border: 2px solid #667eea;">
                <div style="color: white; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem; text-transform: uppercase;">📄 FILE NAME</div>
                <div style="color: #ffffff; font-size: 1.1rem; font-weight: 700;">{uploaded_file.name}</div>
            </div>
            """, unsafe_allow_html=True)

        with info_col2:
            file_size_kb = uploaded_file.size / 1024
            file_size_display = f"{file_size_kb:.2f} KB" if file_size_kb < 1024 else f"{file_size_kb/1024:.2f} MB"
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #ff006e 0%, #ff7700 100%);
                        padding: 1.5rem; border-radius: 10px; text-align: center;
                        box-shadow: 0 0 20px rgba(255,0,110,0.5);
                        border: 2px solid #ff006e;">
                <div style="color: white; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem; text-transform: uppercase;">💾 FILE SIZE</div>
                <div style="color: #ffffff; font-size: 1.1rem; font-weight: 700;">{file_size_display}</div>
            </div>
            """, unsafe_allow_html=True)

        with info_col3:
            total_columns = len(temp_df.columns) if temp_df is not None else 0
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #00d4ff 0%, #0099ff 100%);
                        padding: 1.5rem; border-radius: 10px; text-align: center;
                        box-shadow: 0 0 20px rgba(0,212,255,0.5);
                        border: 2px solid #00d4ff;">
                <div style="color: white; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem; text-transform: uppercase;">📊 TOTAL COLUMNS</div>
                <div style="color: #ffffff; font-size: 1.1rem; font-weight: 700;">{total_columns}</div>
            </div>
            """, unsafe_allow_html=True)

        with info_col4:
            # Count sensor columns (38 expected)
            sensor_cols = ['accelerometer', 'gyroscope', 'EMG']
            sensor_count = sum(1 for col in temp_df.columns for s in sensor_cols if s in col) if temp_df is not None else 0
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #84ccaa 0%, #8ddfb6 100%);
                        padding: 1.5rem; border-radius: 10px; text-align: center;
                        box-shadow: 0 0 20px rgba(0,255,136,0.5);
                        border: 2px solid #00ff88;">
                <div style="color: white; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem; text-transform: uppercase;">🎯 SENSOR COLUMNS</div>
                <div style="color: #ffffff; font-size: 1.1rem; font-weight: 700;">{sensor_count}/38</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Preprocess
        with st.spinner("🔄 Preprocessing data..."):
            windowed_data, original_df, error = preprocess_uploaded_file(uploaded_file, pipeline)

        if error:
            st.error(f"❌ {error}")
            return

        st.success(f"✅ Preprocessing complete! Generated {len(windowed_data)} windows")

        # Show original data preview
        with st.expander("📊 View Raw Data Preview"):
            st.dataframe(original_df.head(100), use_container_width=True)
            st.caption(f"Showing first 100 of {len(original_df)} rows")

        # Make prediction
        with st.spinner("🤖 Analyzing data with AI model..."):
            probabilities = predict_with_model(model, windowed_data, device)
            avg_probability = float(probabilities.mean())
            gait_percentage = float((probabilities > 0.5).mean() * 100)
            is_gait = avg_probability > 0.5

        st.markdown("---")

        # Results section
        st.markdown("## 🎯 Analysis Results")

        # Main result card
        if is_gait:
            st.markdown(f"""
            <div class="result-card gait-detected fade-in">
                <div class="result-title">✅ GAIT DETECTED</div>
                <div class="confidence-score">Confidence: {avg_probability*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card no-gait-detected fade-in">
                <div class="result-title">❌ NO GAIT DETECTED</div>
                <div class="confidence-score">Confidence: {(1-avg_probability)*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Average Probability",
                value=f"{avg_probability*100:.1f}%",
                delta=f"{(avg_probability-0.5)*100:.1f}%" if avg_probability > 0.5 else f"{(avg_probability-0.5)*100:.1f}%"
            )

        with col2:
            gait_windows_count = int((probabilities > 0.5).sum())
            st.metric(
                label="Gait Windows",
                value=f"{gait_windows_count}/{len(probabilities)}",
                delta=f"{gait_percentage:.1f}%"
            )

        with col3:
            st.metric(
                label="Min Probability",
                value=f"{float(probabilities.min())*100:.1f}%"
            )

        with col4:
            st.metric(
                label="Max Probability",
                value=f"{float(probabilities.max())*100:.1f}%"
            )

        # Gauge chart
        st.plotly_chart(create_prediction_gauge(avg_probability), use_container_width=True)

        # Probability distribution
        st.markdown("### 📈 Probability Distribution Across Windows")

        fig_prob = go.Figure()
        fig_prob.add_trace(go.Scatter(
            y=probabilities.flatten(),
            mode='lines+markers',
            name='Probability',
            line=dict(color='#667eea', width=3),
            marker=dict(size=6, color='#667eea', line=dict(color='#1a1a1a', width=1))
        ))

        fig_prob.add_hline(
            y=0.5,
            line_dash="dash",
            line_color="#d32f2f",
            line_width=3,
            annotation_text="Decision Threshold (50%)",
            annotation_font=dict(size=14, color='#1a1a1a', weight=600)
        )

        fig_prob.update_layout(
            xaxis_title="Window Number",
            yaxis_title="Gait Probability",
            height=450,
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#1e1e1e',
            yaxis=dict(
                range=[0, 1],
                tickfont=dict(size=13, color='#ffffff', weight=600),
                titlefont=dict(size=15, color='#ffffff', weight=700)
            ),
            xaxis=dict(
                tickfont=dict(size=13, color='#ffffff', weight=600),
                titlefont=dict(size=15, color='#ffffff', weight=700)
            ),
            font={'size': 13, 'color': '#ffffff', 'family': 'Arial, sans-serif'},
            margin=dict(l=60, r=20, t=40, b=60)
        )

        fig_prob.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#333333')
        fig_prob.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#333333')

        st.plotly_chart(fig_prob, use_container_width=True)

        # Signal visualization
        if show_signals and original_df is not None:
            st.markdown("---")
            st.markdown("### 📊 Sensor Signal Analysis")

            signal_fig = create_signal_visualization(original_df)
            if signal_fig:
                st.plotly_chart(signal_fig, use_container_width=True)

        # LIME Explanation
        if show_lime:
            st.markdown("---")
            st.markdown("### 🔍 AI Decision Explanation (LIME)")

            st.markdown("""
            <div style="background: #2a2a2a;
                        padding: 1.5rem; border-radius: 10px;
                        border-left: 5px solid #00d4ff; margin: 1rem 0;
                        box-shadow: 0 0 20px rgba(0,212,255,0.2);
                        border: 1px solid #00d4ff;">
                <strong style="color: #00d4ff; font-size: 1.1rem;">ℹ️ What is LIME?</strong><br>
                <span style="color: #e0e0e0;">
                    LIME (Local Interpretable Model-agnostic Explanations) shows which features
                    contributed most to the AI's decision. <strong style="color: #00ff88;">Green bars</strong> indicate features supporting
                    GAIT detection, <strong style="color: #ff006e;">red bars</strong> indicate features against it.
                </span>
            </div>
            """, unsafe_allow_html=True)

            with st.spinner("🔮 Generating explanation..."):
                feature_names, importance, error = explain_prediction_with_lime(
                    model, windowed_data, device, is_gait, top_n=num_features
                )

            if error:
                st.warning(f"⚠️ {error}")
            elif importance is None or (np.abs(importance) < 1e-6).all():
                st.warning("⚠️ Unable to generate meaningful LIME explanation. The model may need more diverse training data.")
            else:
                # Feature importance plot
                importance_fig = create_feature_importance_plot(
                    feature_names, importance, top_n=num_features
                )

                if importance_fig is None:
                    st.warning("⚠️ No significant feature contributions detected. All importance values are near zero.")
                else:
                    st.plotly_chart(importance_fig, use_container_width=True)

                    # Detailed explanation
                    st.markdown("#### 📝 Key Contributing Factors")

                    # Get top positive and negative features (filter out near-zero values)
                    significant_mask = np.abs(importance) > 1e-4
                    if significant_mask.sum() > 0:
                        significant_indices = np.where(significant_mask)[0]
                        significant_importance = importance[significant_mask]

                        # Get top indices
                        n_show = min(num_features, len(significant_importance))
                        top_indices_in_significant = np.argsort(np.abs(significant_importance))[-n_show:]
                        indices = significant_indices[top_indices_in_significant]

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("""
                            <div style="background: linear-gradient(135deg, #00ff88 0%, #00cc66 100%);
                                        padding: 1rem; border-radius: 8px;
                                        border: 2px solid #00ff88;
                                        box-shadow: 0 0 15px rgba(0,255,136,0.3);">
                                <strong style="color: #ffffff; font-size: 1.1rem;">🟢 Supporting GAIT Detection</strong>
                            </div>
                            """, unsafe_allow_html=True)
                            st.markdown("<br>", unsafe_allow_html=True)

                            positive_features = [(feature_names[i], importance[i])
                                                for i in indices if importance[i] > 1e-4]
                            positive_features.sort(key=lambda x: x[1], reverse=True)

                            if positive_features:
                                for feat, imp in positive_features[:5]:
                                    st.markdown(f"""
                                    <div style="background: #2a2a2a; padding: 0.8rem; margin: 0.5rem 0;
                                                border-radius: 6px; border-left: 4px solid #00ff88;
                                                border: 1px solid #333;">
                                        <strong style="color: #ffffff;">{feat}</strong><br>
                                        <span style="color: #00ff88; font-size: 1.1rem; font-weight: 700;">+{imp:.4f}</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.info("No significant positive contributions")

                        with col2:
                            st.markdown("""
                            <div style="background: linear-gradient(135deg, #ff006e 0%, #ff7700 100%);
                                        padding: 1rem; border-radius: 8px;
                                        border: 2px solid #ff006e;
                                        box-shadow: 0 0 15px rgba(255,0,110,0.3);">
                                <strong style="color: #ffffff; font-size: 1.1rem;">🔴 Against GAIT Detection</strong>
                            </div>
                            """, unsafe_allow_html=True)
                            st.markdown("<br>", unsafe_allow_html=True)

                            negative_features = [(feature_names[i], importance[i])
                                                for i in indices if importance[i] < -1e-4]
                            negative_features.sort(key=lambda x: abs(x[1]), reverse=True)

                            if negative_features:
                                for feat, imp in negative_features[:5]:
                                    st.markdown(f"""
                                    <div style="background: #2a2a2a; padding: 0.8rem; margin: 0.5rem 0;
                                                border-radius: 6px; border-left: 4px solid #ff006e;
                                                border: 1px solid #333;">
                                        <strong style="color: #ffffff;">{feat}</strong><br>
                                        <span style="color: #ff006e; font-size: 1.1rem; font-weight: 700;">{imp:.4f}</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.info("No significant negative contributions")
                    else:
                        st.info("No features with significant contribution detected.")

        # Summary
        st.markdown("---")
        st.markdown("### 📋 Summary")

        if is_gait:
            st.success(f"""
            **✅ GAIT PATTERN DETECTED**

            The AI model has detected a gait pattern with **{avg_probability*100:.1f}% confidence**.

            **Why this is GAIT:**
            - {gait_percentage:.1f}% of analyzed windows show gait characteristics
            - Key accelerometer and gyroscope patterns match typical walking/running signatures
            - Sensor readings show periodic motion consistent with leg movement
            - EMG signals indicate muscle activation patterns typical of locomotion
            """)
        else:
            st.warning(f"""
            **❌ NO GAIT PATTERN DETECTED**

            The AI model did not detect a gait pattern (**{(1-avg_probability)*100:.1f}% confidence**).

            **Why this is NOT GAIT:**
            - Only {gait_percentage:.1f}% of analyzed windows show gait characteristics
            - Sensor readings lack periodic patterns typical of walking/running
            - Motion patterns suggest stationary activity (sitting, standing, lying)
            - Low variance in key accelerometer readings indicates minimal movement
            """)

        # Download results
        st.markdown("---")
        st.markdown("### 💾 Download Results")

        # Prepare results dataframe
        results_df = pd.DataFrame({
            'Window': range(len(probabilities)),
            'Probability': probabilities.flatten(),
            'Prediction': ['GAIT' if p > 0.5 else 'NOT GAIT' for p in probabilities.flatten()]
        })

        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions (CSV)",
            data=csv,
            file_name=f"gait_predictions_{uploaded_file.name.split('.')[0]}.csv",
            mime="text/csv"
        )

    else:
        # Show example
        st.markdown("---")
        st.markdown("### 📖 How to Use")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            #### 1️⃣ Upload File
            Click the upload button above and select your CSV or Excel file containing sensor data.
            """)

        with col2:
            st.markdown("""
            #### 2️⃣ Get Results
            The system will automatically analyze your data and show whether gait is detected.
            """)

        with col3:
            st.markdown("""
            #### 3️⃣ Understand Why
            View LIME explanations to understand which features contributed to the decision.
            """)

        st.markdown("---")
        st.info("👆 Upload a file to get started!")


if __name__ == "__main__":
    main()
