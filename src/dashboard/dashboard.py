# src/dashboard/dashboard.py
"""
Beautiful Professional Gait Detection Dashboard
Stunning UI with vibrant colors and clear visibility
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
from datetime import datetime

# Add parent directories to path
dashboard_dir = Path(__file__).parent
src_dir = dashboard_dir.parent
project_dir = src_dir.parent
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(project_dir))
sys.path.insert(0, str(dashboard_dir))

from inference_handler import InferenceHandler
from lime_explainer_dash import DashboardLIMEExplainer

# Page configuration
st.set_page_config(
    page_title="Gait Detection AI - Medical Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Beautiful CSS with vibrant colors
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Gradient background */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Content wrapper with glass effect */
    .block-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    /* Stunning header with gradient */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 15s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.3; }
        50% { transform: scale(1.1); opacity: 0.5; }
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 3rem;
        color: white;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        margin: 1rem 0 0 0;
        font-size: 1.3rem;
        opacity: 0.95;
        font-weight: 500;
        position: relative;
        z-index: 1;
    }
    
    /* Colorful metric cards */
    .metric-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        transition: all 0.3s ease;
        border-top: 5px solid;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        background: linear-gradient(90deg, currentColor, transparent);
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 12px 30px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        margin: 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
    }
    
    /* Colorful status badges */
    .status-badge {
        display: inline-block;
        padding: 1rem 2rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.3rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .status-gait {
        background: linear-gradient(135deg, #34d399 0%, #10b981 100%);
        color: white;
    }
    
    .status-no-gait {
        background: linear-gradient(135deg, #f87171 0%, #ef4444 100%);
        color: white;
    }
    
    .status-moderate {
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        color: white;
    }
    
    /* Beautiful info cards */
    .info-card {
        background: linear-gradient(135deg, #e0e7ff 0%, #ddd6fe 100%);
        padding: 2rem;
        color: #1e293b;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.1);
    }
    
    .success-box {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 5px solid #10b981;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.1);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 5px solid #f59e0b;
        padding: 1.5rem;
        color:#ef4444;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.1);
    }
    
    .error-box {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 5px solid #ef4444;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.1);
    }
    
    /* Stunning buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1rem 2.5rem;
        font-size: 1.1rem;
        font-weight: 700;
        border-radius: 50px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Beautiful sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .css-1d391kg .stMarkdown, [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    /* Sidebar inputs */
    .css-1d391kg input, [data-testid="stSidebar"] input {
        background: rgba(255, 255, 255, 0.9);
        border: 2px solid rgba(255, 255, 255, 0.3);
        border-radius: 10px;
        padding: 0.75rem;
        color: #1e293b;
        font-weight: 500;
    }
    
    /* File uploader styling */
    .stFileUploader {
        background: linear-gradient(135deg, #e0e7ff 0%, #ddd6fe 100%);
        border: 3px dashed #667eea;
        border-radius: 20px;
        color: black;
        padding: 3rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: #764ba2;
        transform: scale(1.01);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #e0e7ff 0%, #ddd6fe 100%);
        border-radius: 10px;
        font-weight: 600;
        color: #667eea;
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Success/error messages */
    .stSuccess {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        color: #065f46;
        border-radius: 10px;
        padding: 1rem;
        font-weight: 600;
    }
    
    .stError {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        color: #991b1b;
        border-radius: 10px;
        padding: 1rem;
        font-weight: 600;
    }
    
    /* Section headers */
    h2, h3 {
        color: #1e293b;
        font-weight: 700;
        margin-top: 2rem;
    }
    
    /* Metric styling */
    [data-testid="metric-container"] {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'inference_handler' not in st.session_state:
        st.session_state.inference_handler = None


def render_header():
    """Render the stunning dashboard header."""
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Gait Detection AI System</h1>
        <p>Advanced Medical Analysis with Explainable AI</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render beautiful sidebar."""
    with st.sidebar:
        st.markdown("<h2 style='color: white; text-align: center;'>⚙️ System Control</h2>", unsafe_allow_html=True)
        st.markdown("<hr style='border-color: rgba(255,255,255,0.3);'>", unsafe_allow_html=True)
        
        # Model configuration
        st.markdown("<h3 style='color: white;'>🤖 Model Settings</h3>", unsafe_allow_html=True)
        
        model_path = st.text_input(
            "Model Path",
            value="models/best_model.pt",
            help="Path to trained model file",
            label_visibility="collapsed"
        )
        st.caption("📁 Model: " + model_path)
        
        st.info("✨ Normalizer fits automatically on your data")
        
        device = st.selectbox(
            "Computing Device",
            ["Auto", "CPU", "CUDA"],
            help="Device for inference"
        )
        
        st.markdown("<hr style='border-color: rgba(255,255,255,0.3);'>", unsafe_allow_html=True)
        
        # LIME settings
        st.markdown("<h3 style='color: white;'>🔍 LIME Settings</h3>", unsafe_allow_html=True)
        num_samples = st.slider(
            "Samples to Explain",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of predictions to explain"
        )
        
        num_features = st.slider(
            "Top Features",
            min_value=5,
            max_value=20,
            value=10,
            help="Features to display"
        )
        
        st.markdown("<hr style='border-color: rgba(255,255,255,0.3);'>", unsafe_allow_html=True)
        
        # System status
        st.markdown("<h3 style='color: white;'>📊 System Status</h3>", unsafe_allow_html=True)
        if st.session_state.inference_handler:
            st.success("✅ System Ready")
        else:
            st.warning("⚠️ Not Initialized")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Initialize button
        if st.button("🚀 INITIALIZE SYSTEM", use_container_width=True, type="primary"):
            with st.spinner("🔄 Loading model..."):
                try:
                    st.session_state.inference_handler = InferenceHandler(
                        model_path=model_path,
                        device=device.lower() if device != "Auto" else None
                    )
                    st.success("✅ System initialized!")
                    st.balloons()
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed: {str(e)}")
                    import traceback
                    with st.expander("🐛 Error Details"):
                        st.code(traceback.format_exc())
        
        return {
            'num_samples': num_samples,
            'num_features': num_features
        }


def render_upload_section():
    """Render beautiful file upload section."""
    st.markdown("### 📁 Upload Patient Data")
    
    uploaded_file = st.file_uploader(
        "Drop CSV file here or click to browse",
        type=['csv'],
        help="Upload patient gait sensor data (38 features, CSV format)"
    )
    
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        
        # Beautiful file info cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card" style="border-top-color: #667eea;">
                <div class="metric-label">📄 File Name</div>
                <div style="font-size: 1.2rem; font-weight: 600; color: #1e293b; margin-top: 0.5rem;">
                    {uploaded_file.name}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card" style="border-top-color: #10b981;">
                <div class="metric-label">💾 File Size</div>
                <div style="font-size: 1.2rem; font-weight: 600; color: #1e293b; margin-top: 0.5rem;">
                    {uploaded_file.size / 1024:.2f} KB
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card" style="border-top-color: #f59e0b;">
                <div class="metric-label">⏰ Upload Time</div>
                <div style="font-size: 1.2rem; font-weight: 600; color: #1e293b; margin-top: 0.5rem;">
                    {datetime.now().strftime("%H:%M:%S")}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Preview with beautiful styling
        with st.expander("👁️ Preview Data", expanded=False):
            try:
                df = pd.read_csv(uploaded_file)
                uploaded_file.seek(0)
                st.dataframe(df.head(10), use_container_width=True)
                st.info(f"📊 Total: **{len(df)} rows** × **{len(df.columns)} columns**")
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    return uploaded_file


def create_beautiful_gauge(gait_percentage):
    """Create stunning gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=gait_percentage,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Gait Activity Level", 'font': {'size': 28, 'color': '#1e293b', 'family': 'Inter'}},
        delta={'reference': 50, 'increasing': {'color': "#10b981"}, 'decreasing': {'color': "#ef4444"}},
        number={'font': {'size': 60, 'color': '#667eea', 'family': 'Inter', 'weight': 800}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#667eea"},
            'bar': {'color': "#667eea", 'thickness': 0.8},
            'bgcolor': "white",
            'borderwidth': 3,
            'bordercolor': "#e2e8f0",
            'steps': [
                {'range': [0, 30], 'color': '#fee2e2'},
                {'range': [30, 70], 'color': '#fef3c7'},
                {'range': [70, 100], 'color': '#d1fae5'}
            ],
            'threshold': {
                'line': {'color': "#764ba2", 'width': 6},
                'thickness': 0.9,
                'value': gait_percentage
            }
        }
    ))
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor="white",
        font={'family': "Inter", 'size': 14}
    )
    
    return fig


def create_beautiful_distribution(predictions):
    """Create stunning distribution chart."""
    gait_count = np.sum(predictions == 1)
    non_gait_count = np.sum(predictions == 0)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['Gait Detected', 'Non-Gait'],
        y=[gait_count, non_gait_count],
        marker=dict(
            color=['#10b981', '#ef4444'],
            line=dict(color='white', width=3)
        ),
        text=[f'<b>{gait_count}</b>', f'<b>{non_gait_count}</b>'],
        textposition='inside',
        textfont=dict(size=24, color='white', family='Inter', weight='bold'),
        hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': "Window Classification Distribution",
            'font': {'size': 24, 'color': '#1e293b', 'family': 'Inter', 'weight': 'bold'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Classification",
        yaxis_title="Number of Windows",
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        font=dict(size=14, family='Inter'),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#f1f5f9')
    )
    
    return fig


def render_results(results, lime_settings):
    """Render beautiful results section."""
    st.markdown("<hr style='margin: 2rem 0;'>", unsafe_allow_html=True)
    st.markdown("## 📊 Analysis Results")
    
    # Stunning metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    gait_percentage = (results['num_gait'] / results['num_windows']) * 100
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="border-top-color: #667eea;">
            <div class="metric-label">Total Windows</div>
            <div class="metric-value">{results['num_windows']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-top-color: #10b981;">
            <div class="metric-label">✅ Gait Detected</div>
            <div class="metric-value">{results['num_gait']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-top-color: #ef4444;">
            <div class="metric-label">❌ Non-Gait</div>
            <div class="metric-value">{results['num_non_gait']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="border-top-color: #f59e0b;">
            <div class="metric-label">Activity Level</div>
            <div class="metric-value">{gait_percentage:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Beautiful visualizations
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.plotly_chart(create_beautiful_gauge(gait_percentage), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_beautiful_distribution(results['predictions']), use_container_width=True)
    
    # Clinical interpretation with vibrant colors
    st.markdown("### 🏥 Clinical Interpretation")
    
    if gait_percentage > 70:
        badge_class = "status-gait"
        status_text = "✅ Significant Gait Activity"
    elif gait_percentage > 30:
        badge_class = "status-moderate"
        status_text = "⚠️ Moderate Gait Activity"
    else:
        badge_class = "status-no-gait"
        status_text = "ℹ️ Limited Gait Activity"
    
    st.markdown(f'<div class="status-badge {badge_class}">{status_text}</div>', unsafe_allow_html=True)
    
    if gait_percentage > 70:
        st.markdown(f"""
        <div class="success-box">
            <h3 style="margin-top: 0; color: #065f46;">Significant Gait Activity Detected</h3>
            <p style="font-size: 1.1rem; color: #047857; line-height: 1.6;">
            The analysis shows <strong>substantial gait activity ({gait_percentage:.1f}%)</strong> in the recorded data. 
            This indicates the patient was <strong>actively walking</strong> during the measurement period.
            The model detected consistent walking patterns with proper biomechanical features.
            </p>
        </div>
        """, unsafe_allow_html=True)
    elif gait_percentage > 30:
        st.markdown(f"""
        <div class="warning-box">
            <h3 style="margin-top: 0; color: #92400e;">Moderate Gait Activity</h3>
            <p style="font-size: 1.1rem; color: #b45309; line-height: 1.6;">
            The analysis detected <strong>moderate gait activity ({gait_percentage:.1f}%)</strong>. 
            The patient showed <strong>intermittent walking patterns</strong> during the recording.
            This suggests periods of both activity and rest.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="error-box">
            <h3 style="margin-top: 0; color: #991b1b;">Limited Gait Activity</h3>
            <p style="font-size: 1.1rem; color: #b91c1c; line-height: 1.6;">
            The analysis found <strong>limited gait activity ({gait_percentage:.1f}%)</strong>. 
            Most of the recorded data shows <strong>non-walking activities or rest periods</strong>.
            The patient was predominantly stationary during measurement.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # LIME Explanations
    if 'lime_explanations' in results:
        render_lime_explanations(results['lime_explanations'], lime_settings['num_features'])


def render_lime_explanations(explanations, num_features):
    """Render beautiful LIME explanations."""
    st.markdown("<hr style='margin: 2rem 0;'>", unsafe_allow_html=True)
    st.markdown("## 🔍 AI Explanation (LIME)")
    
    st.markdown("""
    <div class="info-card">
        <h4 style="margin-top: 0;">💡 What is LIME?</h4>
        <p style="font-size: 1.05rem;">
        Local Interpretable Model-agnostic Explanations (LIME) reveals which sensor features 
        influenced each prediction, making AI decisions transparent and trustworthy.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Global feature importance
    if 'global_importance' in explanations:
        st.markdown("### 🌍 Overall Feature Importance")
        
        importance_dict = explanations['global_importance']
        top_features = list(importance_dict.items())[:num_features]
        
        features, importance = zip(*top_features)
        
        # Create stunning gradient colors
        colors = px.colors.sequential.Viridis_r[:len(features)]
        
        fig = go.Figure(go.Bar(
            y=features,
            x=importance,
            orientation='h',
            marker=dict(
                color=importance,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Importance", thickness=15)
            ),
            text=[f'{imp:.4f}' for imp in importance],
            textposition='auto',
            textfont=dict(size=12, family='Inter', weight='bold'),
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Top {len(features)} Most Influential Features",
            xaxis_title="Importance Score",
            yaxis_title="",
            height=max(400, len(features) * 35),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12, family='Inter'),
            xaxis=dict(showgrid=True, gridcolor='#f1f5f9'),
            yaxis=dict(showgrid=False),
            margin=dict(l=200, r=20, t=60, b=40)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Individual explanations with beautiful cards
    if 'samples' in explanations:
        st.markdown("### 🔬 Individual Window Explanations")
        
        for i, explanation in enumerate(explanations['samples'][:3]):
            with st.expander(f"📌 Window {explanation['sample_idx']} - {explanation['predicted_class']}", expanded=(i==0)):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    pred_val = explanation['prediction']
                    pred_class = explanation['predicted_class']
                    r2 = explanation['r2_score']
                    
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #e0e7ff 0%, #ddd6fe 100%); 
                                padding: 1.5rem; border-radius: 12px; text-align: center;'>
                        <div style='font-size: 0.9rem; color: #64748b; font-weight: 600; 
                                    text-transform: uppercase; letter-spacing: 1px;'>
                            Prediction Score
                        </div>
                        <div style='font-size: 3rem; font-weight: 800; color: #667eea; margin: 0.5rem 0;'>
                            {pred_val:.3f}
                        </div>
                        <div style='font-size: 1.2rem; font-weight: 700; 
                                    color: {"#10b981" if pred_class == "Gait" else "#ef4444"};'>
                            {pred_class}
                        </div>
                        <hr style='border-color: rgba(0,0,0,0.1); margin: 1rem 0;'>
                        <div style='font-size: 0.85rem; color: #64748b;'>
                            Model Fit: <strong>{r2:.3f}</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Feature importance
                    importance = explanation['feature_importance']
                    feature_names = explanation['feature_names']
                    
                    abs_importance = np.abs(importance)
                    top_indices = np.argsort(abs_importance)[-10:][::-1]
                    
                    top_imp = importance[top_indices]
                    top_names = [feature_names[int(i)] for i in top_indices]
                    
                    colors = ['#10b981' if x > 0 else '#ef4444' for x in top_imp]
                    
                    fig = go.Figure(go.Bar(
                        x=top_imp,
                        y=top_names,
                        orientation='h',
                        marker=dict(color=colors),
                        text=[f'{x:+.3f}' for x in top_imp],
                        textposition='auto',
                        textfont=dict(size=11, family='Inter', weight='bold', color='white'),
                        hovertemplate='<b>%{y}</b><br>Impact: %{x:+.3f}<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title="Feature Contributions to Prediction",
                        xaxis_title="Impact on Prediction",
                        height=380,
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        showlegend=False,
                        font=dict(size=11, family='Inter'),
                        xaxis=dict(showgrid=True, gridcolor='#f1f5f9', zeroline=True, zerolinecolor='#94a3b8', zerolinewidth=2),
                        yaxis=dict(showgrid=False),
                        margin=dict(l=150, r=20, t=50, b=40)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                <div style='background: #f8fafc; padding: 1rem; border-radius: 8px; 
                            border-left: 4px solid #667eea; margin-top: 1rem;'>
                    <strong>💡 Interpretation:</strong>
                    <span style='color: #10b981; font-weight: 600;'>● Green bars</span> push toward "Gait" | 
                    <span style='color: #ef4444; font-weight: 600;'>● Red bars</span> push toward "Non-Gait"
                </div>
                """, unsafe_allow_html=True)


def main():
    """Main dashboard application."""
    initialize_session_state()
    render_header()
    lime_settings = render_sidebar()
    
    # Main content
    if st.session_state.inference_handler is None:
        st.markdown("""
        <div class="warning-box">
            <h3 style="margin-top: 0;">⚠️ System Not Initialized</h3>
            <p style="font-size: 1.1rem;">Please initialize the system using the sidebar before uploading data:</p>
            <ol style="font-size: 1.05rem; line-height: 1.8;">
                <li>Check that the model path is correct in the sidebar</li>
                <li>Click <strong>"🚀 INITIALIZE SYSTEM"</strong></li>
                <li>Wait for the success message</li>
                <li>Upload your patient data below</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    else:
        # File upload
        uploaded_file = render_upload_section()
        
        # Analyze button
        if uploaded_file is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 ANALYZE PATIENT DATA", use_container_width=True, type="primary"):
                    with st.spinner("🔄 Analyzing data... Please wait..."):
                        try:
                            # Run inference
                            results = st.session_state.inference_handler.predict_from_file(
                                uploaded_file,
                                include_lime=True,
                                num_lime_samples=lime_settings['num_samples']
                            )
                            
                            st.session_state.results = results
                            st.success("✅ Analysis complete!")
                            st.balloons()
                            
                        except Exception as e:
                            st.error(f"❌ Analysis failed: {str(e)}")
                            import traceback
                            with st.expander("🐛 Error Details"):
                                st.code(traceback.format_exc())
        
        # Show results
        if st.session_state.results is not None:
            render_results(st.session_state.results, lime_settings)
    
    # Beautiful footer
    st.markdown("<hr style='margin: 3rem 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #64748b; padding: 2rem; 
                background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); 
                border-radius: 15px;'>
        <p style='font-size: 1.1rem; font-weight: 600; margin: 0;'>
            🏥 <strong>Gait Detection AI System</strong> | Version 1.0
        </p>
        <p style='font-size: 0.95rem; margin: 0.5rem 0 0 0;'>
            For medical professional use only. Results should be reviewed by qualified healthcare providers.
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()