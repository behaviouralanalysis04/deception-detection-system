import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import os
import subprocess

# Import modules
import utils
from audio_analyzer import AudioAnalyzer
from video_processor import VideoAnalysisProcessor
from model_loader import DeceptionModel

# Page configuration
st.set_page_config(
    page_title="Deception Detection System",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize model with caching
@st.cache_resource
def load_model():
    return DeceptionModel()

model = load_model()

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, rgba(102,126,234,0.2), rgba(118,75,162,0.2));
        border-radius: 20px;
        margin-bottom: 2rem;
    }
    .title-text {
        font-size: 48px;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .subtitle {
        color: rgba(255,255,255,0.7);
        text-align: center;
        margin-top: 0.5rem;
    }
    .metric-card {
        background: rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem;
        backdrop-filter: blur(10px);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: transform 0.2s;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(102,126,234,0.4);
    }
    .result-card {
        background: rgba(0,0,0,0.5);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(102,126,234,0.3);
    }
    .indicator-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        margin: 0.25rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .indicator-high {
        background: rgba(220,53,69,0.3);
        color: #ff6b6b;
        border: 1px solid #dc3545;
    }
    .indicator-medium {
        background: rgba(255,193,7,0.3);
        color: #ffd43b;
        border: 1px solid #ffc107;
    }
    .indicator-low {
        background: rgba(40,167,69,0.3);
        color: #51cf66;
        border: 1px solid #28a745;
    }
    hr {
        margin: 2rem 0;
        background: linear-gradient(90deg, transparent, #667eea, #764ba2, transparent);
        height: 2px;
        border: none;
    }
    .info-box {
        background: rgba(102,126,234,0.2);
        border-left: 4px solid #667eea;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# Helper functions
def create_gauge_chart(score, title="Deception Score"):
    """Create a gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20, 'color': 'white'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': 'white', 'tickwidth': 2},
            'bar': {'color': '#667eea'},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 40], 'color': "rgba(40,167,69,0.3)"},
                {'range': [40, 60], 'color': "rgba(255,193,7,0.3)"},
                {'range': [60, 100], 'color': "rgba(220,53,69,0.3)"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': 'white', 'family': 'Arial'},
        height=300,
        margin=dict(l=30, r=30, t=50, b=30)
    )
    return fig

def process_video_file(video_path, progress_callback=None):
    """Process uploaded video file"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Cannot open video file")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 60:
        fps = 30
    
    processor = VideoAnalysisProcessor()
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Process every 3rd frame for efficiency
    process_every = max(1, int(fps / 10))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count % process_every == 0:
            processor.process_frame(frame, frame_count, fps)
        
        # Update progress
        if progress_callback and frame_count % 30 == 0:
            progress = min(90, int(frame_count / total_frames * 100))
            progress_callback(progress)
    
    cap.release()
    
    duration = frame_count / fps
    features = processor.get_features(duration, fps)
    
    # Get deception score
    deception_score = model.predict(features)
    summary = processor.generate_summary(features, deception_score)
    
    return features, summary

def process_audio_file(audio_path):
    """Process audio file"""
    analyzer = AudioAnalyzer(audio_path)
    analyzer.load_audio()
    features = analyzer.extract_all_features()
    audio_score = analyzer.calculate_deception_score(features)
    transcript = analyzer.transcribe_audio()
    return features, audio_score, transcript

# Main Header
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.markdown('<h1 class="title-text">🎭 Deception Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Analysis of Facial Expressions & Speech Patterns</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📊 System Status")
    
    if st.session_state.analysis_history:
        history_df = pd.DataFrame(st.session_state.analysis_history)
        st.metric("Total Analyses", len(history_df))
        if len(history_df) > 0:
            st.metric("Average Score", f"{history_df['score'].mean():.1f}")
            st.metric("Latest Score", f"{history_df['score'].iloc[-1]:.1f}")
    
    st.markdown("---")
    st.markdown("### 🎯 Features Analyzed")
    st.markdown("""
    **Video Analysis:**
    - ✅ Blink rate & duration
    - ✅ Gaze direction tracking
    - ✅ Micro-expression detection
    - ✅ Facial asymmetry
    - ✅ Lip compression
    - ✅ Head movement patterns
    
    **Audio Analysis:**
    - ✅ Speech rate & tempo
    - ✅ Pitch variation
    - ✅ Energy dynamics
    - ✅ Speech activity
    - ✅ Voice quality
    """)
    
    st.markdown("---")
    st.markdown("### 📖 Score Interpretation")
    st.markdown("""
    - **0-40**: Low deception probability 🟢
    - **40-60**: Possible deception 🟡
    - **60-100**: High deception probability 🔴
    
    *Video analysis carries 60% weight, Audio 40%*
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Ensure good lighting
    - Face should be clearly visible
    - Keep head relatively still
    - Speak clearly for audio analysis
    """)

# Main content - Tabs
tab1, tab2, tab3 = st.tabs(["📁 Upload & Analyze", "📊 Results", "📈 History"])

# Tab 1: Upload & Analyze
with tab1:
    st.markdown("### Upload File for Analysis")
    st.markdown("Supported formats: MP4, AVI, MOV (video) | MP3, WAV, M4A (audio)")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['mp4', 'avi', 'mov', 'mkv', 'mp3', 'wav', 'm4a'],
        help="Maximum file size: 200MB"
    )
    
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
            tmp_file.write(uploaded_file.read())
            file_path = tmp_file.name
        
        # Display file preview
        if file_extension in ['mp4', 'avi', 'mov', 'mkv']:
            st.video(file_path)
        else:
            st.audio(file_path)
        
        # Analysis button
        if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                is_video = file_extension in ['mp4', 'avi', 'mov', 'mkv']
                
                if is_video:
                    status_text.info("📹 Analyzing video...")
                    
                    # Process video
                    features, summary = process_video_file(
                        file_path, 
                        lambda p: progress_bar.progress(p)
                    )
                    video_score = summary['deception_score']
                    
                    # Extract and analyze audio from video
                    status_text.info("🎙️ Extracting and analyzing audio...")
                    progress_bar.progress(85)
                    
                    audio_analyzer = AudioAnalyzer()
                    audio_path = audio_analyzer.extract_audio_from_video(file_path)
                    
                    if audio_path and os.path.exists(audio_path):
                        audio_features, audio_score, transcript = process_audio_file(audio_path)
                        # Combined score (60% video, 40% audio)
                        combined_score = video_score * 0.6 + audio_score * 0.4
                        summary['deception_score'] = combined_score
                        summary['classification'] = (
                            'HIGH PROBABILITY OF DECEPTION' if combined_score >= 60 else
                            'POSSIBLE DECEPTION' if combined_score >= 40 else
                            'LOW PROBABILITY OF DECEPTION'
                        )
                        summary['audio_score'] = audio_score
                        os.unlink(audio_path)
                    else:
                        combined_score = video_score
                        transcript = ""
                    
                    results = {
                        'features': features,
                        'summary': summary,
                        'deception_score': combined_score,
                        'video_score': video_score,
                        'audio_score': summary.get('audio_score', 0),
                        'filename': uploaded_file.name,
                        'type': 'Video Analysis',
                        'transcript': transcript
                    }
                    
                else:
                    status_text.info("🎙️ Analyzing audio...")
                    progress_bar.progress(50)
                    
                    audio_features, audio_score, transcript = process_audio_file(file_path)
                    progress_bar.progress(100)
                    
                    classification = (
                        'HIGH PROBABILITY OF DECEPTION' if audio_score >= 60 else
                        'POSSIBLE DECEPTION' if audio_score >= 40 else
                        'LOW PROBABILITY OF DECEPTION'
                    )
                    
                    results = {
                        'audio_features': audio_features,
                        'summary': {
                            'deception_score': audio_score,
                            'classification': classification,
                            'color': '#dc3545' if audio_score >= 60 else '#ffc107' if audio_score >= 40 else '#28a745',
                            'icon': '🔴' if audio_score >= 60 else '🟡' if audio_score >= 40 else '🟢',
                            'indicators': []
                        },
                        'deception_score': audio_score,
                        'filename': uploaded_file.name,
                        'type': 'Audio Analysis',
                        'transcript': transcript
                    }
                
                # Save to session state
                st.session_state.analysis_results = results
                st.session_state.analysis_history.append({
                    'timestamp': datetime.now(),
                    'type': results['type'],
                    'filename': uploaded_file.name,
                    'score': results['deception_score'],
                    'classification': results['summary']['classification']
                })
                
                progress_bar.progress(100)
                status_text.success("✅ Analysis complete!")
                st.balloons()
                st.rerun()
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.info("Please ensure the file is not corrupted and try again.")
            
            finally:
                # Clean up temp file
                try:
                    os.unlink(file_path)
                except:
                    pass

# Tab 2: Results
with tab2:
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        # Main result card
        st.markdown(f"""
        <div class="result-card" style="text-align: center;">
            <span style="font-size: 64px;">{results['summary']['icon']}</span>
            <h1 style="color: {results['summary']['color']}; margin: 10px 0;">
                {results['summary']['classification']}
            </h1>
            <p style="color: rgba(255,255,255,0.7);">Based on {results['type'].lower()}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Score gauge
        gauge = create_gauge_chart(results['deception_score'], "Deception Score")
        st.plotly_chart(gauge, use_container_width=True)
        
        # Detailed metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎥 Video Analysis")
            if 'features' in results:
                f = results['features']
                st.metric("Blink Rate", f"{f.get('blink_rate', 0):.0f} blinks/min")
                st.metric("Gaze Aversion", f"{(f.get('gaze_left_ratio', 0) + f.get('gaze_right_ratio', 0)) * 100:.0f}%")
                st.metric("Micro-expressions", f"{f.get('micro_expression_frequency', 0):.2f}/sec")
                st.metric("Lip Compression", f"{f.get('avg_lip_compression', 0):.1f} px")
            if 'video_score' in results:
                st.metric("Video Deception Score", f"{results['video_score']:.0f}")
        
        with col2:
            st.markdown("### 🎙️ Audio Analysis")
            if 'audio_features' in results:
                af = results['audio_features']
                st.metric("Speech Tempo", f"{af.get('speech_tempo', 0):.0f} BPM")
                st.metric("Pitch Variation", f"{af.get('std_pitch', 0):.0f} Hz")
                st.metric("Speech Activity", f"{af.get('speech_activity', 0) * 100:.0f}%")
                st.metric("Energy Variation", f"{af.get('std_energy', 0):.3f}")
            if 'audio_score' in results:
                st.metric("Audio Deception Score", f"{results['audio_score']:.0f}")
        
        # Detected indicators
        if results['summary']['indicators']:
            st.markdown("### 📋 Detected Indicators")
            cols = st.columns(3)
            for i, (indicator, level) in enumerate(results['summary']['indicators']):
                badge_class = f"indicator-{level}"
                cols[i % 3].markdown(
                    f'<span class="indicator-badge {badge_class}">{indicator}</span>',
                    unsafe_allow_html=True
                )
        
        # Transcript
        if results.get('transcript'):
            st.markdown("### 📝 Speech Transcript")
            st.info(results['transcript'])
        
        # Export button
        st.markdown("---")
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            if st.button("📥 Export Results as CSV", use_container_width=True):
                export_data = {
                    'timestamp': datetime.now().isoformat(),
                    'filename': results.get('filename', 'unknown'),
                    'analysis_type': results['type'],
                    'deception_score': results['deception_score'],
                    'classification': results['summary']['classification']
                }
                
                if 'features' in results:
                    for k, v in results['features'].items():
                        export_data[f'video_{k}'] = v
                if 'audio_features' in results:
                    for k, v in results['audio_features'].items():
                        export_data[f'audio_{k}'] = v
                
                df = pd.DataFrame([export_data])
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download",
                    data=csv,
                    file_name=f"deception_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col_export2:
            if st.button("🔄 New Analysis", use_container_width=True):
                st.session_state.analysis_results = None
                st.rerun()
    
    else:
        st.info("No analysis results yet. Upload a file in the 'Upload & Analyze' tab to begin.")

# Tab 3: History
with tab3:
    if st.session_state.analysis_history:
        history_df = pd.DataFrame(st.session_state.analysis_history)
        
        st.markdown("### Analysis History")
        st.dataframe(
            history_df.sort_values('timestamp', ascending=False),
            use_container_width=True,
            hide_index=True
        )
        
        # Visualization
        st.markdown("### Score Trends")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history_df['timestamp'],
            y=history_df['score'],
            mode='lines+markers',
            name='Deception Score',
            line=dict(color='#667eea', width=3),
            marker=dict(size=10, color='#764ba2')
        ))
        fig.add_hline(y=40, line_dash="dash", line_color="#28a745", 
                      annotation_text="Truthful Threshold")
        fig.add_hline(y=60, line_dash="dash", line_color="#dc3545", 
                      annotation_text="Deceptive Threshold")
        fig.update_layout(
            title="Deception Score Over Time",
            xaxis_title="Date",
            yaxis_title="Score",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pie = go.Figure(data=[go.Pie(
                labels=history_df['classification'].value_counts().index,
                values=history_df['classification'].value_counts().values,
                marker=dict(colors=['#28a745', '#ffc107', '#dc3545']),
                hole=0.3
            )])
            fig_pie.update_layout(
                title="Classification Distribution",
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                height=350
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            if 'type' in history_df.columns:
                avg_by_type = history_df.groupby('type')['score'].mean().reset_index()
                fig_bar = px.bar(
                    avg_by_type, 
                    x='type', 
                    y='score',
                    title="Average Score by Analysis Type",
                    color='type',
                    color_discrete_sequence=['#667eea', '#764ba2']
                )
                fig_bar.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': 'white'},
                    height=350
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        
        if st.button("Clear History", use_container_width=True):
            st.session_state.analysis_history = []
            st.rerun()
    else:
        st.info("No analysis history yet. Upload and analyze a file to see history here.")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    '<p style="text-align: center; color: rgba(255,255,255,0.5);">'
    'Powered by AI & Computer Vision | Deception Detection System v2.0'
    '</p>',
    unsafe_allow_html=True
)