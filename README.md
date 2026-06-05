# 🎭 Deception Detection System

AI-powered system for analyzing facial expressions and speech patterns to detect potential deception.

## Features

- **Video Analysis**: Blink rate, gaze direction, micro-expressions, facial asymmetry, lip compression, head movements
- **Audio Analysis**: Speech tempo, pitch variation, energy dynamics, speech activity
- **Combined Scoring**: 60% video + 40% audio weighted analysis
- **Real-time Results**: Instant feedback with visual indicators
- **Export Capability**: Download results as CSV

## Live Demo

[Deployed on Render](https://your-app.onrender.com)

## Deployment Instructions

### Deploy on Render

1. Fork this repository
2. Go to [Render.com](https://render.com)
3. Click "New +" → "Web Service"
4. Connect your GitHub repository
5. Use the following settings:
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt && chmod +x setup.sh && ./setup.sh`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
6. Click "Create Web Service"

### Local Development

```bash
# Clone repository
git clone https://github.com/yourusername/deception-detection.git
cd deception-detection

# Install dependencies
pip install -r requirements.txt

# Download face cascade
wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml

# Run app
streamlit run app.py