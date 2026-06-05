#!/bin/bash

echo "Setting up Deception Detection System..."

# Create necessary directories
mkdir -p model
mkdir -p temp_uploads
mkdir -p static

# Download a pre-trained face detection model (Haar Cascade) - lightweight alternative to dlib
if [ ! -f "haarcascade_frontalface_default.xml" ]; then
    echo "Downloading Haar Cascade for face detection..."
    wget -q https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
    echo "✅ Haar Cascade downloaded"
fi

# Create default model if not exists (fallback)
if [ ! -f "model/xgboost_model.pkl" ]; then
    echo "Creating fallback model configuration..."
    python -c "
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Create a simple fallback model
X_dummy = np.random.randn(100, 20)
y_dummy = np.random.randint(0, 2, 100)
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_dummy, y_dummy)
joblib.dump(model, 'model/xgboost_model.pkl')
print('✅ Fallback model created')
"
fi

echo "✅ Setup complete!"