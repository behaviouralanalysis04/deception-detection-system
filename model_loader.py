import joblib
import numpy as np
import json
import os

class DeceptionModel:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.load_model()
    
    def load_model(self):
        """Load trained XGBoost model or create fallback"""
        try:
            # Try to load model
            if os.path.exists('model/xgboost_model.pkl'):
                self.model = joblib.load('model/xgboost_model.pkl')
                print("✅ Model loaded successfully")
            else:
                print("⚠️ No model found, using rule-based detection")
                self.model = None
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
        
        # Define feature columns
        self.feature_columns = [
            'blink_rate', 'avg_blink_duration', 'gaze_left_ratio', 'gaze_right_ratio',
            'gaze_center_ratio', 'avg_mouth_open_ratio', 'std_mouth_open_ratio',
            'avg_facial_asymmetry', 'std_facial_asymmetry', 'avg_lip_compression',
            'micro_expression_frequency', 'avg_head_pitch', 'std_head_pitch',
            'avg_head_roll', 'std_head_roll', 'avg_head_yaw', 'std_head_yaw',
            'head_nod_frequency', 'head_shake_frequency', 'head_tilt_frequency'
        ]
    
    def rule_based_score(self, features):
        """Rule-based deception scoring"""
        score = 0
        
        # Blink rate indicators
        blink_rate = features.get('blink_rate', 15)
        if blink_rate > 30:
            score += 15
        elif blink_rate < 10:
            score += 10
        
        # Gaze aversion
        gaze_aversion = features.get('gaze_left_ratio', 0) + features.get('gaze_right_ratio', 0)
        if gaze_aversion > 0.6:
            score += 20
        elif gaze_aversion > 0.4:
            score += 10
        
        # Lip compression
        lip_comp = features.get('avg_lip_compression', 10)
        if lip_comp > 15:
            score += 15
        
        # Facial asymmetry
        asymmetry = features.get('avg_facial_asymmetry', 10)
        if asymmetry > 20:
            score += 15
        
        # Micro-expressions
        micro = features.get('micro_expression_frequency', 0.5)
        if micro > 2:
            score += 20
        elif micro > 1:
            score += 10
        
        # Head movements
        nods = features.get('head_nod_frequency', 0)
        if nods > 1.5:
            score += 5
        
        # Clamp to 0-100
        return min(100, max(0, score))
    
    def predict(self, features_dict):
        """Predict deception probability (0-100)"""
        if self.model is not None:
            try:
                # Prepare feature vector
                feature_vector = []
                for col in self.feature_columns:
                    feature_vector.append(features_dict.get(col, 0))
                
                feature_array = np.array([feature_vector])
                
                # Handle NaN values
                feature_array = np.nan_to_num(feature_array, nan=0.0)
                
                # Get prediction probability (assuming binary classification)
                if hasattr(self.model, 'predict_proba'):
                    proba = self.model.predict_proba(feature_array)[0][1]
                    return proba * 100
                elif hasattr(self.model, 'predict'):
                    pred = self.model.predict(feature_array)[0]
                    return 100 if pred == 1 else 0
                else:
                    return self.rule_based_score(features_dict)
            except Exception as e:
                print(f"Model prediction error: {e}")
                return self.rule_based_score(features_dict)
        else:
            return self.rule_based_score(features_dict)
    
    def get_feature_importance(self):
        """Get feature importance if available"""
        if self.model and hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            if len(importances) == len(self.feature_columns):
                return dict(zip(self.feature_columns, importances))
        return None