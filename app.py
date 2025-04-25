from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = Flask(__name__)

# Load the dataset
@app.route('/load_data')
def load_data():
    global data, soil_encoder, crop_encoder, fertilizer_encoder
    
    # Check if models are already trained
    if os.path.exists('models/classifier.pkl'):
        return jsonify({'status': 'Data already loaded and models trained'})
    
    # Load dataset
    data = pd.read_csv('static/data/f2.csv')
    
    # Encode categorical variables
    soil_encoder = LabelEncoder()
    crop_encoder = LabelEncoder()
    fertilizer_encoder = LabelEncoder()
    
    data['Soil_Type'] = soil_encoder.fit_transform(data['Soil_Type'])
    data['Crop_Type'] = crop_encoder.fit_transform(data['Crop_Type'])
    data['Fertilizer'] = fertilizer_encoder.fit_transform(data['Fertilizer'])
    
    # Save encoders
    os.makedirs('models', exist_ok=True)
    with open('models/soil_encoder.pkl', 'wb') as f:
        pickle.dump(soil_encoder, f)
    with open('models/crop_encoder.pkl', 'wb') as f:
        pickle.dump(crop_encoder, f)
    with open('models/fertilizer_encoder.pkl', 'wb') as f:
        pickle.dump(fertilizer_encoder, f)
    
    # Train models
    train_models()
    
    return jsonify({'status': 'Data loaded and models trained successfully'})

# Train machine learning models
def train_models():
    # Split data
    X = data.drop('Fertilizer', axis=1)
    y = data['Fertilizer']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Train Random Forest for fertilizer recommendation
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Save model
    with open('models/classifier.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    
    # Train model for crop recommendation (using all features except Crop_Type)
    X_crop = data.drop(['Crop_Type', 'Fertilizer'], axis=1)
    y_crop = data['Crop_Type']
    
    X_train_crop, X_test_crop, y_train_crop, y_test_crop = train_test_split(X_crop, y_crop, test_size=0.2, random_state=42)
    
    # Normalize data
    scaler_crop = MinMaxScaler()
    X_train_crop_scaled = scaler_crop.fit_transform(X_train_crop)
    X_test_crop_scaled = scaler_crop.transform(X_test_crop)
    
    # Save scaler
    with open('models/scaler_crop.pkl', 'wb') as f:
        pickle.dump(scaler_crop, f)
    
    # Train Random Forest for crop recommendation
    rf_model_crop = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model_crop.fit(X_train_crop_scaled, y_train_crop)
    
    # Save model
    with open('models/crop_classifier.pkl', 'wb') as f:
        pickle.dump(rf_model_crop, f)

# Load trained models and encoders
def load_models():
    models = {}
    try:
        with open('models/classifier.pkl', 'rb') as f:
            models['fertilizer'] = pickle.load(f)
        with open('models/crop_classifier.pkl', 'rb') as f:
            models['crop'] = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            models['scaler'] = pickle.load(f)
        with open('models/scaler_crop.pkl', 'rb') as f:
            models['scaler_crop'] = pickle.load(f)
        with open('models/soil_encoder.pkl', 'rb') as f:
            models['soil_encoder'] = pickle.load(f)
        with open('models/crop_encoder.pkl', 'rb') as f:
            models['crop_encoder'] = pickle.load(f)
        with open('models/fertilizer_encoder.pkl', 'rb') as f:
            models['fertilizer_encoder'] = pickle.load(f)
        return models
    except Exception as e:
        print(f"Error loading models: {e}")
        return None

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load models
        models = load_models()
        if not models:
            return jsonify({'error': 'Models not loaded. Please load data first.'})
        
        # Get input data
        data = request.json
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        moisture = float(data['moisture'])
        soil_type = data['soil_type']
        nitrogen = float(data['nitrogen'])
        potassium = float(data['potassium'])
        phosphorous = float(data['phosphorous'])
        
        # Encode soil type
        soil_type_encoded = models['soil_encoder'].transform([soil_type])[0]
        
        # Prepare data for crop prediction
        crop_input = np.array([[temperature, humidity, moisture, soil_type_encoded, 
                               nitrogen, potassium, phosphorous]])
        
        # Scale input
        crop_input_scaled = models['scaler_crop'].transform(crop_input)
        
        # Predict crop
        crop_prediction = models['crop'].predict(crop_input_scaled)[0]
        crop_name = models['crop_encoder'].inverse_transform([crop_prediction])[0]
        
        # Get crop probabilities
        crop_probs = models['crop'].predict_proba(crop_input_scaled)[0]
        crop_classes = models['crop_encoder'].inverse_transform(range(len(crop_probs)))
        
        # Get top 3 crops
        top_crop_indices = crop_probs.argsort()[-3:][::-1]
        top_crops = [{'name': models['crop_encoder'].inverse_transform([idx])[0], 
                     'probability': float(crop_probs[idx])} 
                    for idx in top_crop_indices]
        
        # Prepare data for fertilizer prediction (including predicted crop)
        fertilizer_input = np.array([[temperature, humidity, moisture, soil_type_encoded, 
                                     crop_prediction, nitrogen, potassium, phosphorous]])
        
        # Scale input
        fertilizer_input_scaled = models['scaler'].transform(fertilizer_input)
        
        # Predict fertilizer
        fertilizer_prediction = models['fertilizer'].predict(fertilizer_input_scaled)[0]
        fertilizer_name = models['fertilizer_encoder'].inverse_transform([fertilizer_prediction])[0]
        
        # Get fertilizer probabilities
        fertilizer_probs = models['fertilizer'].predict_proba(fertilizer_input_scaled)[0]
        fertilizer_classes = models['fertilizer_encoder'].inverse_transform(range(len(fertilizer_probs)))
        
        # Get top 3 fertilizers
        top_fertilizer_indices = fertilizer_probs.argsort()[-3:][::-1]
        top_fertilizers = [{'name': models['fertilizer_encoder'].inverse_transform([idx])[0], 
                           'probability': float(fertilizer_probs[idx])} 
                          for idx in top_fertilizer_indices]
        
        # Calculate sustainability score (example algorithm)
        sustainability_score = calculate_sustainability_score(temperature, humidity, moisture, 
                                                             soil_type_encoded, crop_prediction,
                                                             nitrogen, potassium, phosphorous)
        
        # Calculate productivity score (example algorithm)
        productivity_score = calculate_productivity_score(temperature, humidity, moisture, 
                                                         soil_type_encoded, crop_prediction,
                                                         nitrogen, potassium, phosphorous,
                                                         fertilizer_prediction)
        
        return jsonify({
            'crop': crop_name,
            'top_crops': top_crops,
            'fertilizer': fertilizer_name,
            'top_fertilizers': top_fertilizers,
            'sustainability_score': sustainability_score,
            'productivity_score': productivity_score
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Calculate sustainability score
def calculate_sustainability_score(temperature, humidity, moisture, soil_type, crop_type,
                                  nitrogen, potassium, phosphorous):
    # Example algorithm - this would be refined based on agricultural science
    # Higher score is better (0-100)
    
    # Penalize excessive fertilizer use
    fertilizer_penalty = (nitrogen + potassium + phosphorous) / 3
    if fertilizer_penalty > 50:
        fertilizer_penalty = (fertilizer_penalty - 50) * 0.5
    else:
        fertilizer_penalty = 0
    
    # Reward appropriate moisture levels
    moisture_score = 0
    if 30 <= moisture <= 60:
        moisture_score = 25
    elif 20 <= moisture < 30 or 60 < moisture <= 70:
        moisture_score = 15
    
    # Base score
    base_score = 75
    
    # Calculate final score
    sustainability_score = base_score + moisture_score - fertilizer_penalty
    
    # Ensure score is between 0 and 100
    sustainability_score = max(0, min(100, sustainability_score))
    
    return round(sustainability_score, 1)

# Calculate productivity score
def calculate_productivity_score(temperature, humidity, moisture, soil_type, crop_type,
                                nitrogen, potassium, phosphorous, fertilizer_type):
    # Example algorithm - this would be refined based on agricultural science
    # Higher score is better (0-100)
    
    # Base score
    base_score = 60
    
    # Adjust based on NPK levels
    npk_score = min(40, (nitrogen + potassium + phosphorous) / 5)
    
    # Adjust based on moisture
    moisture_score = 0
    if 30 <= moisture <= 60:
        moisture_score = 20
    elif 20 <= moisture < 30 or 60 < moisture <= 70:
        moisture_score = 10
    
    # Calculate final score
    productivity_score = base_score + npk_score + moisture_score
    
    # Ensure score is between 0 and 100
    productivity_score = max(0, min(100, productivity_score))
    
    return round(productivity_score, 1)

# Data visualization
@app.route('/visualize')
def visualize():
    return render_template('visualize.html')

# API endpoint for getting visualization data
@app.route('/api/visualization/<viz_type>')
def get_visualization(viz_type):
    try:
        # Load dataset
        data = pd.read_csv('static/data/f2.csv')
        
        plt.figure(figsize=(10, 6))
        
        if viz_type == 'crop_distribution':
            # Create crop distribution plot
            sns.countplot(data=data, x='Crop_Type')
            plt.title('Crop Type Distribution')
            plt.xlabel('Crop Type')
            plt.ylabel('Count')
            plt.xticks(rotation=90)
            
        elif viz_type == 'fertilizer_distribution':
            # Create fertilizer distribution plot
            sns.countplot(data=data, x='Fertilizer')
            plt.title('Fertilizer Distribution')
            plt.xlabel('Fertilizer Type')
            plt.ylabel('Count')
            plt.xticks(rotation=90)
            
        elif viz_type == 'soil_crop_relation':
            # Create soil-crop relation plot
            plt.figure(figsize=(12, 6))
            soil_crop_counts = data.groupby(['Soil_Type', 'Crop_Type']).size().unstack(fill_value=0)
            soil_crop_counts.plot(kind='bar', stacked=True)
            plt.title('Crop Types by Soil Type')
            plt.xlabel('Soil Type')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            
        elif viz_type == 'npk_distribution':
            # Create NPK distribution plot
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            sns.histplot(data=data, x='Nitrogen', kde=True)
            plt.title('Nitrogen Distribution')
            
            plt.subplot(1, 3, 2)
            sns.histplot(data=data, x='Phosphorous', kde=True)
            plt.title('Phosphorous Distribution')
            
            plt.subplot(1, 3, 3)
            sns.histplot(data=data, x='Potassium', kde=True)
            plt.title('Potassium Distribution')
            
        else:
            return jsonify({'error': 'Invalid visualization type'})
        
        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Encode the image to base64
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return jsonify({'image': img_str})
    
    except Exception as e:
        return jsonify({'error': str(e)})

# About page
@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('static/data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Check if dataset exists, if not, create a copy
    if not os.path.exists('static/data/f2.csv'):
        # This assumes the dataset is in the root directory
        if os.path.exists('f2.csv'):
            import shutil
            shutil.copy('f2.csv', 'static/data/f2.csv')
    
    app.run(debug=True)
