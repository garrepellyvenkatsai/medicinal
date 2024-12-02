from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os
import json
import sqlite3
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Paths
MODEL_PATH = 'models/medicinal_leaf_classifier.h5'
CLASS_INDICES_PATH = 'models/class_indices.json'
PLANT_INFO_PATH = 'models/plant_info.json'
UPLOAD_FOLDER = 'static/uploads'
DATABASE_PATH = 'plants.db'

# Load resources
model = load_model(MODEL_PATH)
with open(CLASS_INDICES_PATH, 'r') as f:
    class_indices = json.load(f)
with open(PLANT_INFO_PATH, 'r') as f:
    plant_info = json.load(f)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Database connection helper
def get_database_connection():
    return sqlite3.connect(DATABASE_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'password':
            session['logged_in'] = True
            return redirect(url_for('upload'))
        else:
            return render_template('login.html', error="Invalid credentials.")
    return render_template('login.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    if request.method == 'POST':
        file = request.files.get('leaf_image')
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            
            # Predict plant type
            img = load_img(filepath, target_size=(224, 224))
            img_array = np.expand_dims(img_to_array(img) / 255.0, axis=0)
            prediction = model.predict(img_array)
            class_idx = np.argmax(prediction)
            plant_name = list(class_indices.keys())[list(class_indices.values()).index(class_idx)]
            
            # Fetch plant details
            plant_details = plant_info.get(plant_name, {'uses': 'Unknown', 'benefits': 'Unknown', 'locations': 'Unknown'})
            
            # Fetch plant care details from the database
            conn = get_database_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM plant_care WHERE plant_name = ?", (plant_name,))
            plant_care = cursor.fetchone()
            
            # Fetch recipes from the database
            cursor.execute("SELECT recipe_name, instructions FROM recipes WHERE plant_name = ?", (plant_name,))
            recipes = [{"recipe_name": r[0], "instructions": r[1]} for r in cursor.fetchall()]
            conn.close()
            
            plant_care_details = {
                "sunlight": plant_care[1] if plant_care else "Unknown",
                "watering": plant_care[2] if plant_care else "Unknown",
                "soil": plant_care[3] if plant_care else "Unknown",
                "tips": plant_care[4] if plant_care else "Unknown"
            }
            
            return render_template(
                'result.html', 
                image_path=filepath, 
                plant_name=plant_name, 
                plant_details=plant_details, 
                plant_care_details=plant_care_details,
                recipes=recipes
            )
    return render_template('upload.html')

@app.route('/chart')
def chart():
    if not session.get('logged_in'):
        return redirect(url_for('login'))  # Redirect to login if not logged in
    return render_template('chart.html')

@app.route('/chart_data')
def chart_data():
    # Simulated accuracy data (replace with actual data from your models)
    accuracy_data = {
        'MobileNet': 0.90,
        'ResNet50': 0.85,
        'VGG16': 0.88,
        'Random Forest': 0.82,
        'SVM': 0.80
    }
    return jsonify(accuracy_data)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
