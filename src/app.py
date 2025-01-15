from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from predict import predict_dropout

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'models/ensemble_mod.joblib'
model_data = joblib.load(MODEL_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json
        
        # Convert boolean strings to actual booleans
        boolean_fields = [
            'special_needs', 'electricity_access', 'internet_access',
            'has_computer_lab', 'has_library', 'aid_coverage',
            'parental_involvement', 'community_support'
        ]
        for field in boolean_fields:
            if field in data:
                data[field] = data[field].lower() == 'true'
        
        # Convert numeric strings to numbers
        numeric_fields = [
            'age', 'num_children', 'attendance_rate', 'household_size',
            'num_teachers', 'student_teacher_ratio', 'distance_to_school',
            'years_of_experience'
        ]
        for field in numeric_fields:
            if field in data:
                data[field] = float(data[field])
        
        # Make prediction
        result = predict_dropout(data)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)