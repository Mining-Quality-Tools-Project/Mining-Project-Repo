import joblib
import pandas as pd
import numpy as np
from config import MODEL_PATH

def calculate_risk_score(features):
    """Calculate a detailed risk score based on complex feature combinations"""
    risk_score = 0
    high_risk_conditions = []
    
    # Age and Performance Related
    if features['age'] > 18 and features['attendance_rate'] < 0.7 and features['previous_academic_performance'] == 'poor':
        risk_score += 0.8
        high_risk_conditions.append({
            'category': 'Age and Performance',
            'risk': 'High age combined with poor attendance and performance',
            'severity': 'Critical'
        })
    
    # Gender and Socioeconomic
    if (features['gender'] == 'Female' and 
        features['household_size'] > 6 and 
        features['household_income'] == 'low' and 
        not features['electricity_access']):
        risk_score += 0.7
        high_risk_conditions.append({
            'category': 'Gender and Resources',
            'risk': 'Female student in large, low-income household without electricity',
            'severity': 'High'
        })
    
    # Marriage and Children
    if features['marital_status'] == 'Married' and features['attendance_rate'] < 0.7:
        risk_score += 0.6
        high_risk_conditions.append({
            'category': 'Family Responsibilities',
            'risk': 'Married student with attendance issues',
            'severity': 'High'
        })
    
    # Language Barriers
    if (features['language_spoken'] != features['school_language'] and 
        features['attendance_rate'] < 0.7):
        risk_score += 0.6
        high_risk_conditions.append({
            'category': 'Language Barrier',
            'risk': 'Language mismatch affecting attendance and performance',
            'severity': 'High'
        })
    
    # Distance and Transportation
    if features['distance_to_school'] > 15:
        risk_score += 0.5
        severity = 'Critical' if features['distance_to_school'] > 20 else 'High'
        high_risk_conditions.append({
            'category': 'Access',
            'risk': f'Long distance to school ({features["distance_to_school"]} km)',
            'severity': severity
        })
    
    # School Resources
    if (features['student_teacher_ratio'] > 40 and 
        features['num_teachers'] < 10):
        risk_score += 0.5
        high_risk_conditions.append({
            'category': 'School Resources',
            'risk': 'Overcrowded classrooms with few teachers',
            'severity': 'High'
        })
    
    # Support Systems
    if not features['parental_involvement'] and not features['community_support']:
        risk_score += 0.4
        high_risk_conditions.append({
            'category': 'Support Network',
            'risk': 'Lack of parental and community support',
            'severity': 'High'
        })
    
    # Political Instability
    if features['political_state'] == 'Conflict':
        risk_score += 0.6
        high_risk_conditions.append({
            'category': 'Regional Stability',
            'risk': 'Located in conflict zone',
            'severity': 'Critical'
        })
    
    # Technology Access
    if not features['internet_access'] and not features['has_computer_lab']:
        risk_score += 0.3
        high_risk_conditions.append({
            'category': 'Digital Access',
            'risk': 'No access to internet or computer resources',
            'severity': 'Medium'
        })
    
    return min(1.0, risk_score), high_risk_conditions

def generate_recommendations(features, dropout_prob, risk_conditions):
    """Generate detailed recommendations based on risk factors"""
    risk_level = "Critical" if dropout_prob > 0.8 else "High" if dropout_prob > 0.6 else "Medium" if dropout_prob > 0.4 else "Low"
    
    intervention_plan = {
        'immediate_actions': [],
        'medium_term_actions': [],
        'long_term_actions': []
    }
    
    # Always provide recommendations regardless of risk level
    if risk_level == "Low":
        intervention_plan['immediate_actions'].extend([
            "Schedule regular check-ins with teachers",
            "Monitor attendance patterns",
            "Encourage participation in extracurricular activities"
        ])
        intervention_plan['medium_term_actions'].extend([
            "Set up peer study groups",
            "Develop academic goals with student",
            "Regular parent-teacher communication"
        ])
        intervention_plan['long_term_actions'].extend([
            "Create personal development plan",
            "Build strong support network",
            "Plan for future academic transitions"
        ])
    
    # Add specific recommendations based on risk conditions
    for condition in risk_conditions:
        if condition['category'] == 'Age and Performance':
            intervention_plan['immediate_actions'].extend([
                "Implement intensive tutoring program",
                "Create flexible attendance schedule",
                "Provide catch-up learning materials"
            ])
        
        elif condition['category'] == 'Gender and Resources':
            intervention_plan['immediate_actions'].extend([
                "Connect with women's education support programs",
                "Provide solar-powered study lights",
                "Arrange safe transportation options"
            ])
        
        elif condition['category'] == 'Language Barrier':
            intervention_plan['immediate_actions'].extend([
                "Provide bilingual learning materials",
                "Assign language buddy",
                "Extra language support classes"
            ])
        
        elif condition['category'] == 'Access':
            intervention_plan['immediate_actions'].extend([
                "Organize shared transportation",
                "Provide distance learning materials",
                "Consider satellite classroom options"
            ])
        
        elif condition['category'] == 'School Resources':
            intervention_plan['medium_term_actions'].extend([
                "Implement peer tutoring program",
                "Create smaller study groups",
                "Provide additional learning resources"
            ])
        
        elif condition['category'] == 'Support Network':
            intervention_plan['immediate_actions'].extend([
                "Connect with local mentorship programs",
                "Schedule regular family meetings",
                "Link with community support services"
            ])
        
        elif condition['category'] == 'Regional Stability':
            intervention_plan['immediate_actions'].extend([
                "Develop emergency education plan",
                "Connect with humanitarian aid services",
                "Provide mobile learning resources"
            ])
    
    # Remove duplicates while preserving order
    for key in intervention_plan:
        intervention_plan[key] = list(dict.fromkeys(intervention_plan[key]))
    
    return {
        "risk_level": risk_level,
        "dropout_probability": float(dropout_prob),
        "risk_factors": risk_conditions,
        "intervention_plan": intervention_plan
    }

def predict_dropout(student_data):
    # Load model and preprocessor
    saved_model = joblib.load(MODEL_PATH)
    model = saved_model['model']
    preprocessor = saved_model['preprocessor']
    
    # Calculate risk score based on complex conditions
    risk_score, risk_conditions = calculate_risk_score(student_data)
    
    # Preprocess input data
    X = preprocessor.transform(pd.DataFrame([student_data]))
    
    # Make prediction
    model_prob = model.predict_proba(X)[0]
    
    # Combine model probability with rule-based risk score
    final_prob = 0.7 * model_prob + 0.3 * risk_score
    
    # Generate detailed recommendations
    return generate_recommendations(student_data, final_prob, risk_conditions)