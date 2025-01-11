import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import json
from flask import Flask, request, jsonify, render_template
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DropoutPredictor:
    def __init__(self):
        """Initialize the predictor with necessary encoders and scalers"""
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        self.categorical_columns = [
            'gender', 'previous_academic_performance', 'language_spoken',
            'marital_status', 'qualification', 'household_income',
            'parent_education', 'school_language'
        ]
        self.boolean_columns = [
            'special_needs', 'electricity_access', 'internet_access',
            'has_computer_lab', 'has_library', 'parental_involvement',
            'community_support'
        ]
        self.numerical_columns = [
            'age', 'num_children', 'attendance_rate', 'years_of_experience',
            'household_size', 'student_teacher_ratio', 'distance_to_school'
        ]

    def connect_to_db(self):
        """Create database connection"""
        try:
            connection_string = "postgresql://postgres:root@localhost:5432/Dropouts"
            return create_engine(connection_string)
        except Exception as e:
            logger.error(f"Database connection error: {str(e)}")
            raise

    def fetch_data(self):
        """Fetch and prepare training data"""
        engine = self.connect_to_db()
        query = """
        SELECT 
            f.dropout_status,
            st.age, st.gender, st.special_needs, st.num_children,
            st.previous_academic_performance, st.attendance_rate,
            st.language_spoken, st.marital_status,
            t.qualification, t.years_of_experience,
            se.household_income, se.parent_education, se.electricity_access,
            se.internet_access, se.household_size,
            sc.student_teacher_ratio, sc.has_computer_lab, sc.has_library,
            sc.school_language,
            l.distance_to_school,
            cs.parental_involvement, cs.community_support
        FROM fact_dropouts f
        JOIN dim_student st ON f.student_key = st.student_key
        JOIN dim_teacher t ON f.teacher_key = t.teacher_key
        JOIN dim_socioeconomic se ON f.socioeconomic_key = se.socioeconomic_key
        JOIN dim_school sc ON f.school_key = sc.school_key
        JOIN dim_location l ON f.location_key = l.location_key
        JOIN dim_community_support cs ON f.community_key = cs.community_key;
        """
        return pd.read_sql_query(query, engine)

    def preprocess_data(self, df, training=True):
        """Preprocess the data with advanced feature engineering"""
        # Handle missing values
        imputer = SimpleImputer(strategy='most_frequent')
        
        # Encode categorical variables
        if training:
            for col in self.categorical_columns:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
        else:
            for col in self.categorical_columns:
                df[col] = self.label_encoders[col].transform(df[col])

        # Convert boolean columns
        for col in self.boolean_columns:
            df[col] = df[col].astype(int)

        # Feature engineering
        df['attendance_risk'] = df['attendance_rate'].apply(
            lambda x: 'high' if x < 70 else ('medium' if x < 85 else 'low')
        )
        df['distance_risk'] = df['distance_to_school'].apply(
            lambda x: 'high' if x > 10 else ('medium' if x > 5 else 'low')
        )
        
        # Create interaction features
        df['socioeconomic_score'] = (
            df['household_income'].astype(int) + 
            df['electricity_access'].astype(int) + 
            df['internet_access'].astype(int)
        )
        
        df['school_resource_score'] = (
            df['has_computer_lab'].astype(int) + 
            df['has_library'].astype(int)
        )
        
        # Scale numerical features
        if training:
            self.scaler.fit(df[self.numerical_columns])
        
        df[self.numerical_columns] = self.scaler.transform(df[self.numerical_columns])
        
        return df

    def build_model(self):
        """Build an ensemble model with multiple classifiers"""
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42
        )
        
        gb = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        mlp = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42
        )
        
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('gb', gb),
                ('mlp', mlp)
            ],
            voting='soft'
        )
        
        return ensemble

    def train_model(self):
        """Train the model with cross-validation and hyperparameter tuning"""
        # Fetch and preprocess data
        df = self.fetch_data()
        processed_df = self.preprocess_data(df)
        
        # Prepare features and target
        X = processed_df.drop(['dropout_status', 'attendance_risk', 'distance_risk'], axis=1)
        y = processed_df['dropout_status']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Build and train model
        self.model = self.build_model()
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        # Save feature names
        self.feature_names = X.columns.tolist()
        
        return results

    def generate_recommendations(self, student_data):
        """Generate detailed recommendations based on prediction and feature values"""
        recommendations = []
        risk_factors = []
        
        # Convert student data to DataFrame
        student_df = pd.DataFrame([student_data])
        processed_data = self.preprocess_data(student_df, training=False)
        
        # Get prediction and probability
        dropout_prob = self.model.predict_proba(processed_data)[0][1]
        
        # Risk level determination
        risk_level = 'High' if dropout_prob > 0.7 else 'Medium' if dropout_prob > 0.3 else 'Low'
        
        # Analyze individual factors
        if student_data['attendance_rate'] < 85:
            risk_factors.append({
                'factor': 'Attendance',
                'severity': 'High' if student_data['attendance_rate'] < 70 else 'Medium',
                'recommendation': 'Implement daily attendance monitoring and establish communication with parents/guardians.'
            })
            
        if student_data['distance_to_school'] > 5:
            risk_factors.append({
                'factor': 'Distance to School',
                'severity': 'High' if student_data['distance_to_school'] > 10 else 'Medium',
                'recommendation': 'Consider transportation assistance or remote learning options.'
            })
            
        if not student_data['internet_access']:
            risk_factors.append({
                'factor': 'Internet Access',
                'severity': 'Medium',
                'recommendation': 'Provide information about public internet access points or subsidized internet programs.'
            })
            
        if student_data['previous_academic_performance'] == 'poor':
            risk_factors.append({
                'factor': 'Academic Performance',
                'severity': 'High',
                'recommendation': 'Arrange for additional tutoring and academic support services.'
            })
            
        # Generate comprehensive recommendations
        recommendations = {
            'risk_level': risk_level,
            'dropout_probability': f"{dropout_prob:.2%}",
            'risk_factors': risk_factors,
            'general_recommendations': [
                "Regular monitoring of attendance and academic progress",
                "Engagement with parents/guardians through regular meetings",
                "Connection with community support services",
                "Access to academic counseling and career guidance"
            ],
            'intervention_plan': {
                'immediate_actions': [rf['recommendation'] for rf in risk_factors if rf['severity'] == 'High'],
                'medium_term_actions': [rf['recommendation'] for rf in risk_factors if rf['severity'] == 'Medium'],
                'long_term_support': [
                    "Develop personalized learning plan",
                    "Regular progress reviews",
                    "Mentorship program participation"
                ]
            }
        }
        
        return recommendations

    def save_model(self, filepath='dropout_prediction_model.joblib'):
        """Save the trained model and associated transformers"""
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath='dropout_prediction_model.joblib'):
        """Load the trained model and associated transformers"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.label_encoders = model_data['label_encoders']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        logger.info(f"Model loaded from {filepath}")

# Flask Application
app = Flask(__name__)
predictor = DropoutPredictor()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        recommendations = predictor.generate_recommendations(data)
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    try:
        results = predictor.train_model()
        predictor.save_model()
        return jsonify({
            'message': 'Model trained successfully',
            'results': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def main():
    """Main function to train and save the model"""
    predictor = DropoutPredictor()
    
    # Train the model
    logger.info("Training model...")
    results = predictor.train_model()
    
    # Print results
    logger.info("\nModel Performance:")
    logger.info(f"Accuracy: {results['accuracy']:.2%}")
    logger.info(f"ROC AUC: {results['roc_auc']:.2%}")
    logger.info("\nClassification Report:")
    logger.info(results['classification_report'])
    
    # Save the model
    predictor.save_model()
    
    return predictor

if __name__ == "__main__":
    # Train and save the model first
    predictor = main()
    
    # Run the Flask application
    app.run(debug=True)