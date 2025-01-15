import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_realistic_data(num_samples=10000):
    data = []
    
    # Define base probabilities
    base_dropout_prob = 0.3  # Base probability of dropout
    
    for _ in range(num_samples):
        # Generate basic student information
        age = np.random.randint(12, 25)
        gender = np.random.choice(['Male', 'Female'], p=[0.48, 0.52])
        special_needs = np.random.choice([True, False], p=[0.15, 0.85])
        
        # Generate socioeconomic factors
        household_income = np.random.choice(['Low', 'Medium', 'High'], p=[0.6, 0.3, 0.1])
        household_size = np.random.randint(2, 15)
        electricity_access = np.random.choice([True, False], p=[0.4, 0.6])
        internet_access = np.random.choice([True, False], p=[0.3, 0.7])
        parent_education = np.random.choice(['None', 'Primary', 'Secondary', 'Tertiary'], p=[0.3, 0.4, 0.2, 0.1])
        
        # Generate location and political factors
        distance_to_school = np.random.uniform(0.5, 30)  # kilometers
        political_state = np.random.choice(['Stable', 'Conflict'], p=[0.7, 0.3])
        
        # Generate school-related factors
        num_teachers = np.random.randint(5, 50)
        student_teacher_ratio = np.random.randint(20, 80)
        has_computer_lab = np.random.choice([True, False], p=[0.2, 0.8])
        has_library = np.random.choice([True, False], p=[0.3, 0.7])
        
        # Generate language factors
        language_spoken = np.random.choice(['French', 'English', 'Fulfulde', 'Pidgin', 'Local Dialects'])
        school_language = np.random.choice(['French', 'English'])
        
        # Generate support factors
        aid_coverage = np.random.choice([True, False], p=[0.4, 0.6])
        parental_involvement = np.random.choice([True, False], p=[0.5, 0.5])
        community_support = np.random.choice([True, False], p=[0.4, 0.6])
        
        # Generate teacher factors
        qualification = np.random.choice(["Bachelor's Degree", "Master's Degree", "PhD", "Diploma"], 
                                      p=[0.5, 0.3, 0.1, 0.1])
        years_of_experience = np.random.randint(0, 30)
        
        # Generate academic factors
        previous_academic_performance = np.random.choice(['Excellent', 'Good', 'Average', 'Poor'])
        attendance_rate = np.random.uniform(0.4, 1.0)
        
        # Generate marital status and children
        marital_status = np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'])
        num_children = np.random.randint(0, 5)
        
        # Calculate dropout probability based on complex conditions
        dropout_prob = base_dropout_prob
        
        # Age and academic factors
        if age > 18 and attendance_rate < 0.7 and previous_academic_performance == 'Poor':
            dropout_prob += 0.3
            
        # Gender and socioeconomic factors
        if gender == 'Female' and household_size > 6 and household_income == 'Low' and not electricity_access:
            dropout_prob += 0.25
            
        # Marriage and support factors
        if marital_status in ['Married', 'Divorced'] and attendance_rate < 0.7 and not community_support:
            dropout_prob += 0.2
            
        # Special needs factors
        if special_needs and not electricity_access and previous_academic_performance == 'Poor':
            dropout_prob += 0.3
            
        # Language barriers
        if language_spoken != school_language and attendance_rate < 0.7:
            dropout_prob += 0.2
            
        # Socioeconomic combined factors
        if household_income == 'Low' and household_size > 8 and parent_education == 'None' and not electricity_access:
            dropout_prob += 0.25
            
        # Political and distance factors
        if political_state == 'Conflict' and distance_to_school > 10 and household_income == 'Low':
            dropout_prob += 0.3
            
        # Technology access
        if not internet_access and not has_computer_lab and attendance_rate < 0.7:
            dropout_prob += 0.15
            
        # School resource factors
        if student_teacher_ratio > 50 and num_teachers < 10 and attendance_rate < 0.7:
            dropout_prob += 0.2
            
        # Support network factors
        if not parental_involvement and not community_support and not aid_coverage:
            dropout_prob += 0.25
            
        # Complex combined conditions
        if (gender == 'Female' and age > 18 and attendance_rate < 0.7 and 
            household_size > 6 and political_state == 'Conflict'):
            dropout_prob += 0.35
            
        if (language_spoken != school_language and distance_to_school > 10 and 
            not has_library and household_income == 'Low'):
            dropout_prob += 0.3
            
        # Normalize probability and determine dropout status
        dropout_prob = min(max(dropout_prob, 0), 1)
        dropout_status = np.random.choice([True, False], p=[dropout_prob, 1-dropout_prob])
        
        # Create record
        record = {
            'age': age,
            'gender': gender,
            'special_needs': special_needs,
            'num_children': num_children,
            'previous_academic_performance': previous_academic_performance,
            'attendance_rate': attendance_rate,
            'language_spoken': language_spoken,
            'marital_status': marital_status,
            'household_income': household_income,
            'parent_education': parent_education,
            'electricity_access': electricity_access,
            'internet_access': internet_access,
            'household_size': household_size,
            'political_state': political_state,
            'num_teachers': num_teachers,
            'student_teacher_ratio': student_teacher_ratio,
            'has_computer_lab': has_computer_lab,
            'has_library': has_library,
            'school_language': school_language,
            'distance_to_school': distance_to_school,
            'qualification': qualification,
            'years_of_experience': years_of_experience,
            'aid_coverage': aid_coverage,
            'parental_involvement': parental_involvement,
            'community_support': community_support,
            'dropout_status': dropout_status
        }
        
        data.append(record)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv('data/synthetic_dropout_data.csv', index=False)
    print(f"Generated {num_samples} samples with realistic dropout patterns")
    print(f"Dropout rate: {df['dropout_status'].mean():.2%}")
    
    return df

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    import os
    os.makedirs('data', exist_ok=True)
    
    # Generate data
    df = generate_realistic_data()
    
    # Print summary statistics
    print("\nData Summary:")
    print(f"Total samples: {len(df)}")
    print(f"Features: {df.columns.tolist()}")
    print("\nDropout Statistics:")
    print(df['dropout_status'].value_counts(normalize=True))