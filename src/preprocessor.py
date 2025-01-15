import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

class DropoutPreprocessor:
    def __init__(self):
        self.column_transformer = None
        self.feature_names = None
        
        # Updated features based on actual database schema
        self.numeric_features = [
            'age', 
            'num_children',
            'attendance_rate',
            'household_size',
            'num_teachers',
            'student_teacher_ratio',
            'distance_to_school',
            'years_of_experience'
        ]
        
        self.binary_features = [
            'special_needs',
            'electricity_access',
            'internet_access',
            'has_computer_lab',
            'has_library',
            'aid_coverage',
            'parental_involvement',
            'community_support'
        ]
        
        self.categorical_features = [
            'gender',
            'language_spoken',
            'marital_status',
            'previous_academic_performance',
            'household_income',
            'parent_education',
            'political_state',
            'school_language',
            'qualification'
        ]
        
    def fit_transform(self, df):
        # Convert boolean columns to int
        df = df.copy()
        for col in self.binary_features:
            if col in df.columns and df[col].dtype == bool:
                df[col] = df[col].astype(int)
        
        # First, verify which columns actually exist in the dataframe
        existing_numeric = [col for col in self.numeric_features if col in df.columns]
        existing_binary = [col for col in self.binary_features if col in df.columns]
        existing_categorical = [col for col in self.categorical_features if col in df.columns]
        
        # Create preprocessing pipelines
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        binary_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('scaler', StandardScaler())  # Scale binary features as well
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        
        # Create transformers list with only existing columns
        transformers = []
        if existing_numeric:
            transformers.append(('num', numeric_transformer, existing_numeric))
        if existing_binary:
            transformers.append(('bin', binary_transformer, existing_binary))
        if existing_categorical:
            transformers.append(('cat', categorical_transformer, existing_categorical))
        
        # Combine transformers
        self.column_transformer = ColumnTransformer(
            transformers=transformers,
            remainder='drop',
            verbose_feature_names_out=False
        )
        
        # Transform features
        X = df.drop('dropout_status', axis=1) if 'dropout_status' in df.columns else df
        y = df['dropout_status'] if 'dropout_status' in df.columns else None
        
        # Print available columns for debugging
        print("Available columns in dataframe:", df.columns.tolist())
        print("Using numeric features:", existing_numeric)
        print("Using binary features:", existing_binary)
        print("Using categorical features:", existing_categorical)
        
        X_transformed = self.column_transformer.fit_transform(X)
        
        # Store feature names for later use
        feature_names = []
        
        # Add numeric feature names
        if existing_numeric:
            feature_names.extend(existing_numeric)
            
        # Add binary feature names
        if existing_binary:
            feature_names.extend(existing_binary)
            
        # Add categorical feature names
        if existing_categorical:
            for feat, cats in zip(existing_categorical, 
                                self.column_transformer.named_transformers_['cat'].named_steps['onehot'].categories_):
                for cat in cats[1:]:  # Skip first category as it's dropped
                    feature_names.append(f"{feat}_{cat}")
        
        self.feature_names = feature_names
        
        if y is not None:
            return X_transformed, y
        return X_transformed
    
    def transform(self, df):
        # Convert boolean columns to int for prediction
        df = df.copy()
        for col in self.binary_features:
            if col in df.columns and df[col].dtype == bool:
                df[col] = df[col].astype(int)
        return self.column_transformer.transform(df)
    
    def get_feature_names(self):
        return self.feature_names