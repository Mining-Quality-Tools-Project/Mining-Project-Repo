import pandas as pd
import os

def load_data():
    """Load data from the synthetic CSV file"""
    csv_path = 'data/synthetic_dropout_data.csv'
    
    # Check if file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found at {csv_path}. Please run generate_data.py first.")
    
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Convert boolean columns from string to bool
    bool_columns = [
        'special_needs', 'electricity_access', 'internet_access',
        'has_computer_lab', 'has_library', 'aid_coverage',
        'parental_involvement', 'community_support', 'dropout_status'
    ]
    for col in bool_columns:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    
    print(f"Loaded {len(df)} records from {csv_path}")
    return df