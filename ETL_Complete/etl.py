import pandas as pd
from sqlalchemy import create_engine
import uuid

# Database settings
DB_SETTINGS = {
    "dbname": "Dropout",
    "user": "postgres",
    "password": "root",
    "host": "localhost",
    "port": 5432
}

# Raw data file paths
RAW_FILES = {
    "dim_student": "data/dim_student.csv",
    "dim_time": "data/dim_time.csv",
    "dim_teacher": "data/dim_teacher.csv",
    "dim_socioeconomic": "data/dim_socioeconomic.csv",
    "dim_school": "data/dim_school.csv",
    "dim_location": "data/dim_location.csv",
    "dim_government_aid": "data/dim_government_aid.csv",
    "dim_community_support": "data/dim_community_support.csv",
    "fact_dropouts": "data/fact_dropouts.csv"
}

def read_csv_files(file_paths):
    """Read raw CSV files into dataframes."""
    dataframes = {}
    for table, path in file_paths.items():
        try:
            df = pd.read_csv(path)
            dataframes[table] = df
            print(f"Loaded {table} with {len(df)} rows.")
        except Exception as e:
            print(f"Error loading {table} from {path}: {e}")
    return dataframes

def generate_uuids_if_missing(df, key_column):
    """Generate UUIDs for the primary key column if not already present."""
    if key_column not in df.columns or df[key_column].isnull().all():
        df[key_column] = [str(uuid.uuid4()) for _ in range(len(df))]
    return df

def transform_data(dataframes):
    """Transform data to ensure UUIDs and consistency."""
    for table, df in dataframes.items():
        key_column = f"{table.split('_')[1]}_key"  # e.g., dim_student -> student_key
        if key_column in df.columns or "fact_" in table:  # Ensure foreign key relations
            dataframes[table] = generate_uuids_if_missing(df, key_column)
        else:
            print(f"Skipping UUID generation for {table}, no key column found.")
    return dataframes

def load_data(dataframes, db_settings):
    """Load transformed data into the database."""
    engine = create_engine(
        f"postgresql://{db_settings['user']}:{db_settings['password']}@"
        f"{db_settings['host']}:{db_settings['port']}/{db_settings['dbname']}"
    )

    load_order = [
        'dim_student', 'dim_time', 'dim_teacher', 'dim_socioeconomic',
        'dim_school', 'dim_location', 'dim_government_aid',
        'dim_community_support', 'fact_dropouts'
    ]

    for table in load_order:
        try:
            if table in dataframes:
                print(f"Loading data into {table}...")
                dataframes[table].to_sql(table, engine, if_exists='append', index=False)
                print(f"Successfully loaded {table} into the database.")
            else:
                print(f"Skipping {table}: No data found.")
        except Exception as e:
            print(f"Error loading {table}: {e}")

def main():
    print("Reading CSV files...")
    dataframes = read_csv_files(RAW_FILES)

    print("Transforming data...")
    transformed_data = transform_data(dataframes)

    print("Loading data into the database...")
    load_data(transformed_data, DB_SETTINGS)

    print("ETL process completed successfully!")

if __name__ == "__main__":
    main()
