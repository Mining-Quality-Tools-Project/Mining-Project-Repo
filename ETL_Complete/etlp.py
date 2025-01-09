import pandas as pd
import psycopg2
from io import StringIO

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
    """Read raw CSV files into DataFrames."""
    dataframes = {}
    for table, path in file_paths.items():
        try:
            df = pd.read_csv(path)
            dataframes[table] = df
            print(f"Loaded {table} with {len(df)} rows.")
        except Exception as e:
            print(f"Error loading {table} from {path}: {e}")
    return dataframes

def transform_data(dataframes):
    """Transform data for consistency with the database schema."""
    for table, df in dataframes.items():
        key_column = f"{table.split('_')[1]}_key"  # Example: dim_student -> student_key
        if key_column not in df.columns:
            print(f"Warning: {key_column} not found in {table}, skipping transformation.")
        else:
            # Ensure primary keys are unique and no nulls
            df[key_column] = df[key_column].fillna(method='ffill').astype(str).str.strip()
            dataframes[table] = df
    return dataframes

def load_data_with_copy(dataframes, db_settings):
    """Load transformed data into the database using COPY."""
    conn = psycopg2.connect(**db_settings)
    cur = conn.cursor()

    # Define the load order based on foreign key dependencies
    load_order = [
        'dim_student', 'dim_time', 'dim_teacher', 'dim_socioeconomic',
        'dim_school', 'dim_location', 'dim_government_aid',
        'dim_community_support', 'fact_dropouts'
    ]

    for table in load_order:
        try:
            if table in dataframes:
                print(f"Loading data into {table}...")

                # Prepare the data as a CSV string
                output = StringIO()
                dataframes[table].to_csv(output, sep='\t', header=False, index=False)
                output.seek(0)

                # Use the COPY command for efficient bulk loading
                cur.copy_from(output, table, sep='\t', null='')
                conn.commit()

                print(f"Successfully loaded {table} into the database.")
            else:
                print(f"Skipping {table}: No data found.")
        except psycopg2.Error as e:
            conn.rollback()
            print(f"Error loading {table}: {e}")
        except Exception as e:
            print(f"Unexpected error while loading {table}: {e}")
        finally:
            output.close()

    cur.close()
    conn.close()

def main():
    print("Starting ETL process...")

    print("Step 1: Reading CSV files...")
    dataframes = read_csv_files(RAW_FILES)

    print("Step 2: Transforming data...")
    transformed_data = transform_data(dataframes)

    print("Step 3: Loading data into the database...")
    load_data_with_copy(transformed_data, DB_SETTINGS)

    print("ETL process completed successfully!")

if __name__ == "__main__":
    main()
