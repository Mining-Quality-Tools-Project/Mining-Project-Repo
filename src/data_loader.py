import pandas as pd
from sqlalchemy import create_engine
from config import DB_CONFIG

def create_connection():
    conn_str = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
    return create_engine(conn_str)

def load_data():
    engine = create_connection()
    
    query = """
    SELECT 
        f.dropout_status,
        s.age, s.gender, s.special_needs, s.num_children, 
        s.previous_academic_performance, s.attendance_rate,
        s.language_spoken, s.marital_status,
        se.household_income, se.parent_education, se.electricity_access,
        se.internet_access, se.household_size, se.political_state,
        sc.num_teachers, sc.student_teacher_ratio, sc.has_computer_lab,
        sc.has_library, sc.school_language,
        l.distance_to_school,
        t.qualification, t.years_of_experience,
        ga.aid_coverage,
        cs.parental_involvement, cs.community_support
    FROM fact_dropouts f
    JOIN dim_student s ON f.student_key = s.student_key
    JOIN dim_socioeconomic se ON f.socioeconomic_key = se.socioeconomic_key
    JOIN dim_school sc ON f.school_key = sc.school_key
    JOIN dim_location l ON f.location_key = l.location_key
    JOIN dim_teacher t ON f.teacher_key = t.teacher_key
    JOIN dim_government_aid ga ON f.aid_key = ga.aid_key
    JOIN dim_community_support cs ON f.community_key = cs.community_key
    """
    
    df = pd.read_sql(query, engine)
    return df