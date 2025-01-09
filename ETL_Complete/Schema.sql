-- Table: dim_student
CREATE TABLE dim_student (
    student_key UUID PRIMARY KEY, -- Changed from student_id
    age INT NOT NULL,
    gender TEXT NOT NULL,
    special_needs BOOLEAN NOT NULL,
    num_children INT NOT NULL,
    previous_academic_performance TEXT NOT NULL,
    attendance_rate NUMERIC(5, 2) NOT NULL,
    language_spoken TEXT NOT NULL,
    marital_status TEXT NOT NULL
);

-- Table: dim_time
CREATE TABLE dim_time (
    time_key UUID PRIMARY KEY, -- Changed from student_id
    academic_year TEXT NOT NULL,
    semester TEXT NOT NULL,
    month TEXT NOT NULL,
    holiday BOOLEAN NOT NULL
);

-- Table: dim_teacher
CREATE TABLE dim_teacher (
    teacher_key UUID PRIMARY KEY, -- Changed from teacher_id
    qualification TEXT NOT NULL,
    subject_expertise TEXT NOT NULL,
    years_of_experience INT NOT NULL
);

-- Table: dim_socioeconomic
CREATE TABLE dim_socioeconomic (
    socioeconomic_key UUID PRIMARY KEY, -- Changed from student_id
    household_income TEXT NOT NULL,
    parent_education TEXT,
    electricity_access BOOLEAN NOT NULL,
    internet_access BOOLEAN NOT NULL,
    household_size INT NOT NULL,
    religion TEXT NOT NULL,
    political_state TEXT NOT NULL
);

-- Table: dim_school
CREATE TABLE dim_school (
    school_key UUID PRIMARY KEY, -- Changed from school_name
    school_name TEXT NOT NULL,
    num_teachers INT NOT NULL,
    student_teacher_ratio NUMERIC(5, 2) NOT NULL,
    has_computer_lab BOOLEAN NOT NULL,
    has_library BOOLEAN NOT NULL,
    school_language TEXT NOT NULL
);

-- Table: dim_location
CREATE TABLE dim_location (
    location_key UUID PRIMARY KEY, -- Changed from student_id
    region TEXT NOT NULL,
    village TEXT NOT NULL,
    distance_to_school INT NOT NULL
);

-- Table: dim_government_aid
CREATE TABLE dim_government_aid (
    aid_key UUID PRIMARY KEY, -- Changed from program_name
    program_name TEXT NOT NULL,
    region TEXT NOT NULL,
    aid_type TEXT NOT NULL,
    aid_coverage BOOLEAN NOT NULL
);

-- Table: dim_community_support
CREATE TABLE dim_community_support (
    community_key UUID PRIMARY KEY, -- Changed from student_id
    parental_involvement BOOLEAN NOT NULL,
    community_support BOOLEAN NOT NULL
);

-- Table: fact_dropouts
CREATE TABLE fact_dropouts (
    fact_key UUID PRIMARY KEY, -- New primary key for fact table
    student_key UUID NOT NULL, -- Foreign key from dim_student
    time_key UUID NOT NULL, -- Foreign key from dim_time
    teacher_key UUID NOT NULL, -- Foreign key from dim_teacher
    socioeconomic_key UUID NOT NULL, -- Foreign key from dim_socioeconomic
    school_key UUID NOT NULL, -- Foreign key from dim_school
    location_key UUID NOT NULL, -- Foreign key from dim_location
    aid_key UUID NOT NULL, -- Foreign key from dim_government_aid
    community_key UUID NOT NULL, -- Foreign key from dim_community_support
    dropout_status BOOLEAN NOT NULL,
    reason_for_dropout TEXT,
    CONSTRAINT fk_fact_dim_student FOREIGN KEY (student_key) REFERENCES dim_student (student_key),
    CONSTRAINT fk_fact_dim_time FOREIGN KEY (time_key) REFERENCES dim_time (time_key),
    CONSTRAINT fk_fact_dim_teacher FOREIGN KEY (teacher_key) REFERENCES dim_teacher (teacher_key),
    CONSTRAINT fk_fact_dim_socioeconomic FOREIGN KEY (socioeconomic_key) REFERENCES dim_socioeconomic (socioeconomic_key),
    CONSTRAINT fk_fact_dim_school FOREIGN KEY (school_key) REFERENCES dim_school (school_key),
    CONSTRAINT fk_fact_dim_location FOREIGN KEY (location_key) REFERENCES dim_location (location_key),
    CONSTRAINT fk_fact_dim_government_aid FOREIGN KEY (aid_key) REFERENCES dim_government_aid (aid_key),
    CONSTRAINT fk_fact_dim_community_support FOREIGN KEY (community_key) REFERENCES dim_community_support (community_key)
);
