import uuid
import random
import csv

# Constants
REGIONS = {
    "Adamawa": ["Ngaoundéré", "Tignère", "Meiganga"],
    "Centre": ["Yaoundé", "Mfou", "Obala"],
    "East": ["Bertoua", "Abong-Mbang", "Batouri"],
    "Far North": ["Maroua", "Kousseri", "Mora"],
    "Littoral": ["Douala", "Nkongsamba", "Edea"],
    "North": ["Garoua", "Guider", "Pitoa"],
    "Northwest": ["Bamenda", "Ndop", "Kumbo"],
    "South": ["Ebolowa", "Ambam", "Kribi"],
    "Southwest": ["Buea", "Kumba", "Limbe"],
    "West": ["Bafoussam", "Dschang", "Foumban"],
}

RELIGIONS = ["Christianity", "Islam", "Traditional Beliefs"]
LANGUAGES = ["French", "English", "Fulfulde", "Pidgin", "Local Dialects"]
POLITICAL_STATES = ["Stable", "Conflict"]
MARITAL_STATUS = ["Single", "Married", "Divorced", "Widowed"]
ACADEMIC_PERFORMANCE = ["Excellent", "Good", "Average", "Poor"]
QUALIFICATIONS = ["Bachelor's Degree", "Master's Degree", "PhD", "Diploma"]
SUBJECTS = ["Mathematics", "English", "Science", "History", "Geography"]

# Helper functions
def generate_uuid():
    return str(uuid.uuid4())

def random_choice_with_weights(choices, weights=None):
    return random.choices(choices, weights=weights, k=1)[0]

# Generate data for each dimension
def generate_dim_student(num_records):
    students = []
    for _ in range(num_records):
        students.append({
            "student_key": generate_uuid(),
            "age": random.randint(10, 22),
            "gender": random_choice_with_weights(["Male", "Female"], weights=[0.5, 0.5]),
            "special_needs": random_choice_with_weights([True, False], weights=[0.1, 0.9]),
            "num_children": random.randint(0, 3),
            "previous_academic_performance": random_choice_with_weights(ACADEMIC_PERFORMANCE, weights=[0.2, 0.3, 0.3, 0.2]),
            "attendance_rate": round(random.uniform(50, 100), 2),
            "language_spoken": random_choice_with_weights(LANGUAGES),
            "marital_status": random_choice_with_weights(MARITAL_STATUS, weights=[0.8, 0.15, 0.03, 0.02]),
        })
    return students

def generate_dim_time(num_records):
    times = []
    for _ in range(num_records):
        times.append({
            "time_key": generate_uuid(),
            "academic_year": random.choice(["2023/2024", "2024/2025"]),
            "semester": random.choice(["First", "Second"]),
            "month": random.choice(["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]),
            "holiday": random.choice([True, False]),
        })
    return times

def generate_dim_teacher(num_records):
    teachers = []
    for _ in range(num_records):
        teachers.append({
            "teacher_key": generate_uuid(),
            "qualification": random.choice(QUALIFICATIONS),
            "subject_expertise": random.choice(SUBJECTS),
            "years_of_experience": random.randint(1, 30),
        })
    return teachers

def generate_dim_socioeconomic(num_records):
    socioeconomic = []
    for _ in range(num_records):
        socioeconomic.append({
            "socioeconomic_key": generate_uuid(),
            "household_income": random.choice(["Low", "Medium", "High"]),
            "parent_education": random.choice(["None", "Primary", "Secondary", "Tertiary"]),
            "electricity_access": random.choice([True, False]),
            "internet_access": random.choice([True, False]),
            "household_size": random.randint(1, 12),
            "religion": random.choice(RELIGIONS),
            "political_state": random_choice_with_weights(POLITICAL_STATES, weights=[0.7, 0.3]),
        })
    return socioeconomic

def generate_dim_school(num_records):
    schools = []
    for _ in range(num_records):
        num_teachers = random.randint(5, 50)  # Number of teachers in the school
        student_teacher_ratio = round(random.uniform(10, 50), 2)  # Student-teacher ratio
        estimated_students = int(num_teachers * student_teacher_ratio)  # Approximation of total students
        has_computer_lab = random.choice([True, False])
        has_library = random.choice([True, False])
        school_language = random.choice(LANGUAGES[:2])  # French or English for schools in Cameroon
        
        schools.append({
            "school_key": generate_uuid(),
            "school_name": f"School_{random.randint(1, 1000)}",
            "num_teachers": num_teachers,
            "student_teacher_ratio": student_teacher_ratio,
            "has_computer_lab": has_computer_lab,
            "has_library": has_library,
            "school_language": school_language,
        })
    return schools


def generate_dim_location(num_records):
    locations = []
    for _ in range(num_records):
        region = random.choice(list(REGIONS.keys()))
        locations.append({
            "location_key": generate_uuid(),
            "region": region,
            "village": random.choice(REGIONS[region]),
            "distance_to_school": random.randint(1, 30),
        })
    return locations

def generate_dim_government_aid(num_records):
    aids = []
    for _ in range(num_records):
        aids.append({
            "aid_key": generate_uuid(),
            "program_name": f"Aid_Program_{random.randint(1, 100)}",
            "region": random.choice(list(REGIONS.keys())),
            "aid_type": random.choice(["Scholarship", "Infrastructure", "Teacher Training"]),
            "aid_coverage": random.choice([True, False]),
        })
    return aids

def generate_dim_community_support(num_records):
    supports = []
    for _ in range(num_records):
        supports.append({
            "community_key": generate_uuid(),
            "parental_involvement": random.choice([True, False]),
            "community_support": random.choice([True, False]),
        })
    return supports

def generate_fact_dropouts(students, times, teachers, socioeconomic, schools, locations, aids, supports):
    facts = []
    for student in students:
        facts.append({
            "fact_key": generate_uuid(),
            "student_key": student["student_key"],  # Match key from dim_student
            "time_key": random.choice(times)["time_key"],  # Match key from dim_time
            "teacher_key": random.choice(teachers)["teacher_key"],  # Match key from dim_teacher
            "socioeconomic_key": random.choice(socioeconomic)["socioeconomic_key"],  # Match key from dim_socioeconomic
            "school_key": random.choice(schools)["school_key"],  # Match key from dim_school
            "location_key": random.choice(locations)["location_key"],  # Match key from dim_location
            "aid_key": random.choice(aids)["aid_key"],  # Match key from dim_government_aid
            "community_key": random.choice(supports)["community_key"],  # Match key from dim_community_support
            "dropout_status": random_choice_with_weights([True, False], weights=[0.3, 0.7]),
            "reason_for_dropout": random.choice([
                "Financial difficulties",
                "Distance to school",
                "Language barrier",
                "Marriage/Children",
                "Special needs",
                "Political instability"
            ]),
        })
    return facts

# Save to CSV
def save_to_csv(data, filename):
    keys = data[0].keys()
    with open(filename, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)

# Main function
def main():
    num_records = 1000

    students = generate_dim_student(num_records)
    times = generate_dim_time(num_records)
    teachers = generate_dim_teacher(num_records)
    socioeconomic = generate_dim_socioeconomic(num_records)
    schools = generate_dim_school(num_records)
    locations = generate_dim_location(num_records)
    aids = generate_dim_government_aid(num_records)
    supports = generate_dim_community_support(num_records)
    facts = generate_fact_dropouts(students, times, teachers, socioeconomic, schools, locations, aids, supports)

    save_to_csv(students, "dim_student.csv")
    save_to_csv(times, "dim_time.csv")
    save_to_csv(teachers, "dim_teacher.csv")
    save_to_csv(socioeconomic, "dim_socioeconomic.csv")
    save_to_csv(schools, "dim_school.csv")
    save_to_csv(locations, "dim_location.csv")
    save_to_csv(aids, "dim_government_aid.csv")
    save_to_csv(supports, "dim_community_support.csv")
    save_to_csv(facts, "fact_dropouts.csv")

if __name__ == "__main__":
    main()
