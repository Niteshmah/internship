import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import json
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from typing import List, Dict, Tuple

class Student:
    def __init__(self, student_id: str, name: str, skills: List[str], 
                 academic_background: str, gpa: float, preferred_locations: List[str],
                 preferred_industries: List[str], experience_level: str):
        self.student_id = student_id
        self.name = name
        self.skills = skills
        self.academic_background = academic_background
        self.gpa = gpa
        self.preferred_locations = preferred_locations
        self.preferred_industries = preferred_industries
        self.experience_level = experience_level
    
    def to_dict(self):
        return {
            'student_id': self.student_id,
            'name': self.name,
            'skills': self.skills,
            'academic_background': self.academic_background,
            'gpa': self.gpa,
            'preferred_locations': self.preferred_locations,
            'preferred_industries': self.preferred_industries,
            'experience_level': self.experience_level
        }

class Internship:
    def __init__(self, internship_id: str, company_name: str, role_title: str,
                 required_skills: List[str], location: str, industry: str,
                 duration_months: int, min_gpa: float, experience_required: str,
                 description: str):
        self.internship_id = internship_id
        self.company_name = company_name
        self.role_title = role_title
        self.required_skills = required_skills
        self.location = location
        self.industry = industry
        self.duration_months = duration_months
        self.min_gpa = min_gpa
        self.experience_required = experience_required
        self.description = description
    
    def to_dict(self):
        return {
            'internship_id': self.internship_id,
            'company_name': self.company_name,
            'role_title': self.role_title,
            'required_skills': self.required_skills,
            'location': self.location,
            'industry': self.industry,
            'duration_months': self.duration_months,
            'min_gpa': self.min_gpa,
            'experience_required': self.experience_required,
            'description': self.description
        }

class AIMatchmakingSystem:
    def __init__(self):
        self.students = []
        self.internships = []
        self.interaction_history = []
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.scaler = StandardScaler()
        self.setup_database()
    
    def setup_database(self):
        """Initialize SQLite database for storing data"""
        self.conn = sqlite3.connect('internship_matching.db', check_same_thread=False)
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                name TEXT,
                skills TEXT,
                academic_background TEXT,
                gpa REAL,
                preferred_locations TEXT,
                preferred_industries TEXT,
                experience_level TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS internships (
                internship_id TEXT PRIMARY KEY,
                company_name TEXT,
                role_title TEXT,
                required_skills TEXT,
                location TEXT,
                industry TEXT,
                duration_months INTEGER,
                min_gpa REAL,
                experience_required TEXT,
                description TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                student_id TEXT,
                internship_id TEXT,
                action TEXT,
                rating INTEGER,
                timestamp TEXT
            )
        ''')
        
        self.conn.commit()
    
    def add_student(self, student: Student):
        """Add a student to the system"""
        self.students.append(student)
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO students VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            student.student_id, student.name, json.dumps(student.skills),
            student.academic_background, student.gpa,
            json.dumps(student.preferred_locations),
            json.dumps(student.preferred_industries),
            student.experience_level
        ))
        self.conn.commit()
    
    def add_internship(self, internship: Internship):
        """Add an internship to the system"""
        self.internships.append(internship)
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO internships VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            internship.internship_id, internship.company_name, internship.role_title,
            json.dumps(internship.required_skills), internship.location,
            internship.industry, internship.duration_months, internship.min_gpa,
            internship.experience_required, internship.description
        ))
        self.conn.commit()
    
    def get_all_students(self):
        """Get all students from database"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM students')
        students_data = cursor.fetchall()
        
        students = []
        for row in students_data:
            student_dict = {
                'student_id': row[0],
                'name': row[1],
                'skills': json.loads(row[2]),
                'academic_background': row[3],
                'gpa': row[4],
                'preferred_locations': json.loads(row[5]),
                'preferred_industries': json.loads(row[6]),
                'experience_level': row[7]
            }
            students.append(student_dict)
        return students
    
    def get_all_internships(self):
        """Get all internships from database"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM internships')
        internships_data = cursor.fetchall()
        
        internships = []
        for row in internships_data:
            internship_dict = {
                'internship_id': row[0],
                'company_name': row[1],
                'role_title': row[2],
                'required_skills': json.loads(row[3]),
                'location': row[4],
                'industry': row[5],
                'duration_months': row[6],
                'min_gpa': row[7],
                'experience_required': row[8],
                'description': row[9]
            }
            internships.append(internship_dict)
        return internships
    
    def calculate_skill_similarity(self, student_skills: List[str], 
                                 internship_skills: List[str]) -> float:
        """Calculate skill-based similarity using Jaccard similarity"""
        set_student = set([skill.lower() for skill in student_skills])
        set_internship = set([skill.lower() for skill in internship_skills])
        
        intersection = len(set_student.intersection(set_internship))
        union = len(set_student.union(set_internship))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_content_based_similarity(self, student_skills, student_gpa, 
                                         student_locations, student_industries, 
                                         student_experience, internship) -> float:
        """Calculate content-based similarity score"""
        skill_sim = self.calculate_skill_similarity(student_skills, internship['required_skills'])
        location_match = 1.0 if internship['location'] in student_locations else 0.0
        industry_match = 1.0 if internship['industry'] in student_industries else 0.0
        gpa_match = 1.0 if student_gpa >= internship['min_gpa'] else 0.0
        
        exp_levels = {'beginner': 1, 'intermediate': 2, 'advanced': 3}
        student_exp = exp_levels.get(student_experience.lower(), 1)
        required_exp = exp_levels.get(internship['experience_required'].lower(), 1)
        exp_match = 1.0 if student_exp >= required_exp else 0.5
        
        total_score = (skill_sim * 0.4 + location_match * 0.2 + 
                      industry_match * 0.2 + gpa_match * 0.1 + exp_match * 0.1)
        
        return total_score
    
    def hybrid_recommendations(self, student_id: str, top_n: int = 10) -> List[Dict]:
        """Generate hybrid recommendations"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM students WHERE student_id = ?', (student_id,))
        student_row = cursor.fetchone()
        
        if not student_row:
            return []
        
        student_data = {
            'skills': json.loads(student_row[2]),
            'gpa': student_row[4],
            'preferred_locations': json.loads(student_row[5]),
            'preferred_industries': json.loads(student_row[6]),
            'experience_level': student_row[7]
        }
        
        internships = self.get_all_internships()
        
        recommendations = []
        for internship in internships:
            score = self.calculate_content_based_similarity(
                student_data['skills'], student_data['gpa'],
                student_data['preferred_locations'], student_data['preferred_industries'],
                student_data['experience_level'], internship
            )
            
            rec_data = internship.copy()
            rec_data['match_score'] = score
            recommendations.append(rec_data)
        
        recommendations.sort(key=lambda x: x['match_score'], reverse=True)
        return recommendations[:top_n]
    
    def record_interaction(self, student_id: str, internship_id: str, 
                          action: str, rating: int = None):
        """Record student interaction for learning"""
        cursor = self.conn.cursor()
        timestamp = datetime.now().isoformat()
        cursor.execute('''
            INSERT INTO interactions VALUES (?, ?, ?, ?, ?)
        ''', (student_id, internship_id, action, rating, timestamp))
        self.conn.commit()
    
    def get_analytics(self) -> Dict:
        """Get system analytics"""
        cursor = self.conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM students')
        student_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM internships')
        internship_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM interactions')
        interaction_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT required_skills FROM internships')
        all_skills = []
        for row in cursor.fetchall():
            skills = json.loads(row[0])
            all_skills.extend(skills)
        
        skill_counts = {}
        for skill in all_skills:
            skill_counts[skill] = skill_counts.get(skill, 0) + 1
        
        top_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'total_students': student_count,
            'total_internships': internship_count,
            'total_interactions': interaction_count,
            'top_skills_in_demand': top_skills
        }

# Flask app setup
app = Flask(__name__)
matching_system = AIMatchmakingSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/students', methods=['POST'])
def add_student():
    data = request.json
    student = Student(
        student_id=data['student_id'],
        name=data['name'],
        skills=data['skills'],
        academic_background=data['academic_background'],
        gpa=data['gpa'],
        preferred_locations=data['preferred_locations'],
        preferred_industries=data['preferred_industries'],
        experience_level=data['experience_level']
    )
    matching_system.add_student(student)
    return jsonify({'message': 'Student added successfully'})

@app.route('/api/students', methods=['GET'])
def get_students():
    students = matching_system.get_all_students()
    return jsonify({'students': students})

@app.route('/api/internships', methods=['POST'])
def add_internship():
    data = request.json
    internship = Internship(
        internship_id=data['internship_id'],
        company_name=data['company_name'],
        role_title=data['role_title'],
        required_skills=data['required_skills'],
        location=data['location'],
        industry=data['industry'],
        duration_months=data['duration_months'],
        min_gpa=data['min_gpa'],
        experience_required=data['experience_required'],
        description=data['description']
    )
    matching_system.add_internship(internship)
    return jsonify({'message': 'Internship added successfully'})

@app.route('/api/internships', methods=['GET'])
def get_internships():
    internships = matching_system.get_all_internships()
    return jsonify({'internships': internships})

@app.route('/api/recommendations/<student_id>', methods=['GET'])
def get_recommendations(student_id):
    top_n = request.args.get('top_n', 10, type=int)
    recommendations = matching_system.hybrid_recommendations(student_id, top_n)
    return jsonify({'recommendations': recommendations})

@app.route('/api/interact', methods=['POST'])
def record_interaction():
    data = request.json
    matching_system.record_interaction(
        data['student_id'],
        data['internship_id'],
        data['action'],
        data.get('rating')
    )
    return jsonify({'message': 'Interaction recorded'})

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    analytics = matching_system.get_analytics()
    return jsonify(analytics)

@app.route('/debug')
def debug():
    import os
    template_dir = app.template_folder
    return f"Template directory: {template_dir}<br>Files: {os.listdir(template_dir) if os.path.exists(template_dir) else 'Directory not found'}"


if __name__ == '__main__':
    # Add sample data for testing
    sample_students = [
        Student('s1', 'Alice Johnson', ['Python', 'Machine Learning', 'Data Analysis'],
                'Computer Science', 3.8, ['New York', 'San Francisco'], ['Technology'], 'intermediate'),
        Student('s2', 'Bob Smith', ['Java', 'Spring Boot', 'SQL'],
                'Software Engineering', 3.6, ['Boston', 'Austin'], ['Technology', 'Finance'], 'beginner'),
        Student('s3', 'Carol Davis', ['Marketing', 'Social Media', 'Analytics'],
                'Business', 3.7, ['Chicago', 'Denver'], ['Marketing', 'Consulting'], 'intermediate')
    ]
    
    sample_internships = [
        Internship('i1', 'TechCorp', 'Data Science Intern', ['Python', 'Machine Learning'],
                  'San Francisco', 'Technology', 3, 3.5, 'intermediate',
                  'Work on ML models for product recommendations'),
        Internship('i2', 'FinanceInc', 'Software Developer Intern', ['Java', 'SQL'],
                  'New York', 'Finance', 4, 3.4, 'beginner',
                  'Develop backend services for financial applications'),
        Internship('i3', 'MarketingPro', 'Digital Marketing Intern', ['Marketing', 'Analytics'],
                  'Chicago', 'Marketing', 2, 3.2, 'beginner',
                  'Create and analyze marketing campaigns')
    ]
    
    for student in sample_students:
        matching_system.add_student(student)
    
    for internship in sample_internships:
        matching_system.add_internship(internship)
    
    app.run(debug=True, port=5000)




    
