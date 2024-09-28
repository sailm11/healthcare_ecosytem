from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, login_user, logout_user, login_required, UserMixin, current_user
from pymongo import MongoClient
from mongoengine import Document, StringField, ListField, ReferenceField, DateTimeField, EmbeddedDocumentField, EmbeddedDocument, connect, FileField, FloatField
from bson.objectid import ObjectId
import datetime
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import os
import pickle
import pandas as pd
from functools import lru_cache
from scraping import scrape_doctors
import time
from bs4 import BeautifulSoup


app = Flask(__name__, template_folder='templates', static_folder='staticFolder')
app.secret_key = 'KUNDAN1234'

# Setup MongoDB Connection
client = MongoClient("mongodb+srv://kundankumawat003:kundan1234@cluster0.7hqin.mongodb.net/")
db = client['medical-data']
users_collection = db['users']
medical_collection = db['medical_data']

connect('medical-data', host="mongodb+srv://kundankumawat003:kundan1234@cluster0.7hqin.mongodb.net/medical-data")

# Set up bcrypt for password hashing
bcrypt = Bcrypt(app)

# Set up Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# MongoEngine User model
class UserDocument(Document):
    username = StringField(required=True, unique=True)
    email = StringField(required=True, unique=True)
    password = StringField(required=True)
    profile_picture = FileField()
    friends = ListField(ReferenceField('self'))
    groups_joined = ListField(ReferenceField('CommunityGroup'))
    registration_date = DateTimeField(default=datetime.datetime.utcnow)

#user_responses
# response_arr={
#     'inputs':'',
#     'outputs':''
# }

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, user_id, username):
        self.id = user_id
        self.username = username

    @staticmethod
    def get(user_id):
        user_data = users_collection.find_one({"_id": ObjectId(user_id)})
        if user_data:
            return User(user_id=user_data['_id'], username=user_data['username'])
        return None

# Medical Data Collection Schema
class MedicalData(Document):
    user = ReferenceField(UserDocument, required=True)
    first_name = StringField(required=True)
    last_name = StringField(required=True)
    age = StringField()
    profession = StringField()
    working_hours = StringField()
    gender = StringField()
    height = StringField()
    weigth = StringField()
    previous_surgery_name  = StringField()
    previous_surgery_date = DateTimeField()
    complications_during_surgery = StringField()
    anestesia_history = StringField()
    chronic_conditions= StringField()
    current_medications= StringField()
    known_allergies = StringField()
    disease = StringField()
    drug_name = StringField()
    medication_duration = StringField()
    addication_name = StringField()
    addication_frequency = StringField()
    addication_duration = StringField()
    heart_rate = FloatField()  # Beats per minute (BPM)
    blood_pressure = StringField()  # E.g., "120/80 mmHg"
    sugar_level = FloatField()  # Blood sugar level
    diabetes_status = StringField(choices=["No", "Pre-diabetes", "Diabetes"])
    smartwatch_data = ListField(StringField())  # Data synced from smartwatch
    medical_appointments = ListField(StringField())  # List of appointment dates
    medical_reports = ListField(FileField())  # Medical reports (scanned files or PDFs)
    diagnosis = StringField()  # Diagnosis details
    medicines_prescribed = ListField(StringField())  # List of medicines
    created_at = DateTimeField(default=datetime.datetime.utcnow)

# Community Group Schema
class CommunityGroup(Document):
    group_name = StringField(required=True, unique=True)
    description = StringField()
    created_by = ReferenceField(UserDocument)
    group_members = ListField(ReferenceField(UserDocument))
    profile_picture = FileField()
    created_at = DateTimeField(default=datetime.datetime.utcnow)

# Post Schema
class Post(Document):
    content = StringField(required=True)
    posted_by = ReferenceField(UserDocument, required=True)
    posted_on_group = ReferenceField(CommunityGroup, required=True)
    created_at = DateTimeField(default=datetime.datetime.utcnow)

class ChatBot(Document):
    owner = ReferenceField(UserDocument)
    user_input = StringField()
    output = StringField()
    timestamp = DateTimeField(default=datetime.datetime.utcnow)

class Appointment(Document):
    doctor_name = StringField(required=True)
    specialization = StringField(required=True)
    location = StringField(required=True)
    consultation_fee = StringField(required=True)
    clinic_name = StringField(required=True)
    experience = StringField(required=True)
    user_id = ReferenceField(User)


@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

# Registration route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        # Check if username exists
        if users_collection.find_one({'username': username}):
            flash('Username already exists. Please choose another one.', 'danger')
            return redirect(url_for('register'))

        # Insert user into MongoDB
        users_collection.insert_one({'username': username, 'email': email, 'password': hashed_password})
        flash('Registration successful. Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Find user in MongoDB
        user_data = users_collection.find_one({'username': username})
        if user_data and bcrypt.check_password_hash(user_data['password'], password):
            user = User(user_id=str(user_data['_id']), username=user_data['username'])
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('index'))

        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/book_appointment', methods=['POST'])
def book_appointment():
    if request.method == 'POST':
        appointment = Appointment(
            doctor_name=request.form['doctor_name'],
            specialization=request.form['specialization'],
            location=request.form['location'],
            consultation_fee=request.form['consultation_fee'],
            clinic_name=request.form['clinic_name'],
            experience=request.form['experience'],
            user_id=current_user.id  # Using current_user ID
        )
        appointment.save()
        return redirect(url_for('reminder'))

# Home route
@app.route('/')
def index():
    return render_template('index.html', username=current_user.username)

@app.route('/upload', methods=["GET", "POST"])
def upload():
    
    return render_template('report.html')

# Logout route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/profile', methods=["GET", "POST"])
@login_required
def profile():

    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        age = request.form['age']
        profession = request.form['profession']
        working_hours = request.form['working_hours']
        gender = request.form['gender']
        height = request.form['height']
        weigth = request.form['weigth']
        previous_surgery_name = request.form['previous_surgery_name']
        previous_surgery_date = request.form['previous_surgery_date']
        complications_during_surgery = request.form['complications_during_surgery']
        anestesia_history = request.form['anestesia_history']
        chronic_conditions = request.form['chronic_conditions']
        current_medications = request.form['current_medications']
        known_allergies = request.form['known_allergies']
        disease = request.form['disease']
        drug_name = request.form['drug_name']
        medication_duration = request.form['medication_duration']
        addication_name = request.form['addication_name']
        addication_frequency = request.form['addication_frequency']
        addication_duration = request.form['addication_duration']

        medical_data = MedicalData(
            user=current_user.username,
            first_name = first_name,
            last_name = last_name,
            age = age,
            profession=profession,
            working_hours=working_hours,
            gender=gender,
            height=height ,
            weigth=weigth,
            previous_surgery_name=previous_surgery_name,
            previous_surgery_date=previous_surgery_date,
            complications_during_surgery=complications_during_surgery,
            anestesia_history=anestesia_history,
            chronic_conditions=chronic_conditions,
            current_medications=current_medications,
            known_allergies=known_allergies,
            disease=disease,
            drug_name=drug_name,
            medication_duration=medication_duration,
            addication_name=addication_name,
            addication_frequency=addication_frequency,
            addication_duration=addication_duration
        )
        medical_data.save()
        flash('Medical data added successfully', 'success')
        print('#####')
        print(current_user)
        return redirect(url_for('index', username=current_user.username))
    
    return render_template('profile.html')

@app.route('/calender')
@login_required
def calender():
    return render_template('calender.html')



file_path = r"C:\Users\kunda\Downloads\train.csv"
questions, answers, question_embeddings = None, None, None

# Function to load data and compute embeddings

# Path to cache the embeddings and related data
cache_file_path = "embedding_cache.pkl"

@lru_cache(maxsize=None)
def load_data_and_compute_embeddings(file_path):
    # Initialize the Sentence-BERT model (needed regardless of cache)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Try loading the cached embeddings
    if os.path.exists(cache_file_path):
        try:
            with open(cache_file_path, 'rb') as f:
                cached_data = pickle.load(f)
                questions = cached_data['questions']
                answers = cached_data['answers']
                question_embeddings = cached_data['question_embeddings']
                print("Loaded cached embeddings.")
                return questions, model, answers, question_embeddings
        except (EOFError, pickle.UnpicklingError):
            print("Cache file is corrupted or incomplete. Recomputing embeddings...")
            os.remove(cache_file_path)  # Remove the corrupted cache file
    
    # Load the dataset if no valid cache is found
    print("Cache not found. Computing embeddings...")
    data = pd.read_csv(file_path)
    questions = data['Question'].values
    answers = data['Answer'].values
    
    # Precompute question embeddings
    question_embeddings = model.encode(questions, convert_to_tensor=True, show_progress_bar=True)
    
    # Cache the embeddings and data
    with open(cache_file_path, 'wb') as f:
        pickle.dump({
            'questions': questions,
            'answers': answers,
            'question_embeddings': question_embeddings
        }, f)
    
    print("Embeddings computed and cached.")
    return questions, model, answers, question_embeddings

# Load dataset and compute embeddings
questions,model, answers, question_embeddings = load_data_and_compute_embeddings(file_path)

# Define a dictionary of generic responses for greetings and other conversational inputs
generic_responses = {
    "hi": "Hello! How can I assist you today?",
    "hello": "Hi there! How can I help you?",
    "hey": "Hey! How can I assist?",
    "how are you": "I'm a chatbot, but I'm here to help you with any medical questions!",
    "good morning": "Good morning! How can I assist you today?",
    "good afternoon": "Good afternoon! What can I do for you today?",
    "good evening": "Good evening! How can I assist you?",
    "thanks": "You're welcome! Feel free to ask more questions.",
    "thank you": "You're welcome! How else can I assist you?",
}

# Function to handle generic inputs with predefined responses
def get_generic_response(user_input):
    user_input_lower = user_input.lower()
    for greeting, response in generic_responses.items():
        if greeting in user_input_lower:
            return response
    return None

# Function to find the best match for a given question
def get_best_match(user_question, model, question_embeddings, questions, answers, top_n=5):
    # Encode the user question
    user_question_embedding = model.encode(user_question, convert_to_tensor=True)
    similarity_scores = util.pytorch_cos_sim(user_question_embedding, question_embeddings).flatten()
    top_n_indices = similarity_scores.topk(k=top_n).indices.cpu().numpy()
    best_question = questions[top_n_indices[0]]  # Get the best match
    best_answer = answers[top_n_indices[0]]
    
    if similarity_scores[top_n_indices[0]] < 0.5:
        return "I am sorry, can you describe your question more precisely?"
    
    return best_answer

@app.route('/chat', methods=["GET", "POST"])
@login_required
def chat():
    user_input = ''
    output = ''
    if request.method == "POST":
        user_input = request.form['user_input']
        # response_arr.append(output)

        generic_response = get_generic_response(user_input)
        if generic_response:
            output= generic_response
        else:
            response = get_best_match(user_input, model, question_embeddings, questions, answers, top_n=5)
            output=response
        
        chat_record = ChatBot(
            owner=current_user.username,  # Stores current user's ID as owner
            user_input=user_input,
            output=output
        )

        chat_record.save()

    
    

    return render_template('chat.html', user_input=user_input, output=output)

# Wait for the page to load completely



@app.route('/reminders', methods=["GET", "POST"])
@login_required
def reminder():
    doctor_data = []  # Initialize doctor_data to avoid reference errors
    if request.method == "POST":
        specialist = request.form['specialist']
        location = request.form['location']
        
        # Ensure the scrape_doctors function is called with the correct parameters
        doctor_data = scrape_doctors(specialist=specialist, location=location)
        
        # Debug output to check if data is returned
        print(doctor_data)

    return render_template('appointmentdetails.html', doctor_data=doctor_data)

if __name__ == '__main__':
    app.run(debug=True)