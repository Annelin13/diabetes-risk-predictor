import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, session, render_template, redirect, url_for, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime, timedelta
from flask_cors import CORS
import threading
import time
from dotenv import load_dotenv
load_dotenv("mail.env")

# ---------------- Flask Setup ----------------
app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'diabetes-prediction-secret-key-2025'

# Database setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///diabetes_users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Email setup
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'True').lower() == 'true'
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER')

# Email setup
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'True').lower() == 'true'
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER')

# Debug: Print email configuration
print("🔧 Email Configuration:")
print(f"MAIL_SERVER: {app.config['MAIL_SERVER']}")
print(f"MAIL_PORT: {app.config['MAIL_PORT']}")
print(f"MAIL_USE_TLS: {app.config['MAIL_USE_TLS']}")
print(f"MAIL_USERNAME: {app.config['MAIL_USERNAME']}")
print(f"MAIL_PASSWORD: {'*' * len(app.config['MAIL_PASSWORD']) if app.config['MAIL_PASSWORD'] else 'NOT SET'}")
print(f"MAIL_DEFAULT_SENDER: {app.config['MAIL_DEFAULT_SENDER']}")

# Warn if mail credentials are missing
if not app.config.get('MAIL_USERNAME') or not app.config.get('MAIL_PASSWORD'):
    print("⚠️ Warning: Mail credentials not found. Emails won’t send.")

# Initialize extensions
db = SQLAlchemy(app)
mail = Mail(app)


# ---------------- Database Models ----------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_test_reminder = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    age = db.Column(db.Float)
    gender = db.Column(db.String(10))
    height = db.Column(db.Float)
    weight = db.Column(db.Float)
    glucose = db.Column(db.Float)
    bp = db.Column(db.String(20))
    bmi = db.Column(db.Float)
    family_history = db.Column(db.String(10))
    pregnancies = db.Column(db.Integer)
    probability = db.Column(db.Float)
    risk = db.Column(db.String(20))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Create database tables
with app.app_context():
    try:
        db.create_all()
        print("Database tables checked/created successfully")

        # Create demo user if missing
        if not User.query.filter_by(email="demo@example.com").first():
            hashed_pw = generate_password_hash("demo123")
            demo_user = User(name="Demo User", email="demo@example.com", password=hashed_pw)
            db.session.add(demo_user)
            db.session.commit()
            print("Demo user created: demo@example.com / demo123")
    except Exception as e:
        print(f"Database setup error: {e}")

# ---------------- Email Utility Functions ----------------
def send_email(to, subject, template):
    """Send email to user"""
    try:
        msg = Message(
            subject=subject,
            recipients=[to],
            html=template
        )
        mail.send(msg)
        print(f"Email sent successfully to {to}")
        return True
    except Exception as e:
        print(f"Error sending email to {to}: {e}")
        return False

def send_welcome_email(user):
    """Send welcome email after registration"""
    subject = "Welcome to Diabetes Predictor - Start Your Health Journey!"
    
    template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; color: #333; background-color: #f5f5f5; margin: 0; padding: 20px; }}
            .container {{ max-width: 650px; margin: 0 auto; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
            .header {{ background: linear-gradient(135deg, #0a3d91, #357ab8); color: white; padding: 30px; text-align: center; }}
            .content {{ padding: 35px; line-height: 1.6; }}
            .button {{ background: #0a3d91; color: white !important; padding: 14px 28px; text-decoration: none; border-radius: 8px; display: inline-block; margin: 15px 0; font-weight: bold; font-size: 16px; text-align: center; }}
            .footer {{ text-align: center; margin-top: 25px; color: #666; font-size: 12px; padding: 20px; background: #f9f9f9; }}
            h1 {{ margin: 0; font-size: 28px; }}
            h2 {{ color: #0a3d91; margin-top: 0; }}
            h3 {{ color: #357ab8; margin: 20px 0 10px 0; }}
            ul {{ padding-left: 20px; }}
            li {{ margin-bottom: 8px; }}
            strong {{ color: #0a3d91; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎉 Welcome to Diabetes Predictor!</h1>
            </div>
            <div class="content">
                <h2>Hello {user.name}!</h2>
                <p>Thank you for joining our community dedicated to proactive health management.</p>
                
                <h3>🚀 Get Started Now:</h3>
                <p>Take your first diabetes risk assessment to establish your baseline health profile.</p>
                <a href="https://millesimally-graphological-chin.ngrok-free.dev" class="button" style="color: white !important;">Take Your First Test</a>
                
                <h3>📅 Monthly Health Check:</h3>
                <p>We'll remind you to take the test every month to track your health progress.</p>
                
                <h3>💡 Why Regular Testing Matters:</h3>
                <ul>
                    <li>Track changes in your health metrics</li>
                    <li>Monitor the effectiveness of lifestyle changes</li>
                    <li>Early detection of potential risks</li>
                    <li>Personalized health insights</li>
                </ul>
                
                <p><strong>Your next reminder will be sent in 1 month.</strong></p>
                
                <p>Stay healthy,<br><strong>The Diabetes Predictor Team</strong></p>
            </div>
            <div class="footer">
                <p>This is an automated message. Please do not reply to this email.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return send_email(user.email, subject, template)

def send_test_reminder_email(user, days_since_last_test):
    """Send monthly test reminder email"""
    subject = "📊 Time for Your Monthly Diabetes Risk Assessment!"
    
    template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; color: #333; background-color: #f5f5f5; margin: 0; padding: 20px; }}
            .container {{ max-width: 650px; margin: 0 auto; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
            .header {{ background: linear-gradient(135deg, #0a3d91, #357ab8); color: white; padding: 30px; text-align: center; }}
            .content {{ padding: 35px; line-height: 1.6; }}
            .button {{ background: #0a3d91; color: white !important; padding: 14px 28px; text-decoration: none; border-radius: 8px; display: inline-block; margin: 15px 0; font-weight: bold; font-size: 16px; text-align: center; }}
            .footer {{ text-align: center; margin-top: 25px; color: #666; font-size: 12px; padding: 20px; background: #f9f9f9; }}
            .urgent {{ background: #ff6b6b; color: white; padding: 15px; border-radius: 8px; margin: 20px 0; border-left: 5px solid #e74c3c; }}
            h1 {{ margin: 0; font-size: 28px; }}
            h2 {{ color: #0a3d91; margin-top: 0; }}
            h3 {{ color: #357ab8; margin: 20px 0 10px 0; }}
            ul {{ padding-left: 20px; }}
            li {{ margin-bottom: 8px; }}
            strong {{ color: #0a3d91; }}
            em {{ color: #666; font-style: italic; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📈 Monthly Health Check Reminder</h1>
            </div>
            <div class="content">
                <h2>Hello {user.name}!</h2>
                
                <p>It's been <strong>{days_since_last_test} days</strong> since your last diabetes risk assessment.</p>
                
                <div class="urgent">
                    <h3 style="color: white; margin-top: 0;">🩺 Why Regular Testing is Crucial:</h3>
                    <p style="margin-bottom: 0;">Regular monitoring helps detect changes early and allows for timely interventions.</p>
                </div>
                
                <h3>✅ Benefits of Monthly Testing:</h3>
                <ul>
                    <li>Track your health progress over time</li>
                    <li>Identify positive changes from lifestyle adjustments</li>
                    <li>Stay motivated with visible results</li>
                    <li>Early detection of concerning trends</li>
                </ul>
                
                <a href="https://millesimally-graphological-chin.ngrok-free.dev" class="button" style="color: white !important;">Take Your Monthly Test Now</a>
                
                <p><em>This test takes only 2 minutes and could provide valuable insights into your health!</em></p>
                
                <p>Stay proactive about your health,<br><strong>The Diabetes Predictor Team</strong></p>
            </div>
            <div class="footer">
                <p>You can manage your email preferences in your account settings.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return send_email(user.email, subject, template)

# ---------------- Background Task for Email Reminders ----------------
def check_and_send_reminders():
    """Check for users who need test reminders"""
    with app.app_context():
        try:
            # Find active users who haven't taken a test in the last 30 days
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            
            users_to_remind = User.query.filter(
                User.is_active == True,
                User.last_test_reminder <= thirty_days_ago
            ).all()
            
            for user in users_to_remind:
                # Get user's last test date
                last_test = Prediction.query.filter_by(user_id=user.id).order_by(Prediction.created_at.desc()).first()
                
                if last_test:
                    days_since_last_test = (datetime.utcnow() - last_test.created_at).days
                else:
                    days_since_last_test = (datetime.utcnow() - user.created_at).days
                
                # Send reminder if it's been more than 30 days
                if days_since_last_test >= 30:
                    print(f"Sending reminder to {user.email} - {days_since_last_test} days since last test")
                    if send_test_reminder_email(user, days_since_last_test):
                        # Update last reminder timestamp
                        user.last_test_reminder = datetime.utcnow()
                        db.session.commit()
            
            print(f"Reminder check completed. Processed {len(users_to_remind)} users.")
            
        except Exception as e:
            print(f"Error in reminder task: {e}")

def start_reminder_scheduler():
    """Start background scheduler for email reminders"""
    def run_scheduler():
        while True:
            try:
                check_and_send_reminders()
            except Exception as e:
                print(f"Scheduler error: {e}")
            # Run every 24 hours
            time.sleep(24 * 60 * 60)  # 24 hours
    
    # Start the scheduler in a separate thread
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    print("Email reminder scheduler started")

# ---------------- ML Model Setup ----------------
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

def compute_bmi(h, w):
    try:
        return float(w) / ((float(h) / 100) ** 2) if h and w else None
    except:
        return None

def bucket_risk(p):
    if p < 0.33: return "Low"
    if p < 0.66: return "Medium"
    return "High"

def load_and_preprocess_diabetes_data():
    """Load and preprocess the diabetes.csv dataset"""
    try:
        df = pd.read_csv('diabetes.csv')
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print("ERROR: diabetes.csv file not found.")
        return None
    except Exception as e:
        print(f"Error loading diabetes.csv: {e}")
        return None

def create_sample_dataset():
    """Create a sample dataset if diabetes.csv is not available"""
    print("Creating sample dataset as fallback...")
    np.random.seed(42)
    n_samples = 1000
    
    pregnancies = np.random.poisson(2, n_samples)
    glucose = np.random.normal(120, 30, n_samples)
    blood_pressure = np.random.normal(70, 12, n_samples)
    skin_thickness = np.random.normal(20, 10, n_samples)
    insulin = np.random.normal(80, 100, n_samples)
    bmi = np.random.normal(32, 8, n_samples)
    diabetes_pedigree = np.random.exponential(0.5, n_samples)
    age = np.random.normal(33, 12, n_samples)
    
    z = (0.1 * pregnancies + 0.05 * (glucose-100) + 0.02 * (blood_pressure-70) + 
         0.03 * (bmi-25) + 0.1 * diabetes_pedigree + 0.01 * (age-30))
    probability = 1 / (1 + np.exp(-z))
    outcome = (probability > 0.5).astype(int)
    
    data = {
        'Pregnancies': pregnancies, 'Glucose': glucose, 'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness, 'Insulin': insulin, 'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree, 'Age': age, 'Outcome': outcome
    }
    
    df = pd.DataFrame(data)
    
    # Clip values to realistic ranges
    df['Pregnancies'] = np.clip(df['Pregnancies'], 0, 15).astype(int)
    df['Glucose'] = np.clip(df['Glucose'], 50, 200)
    df['BloodPressure'] = np.clip(df['BloodPressure'], 40, 120)
    df['SkinThickness'] = np.clip(df['SkinThickness'], 10, 60)
    df['Insulin'] = np.clip(df['Insulin'], 0, 300)
    df['BMI'] = np.clip(df['BMI'], 18, 50)
    df['Age'] = np.clip(df['Age'], 20, 80)
    
    print("Sample dataset created successfully")
    return df

def train_or_load():
    """Train model on diabetes.csv or load existing model"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print("Existing model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading existing model: {e}. Training new model...")
    
    df = load_and_preprocess_diabetes_data()
    if df is None:
        print("Falling back to sample dataset...")
        df = create_sample_dataset()
    
    feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'Age']
    available_cols = [col for col in feature_columns if col in df.columns]
    
    if not available_cols:
        available_cols = [col for col in df.columns if col != 'Outcome' and pd.api.types.is_numeric_dtype(df[col])]
        print(f"Using available numeric columns: {available_cols}")
    
    if not available_cols:
        print("ERROR: No suitable features found in the dataset!")
        return None
    
    print(f"Using features: {available_cols}")
    
    X = df[available_cols]
    y = df["Outcome"]
    
    # Handle zeros as missing values
    columns_to_clean = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in columns_to_clean:
        if col in X.columns:
            X.loc[:, col] = X[col].replace(0, np.nan)

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    preprocessor = ColumnTransformer([("num", num_pipe, available_cols)])
    
    model = Pipeline([
        ("pre", preprocessor), 
        ("clf", LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Training model...")
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print("\n" + "="*50)
    print("MODEL TRAINING RESULTS")
    print("="*50)
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Testing accuracy: {test_score:.4f}")
    print(f"Features used: {available_cols}")
    print("="*50)
    
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    return model

# Initialize model
try:
    MODEL = train_or_load()
    if MODEL is not None:
        print("Model loaded/trained successfully")
    else:
        print("WARNING: Model initialization failed. Using fallback predictions.")
except Exception as e:
    print(f"Error initializing model: {e}")
    MODEL = None

# ---------------- Routes ----------------
@app.route("/")
def home():
    return render_template('index.html')

@app.route("/login")
def login():
    return render_template('login.html')

@app.route("/register")
def register():
    return render_template('register.html')

# ---------------- API Endpoints ----------------
@app.route("/api/status")
def api_status():
    try:
        if 'user_id' in session:
            user = db.session.get(User, session['user_id'])
            if user:
                return jsonify({'logged_in': True, 'name': user.name})
        return jsonify({'logged_in': False})
    except Exception as e:
        return jsonify({'logged_in': False})

@app.route("/api/login", methods=["POST"])
def api_login():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
            
        email = data.get('email', '').strip()
        password = data.get('password', '')

        if not email or not password:
            return jsonify({'success': False, 'error': 'Email and password are required'}), 400

        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['user_name'] = user.name
            return jsonify({'success': True, 'name': user.name})
        return jsonify({'success': False, 'error': 'Invalid email or password'}), 401
    except Exception as e:
        return jsonify({'success': False, 'error': 'Server error during login'}), 500

@app.route("/api/register", methods=["POST"])
def api_register():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify(success=False, error="No data provided"), 400
            
        name = data.get("name", "").strip()
        email = data.get("email", "").strip()
        password = data.get("password", "")
        confirm_password = data.get("confirm_password", "")

        if not name or not email or not password or not confirm_password:
            return jsonify(success=False, error="All fields are required."), 400

        if password != confirm_password:
            return jsonify(success=False, error="Passwords do not match."), 400

        if len(password) < 6:
            return jsonify(success=False, error="Password must be at least 6 characters long."), 400

        if User.query.filter_by(email=email).first():
            return jsonify(success=False, error="Email already registered."), 400

        hashed_pw = generate_password_hash(password)
        user = User(name=name, email=email, password=hashed_pw)
        db.session.add(user)
        db.session.commit()
        
        # Send welcome email
        try:
            send_welcome_email(user)
        except Exception as email_error:
            print(f"Failed to send welcome email: {email_error}")
        
        # Auto-login after registration
        session['user_id'] = user.id
        session['user_name'] = user.name
        
        return jsonify(success=True, name=user.name, redirect_url="/")
        
    except Exception as e:
        return jsonify(success=False, error="Server error during registration"), 500

@app.route("/api/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out successfully'})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Please login first'}), 401

    try:
        data = request.get_json()

        # Parse inputs
        age = float(data.get("age", 0))
        glucose = float(data.get("glucose", 0))
        bp_input = data.get("bp", "120/80")
        
        # Extract systolic BP
        try:
            if '/' in bp_input:
                bp_systolic = float(bp_input.split('/')[0])
            else:
                bp_systolic = float(bp_input)
        except:
            bp_systolic = 120.0
            
        height = float(data.get("height", 0)) if data.get("height") else None
        weight = float(data.get("weight", 0)) if data.get("weight") else None
        
        # Calculate BMI
        bmi = compute_bmi(height, weight)
        if bmi is None:
            bmi = 25.0

        # Get pregnancies
        pregnancies = int(data.get("pregnancies", 0))
        if data.get("gender") == "Male":
            pregnancies = 0

        # Prepare data for model prediction
        if MODEL is not None:
            try:
                feature_names = MODEL.named_steps['pre'].transformers_[0][2]
                input_data = {}
                
                feature_mapping = {
                    'Pregnancies': pregnancies,
                    'Glucose': glucose,
                    'BloodPressure': bp_systolic,
                    'BMI': bmi,
                    'Age': age
                }
                
                for feature in feature_names:
                    if feature in feature_mapping:
                        input_data[feature] = feature_mapping[feature]
                    else:
                        if feature == 'SkinThickness':
                            input_data[feature] = 20.0
                        elif feature == 'Insulin':
                            input_data[feature] = 80.0
                        elif feature == 'DiabetesPedigreeFunction':
                            input_data[feature] = 0.5
                        else:
                            input_data[feature] = 0.0
                
                row = pd.DataFrame([input_data])[feature_names]
                prob = MODEL.predict_proba(row)[0, 1]
                
            except Exception as model_error:
                print(f"Model prediction error: {model_error}. Using fallback.")
                prob = calculate_fallback_risk(age, glucose, bmi, pregnancies)
        else:
            prob = calculate_fallback_risk(age, glucose, bmi, pregnancies)
            
        risk = bucket_risk(prob)

        # Save prediction in DB
        new_pred = Prediction(
            user_id=session['user_id'],
            age=age,
            gender=data.get("gender"),
            height=height,
            weight=weight,
            glucose=glucose,
            bp=data.get("bp"),
            bmi=bmi,
            family_history=data.get("family_history"),
            pregnancies=pregnancies,
            probability=prob,
            risk=risk
        )
        db.session.add(new_pred)
        
        # Update user's last test reminder timestamp
        user = db.session.get(User, session['user_id'])
        user.last_test_reminder = datetime.utcnow()
        db.session.commit()

        return jsonify({
            "success": True,
            "probability": round(float(prob), 4),
            "risk": risk,
            "risk_score": int(prob * 100),
            "inputs": {
                "age": age, 
                "glucose": glucose, 
                "bp": data.get("bp"), 
                "height": height, 
                "weight": weight, 
                "bmi": round(bmi, 1),
                "gender": data.get("gender"),
                "family_history": data.get("family_history"),
                "pregnancies": pregnancies
            }
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Prediction failed: {str(e)}"
        }), 500

def calculate_fallback_risk(age, glucose, bmi, pregnancies):
    """Fallback risk calculation when model is not available"""
    risk_score = 0
    
    if age >= 45: risk_score += 0.25
    elif age >= 35: risk_score += 0.15
    elif age >= 25: risk_score += 0.05
    
    if glucose >= 126: risk_score += 0.30
    elif glucose >= 100: risk_score += 0.15
    
    if bmi >= 30: risk_score += 0.25
    elif bmi >= 25: risk_score += 0.15
    
    if pregnancies >= 3: risk_score += 0.10
    
    return min(risk_score, 0.95)

@app.route("/api/history")
def api_history():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Please login first'}), 401
    
    try:
        preds = Prediction.query.filter_by(user_id=session['user_id']).order_by(Prediction.created_at.desc()).all()
        history = [
            {
                "id": p.id,
                "date": p.created_at.strftime("%Y-%m-%d %H:%M"),
                "age": p.age, 
                "glucose": p.glucose, 
                "bp": p.bp, 
                "bmi": round(p.bmi, 1) if p.bmi else None,
                "probability": round(p.probability, 4), 
                "risk": p.risk,
                "risk_score": int(p.probability * 100)
            }
            for p in preds
        ]
        return jsonify(history)
    except Exception as e:
        return jsonify({'success': False, 'error': 'Failed to load history'}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route("/api/clear_history", methods=["POST"])
def api_clear_history():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Please login first'}), 401
    
    try:
        deleted_count = Prediction.query.filter_by(user_id=session['user_id']).delete()
        db.session.commit()
        return jsonify({'success': True, 'message': f'History cleared successfully ({deleted_count} records deleted)'})
    except Exception as e:
        return jsonify({'success': False, 'error': f'Failed to clear history: {str(e)}'}), 500

@app.route("/test-email")
def test_email():
    """Test email functionality"""
    try:
        # Check if we have email credentials
        if not app.config.get('MAIL_USERNAME') or not app.config.get('MAIL_PASSWORD'):
            return jsonify({
                "success": False,
                "error": "Email credentials not configured"
            })
        
        # Try to send a test email
        test_user = User(
            name="Test User",
            email=app.config['MAIL_USERNAME']  # Send to yourself
        )
        
        if send_welcome_email(test_user):
            return jsonify({
                "success": True,
                "message": f"Test email sent to {app.config['MAIL_USERNAME']}"
            })
        else:
            return jsonify({
                "success": False,
                "error": "Failed to send email"
            })
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Email test failed: {str(e)}"
        })

# ---------------- Run ----------------
if __name__ == "__main__":
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        start_reminder_scheduler()

    print("Starting Diabetes Prediction API...")
    print("Demo credentials: demo@example.com / demo123")
    print("Access the application at: http://localhost:5000")
    print("Email reminder system: ACTIVE")
    app.run(debug=True, port=5000)
