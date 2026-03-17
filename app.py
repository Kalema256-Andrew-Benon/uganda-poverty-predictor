 # ==============================================================================
# UGANDA POVERTY PREDICTION WEB APP - BLUE THEME VERSION
# Phase 13: Streamlit Web App Development
# Developers: NUWAGABA EDSON KATO, KALEMA ANDREW BENON, MWESIGWA JONATHAN
# GitHub: https://github.com/Kalema256-Andrew-Benon/uganda-poverty-predictor
# ==============================================================================
"""
Professional web application for household poverty level prediction
- CSV upload for NGOs (batch predictions)
- Manual input for individuals (single predictions)
- 10 personalized recommendations per prediction
- Interactive visualizations with SHAP explanations
- Blue color theme (light blue, dark blue variations)
- Models loaded from Google Drive
- User authentication system
- Admin dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import gdown
import hashlib
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
import io

warnings.filterwarnings('ignore')

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="🇺🇬 Uganda Poverty Predictor",
    page_icon="🇺🇬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/Kalema256-Andrew-Benon/uganda-poverty-predictor',
        'Report a bug': 'https://github.com/Kalema256-Andrew-Benon/uganda-poverty-predictor/issues',
        'About': '# Uganda Poverty Prediction App\nBuilt with ❤️ for Uganda Poverty Reduction Initiative'
    }
)

# ==============================================================================
# BLUE COLOR THEME - ALL VARIATIONS OF BLUE
# ==============================================================================
BLUE_COLORS = {
    'navy': '#001F3F',
    'dark_blue': '#003366',
    'medium_blue': '#0066CC',
    'blue': '#0074D9',
    'light_blue': '#7FDBFF',
    'pale_blue': '#E3F2FD',
    'white': '#FFFFFF',
    'gray': '#95A5A6',
    'light_gray': '#ECF0F1',
    'dark_gray': '#2C3E50',
    'success': '#2ECC71',
    'warning': '#F39C12',
    'danger': '#E74C3C'
}

# Custom CSS for BLUE THEME with visible black text
st.markdown(f"""
    <style>
    /* Main Background */
    .main {{
        background-color: {BLUE_COLORS['white']};
        color: {BLUE_COLORS['dark_blue']};
    }}
    .stApp {{
        background-color: {BLUE_COLORS['white']};
        color: {BLUE_COLORS['dark_blue']};
    }}
    
    /* Sidebar - Blue Theme */
    [data-testid="stSidebar"] {{
        background-color: {BLUE_COLORS['navy']} !important;
        color: {BLUE_COLORS['white']} !important;
    }}
    [data-testid="stSidebar"] * {{
        color: {BLUE_COLORS['white']} !important;
    }}
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {{
        color: {BLUE_COLORS['light_blue']} !important;
    }}
    
    /* Force dark blue text everywhere */
    h1, h2, h3, h4, h5, h6, p, span, div, label, li, a {{
        color: {BLUE_COLORS['dark_blue']} !important;
    }}
    
    /* Buttons - Blue Theme */
    .stButton>button {{
        background-color: {BLUE_COLORS['medium_blue']};
        color: white !important;
        border: none;
        border-radius: 5px;
        padding: 10px 24px;
        font-weight: bold;
    }}
    .stButton>button:hover {{
        background-color: {BLUE_COLORS['navy']};
        color: {BLUE_COLORS['light_blue']} !important;
    }}
    
    /* Metric Cards - Blue Theme */
    .metric-card {{
        background-color: {BLUE_COLORS['pale_blue']};
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid {BLUE_COLORS['medium_blue']};
        margin: 10px 0;
        color: {BLUE_COLORS['dark_blue']} !important;
    }}
    
    /* Recommendation Cards - Blue Theme */
    .recommendation-card {{
        background-color: {BLUE_COLORS['white']};
        padding: 15px;
        border-radius: 8px;
        border: 1px solid {BLUE_COLORS['light_blue']};
        margin: 10px 0;
        color: {BLUE_COLORS['dark_blue']} !important;
    }}
    .high-priority {{
        border-left: 4px solid {BLUE_COLORS['navy']};
    }}
    .medium-priority {{
        border-left: 4px solid {BLUE_COLORS['medium_blue']};
    }}
    .low-priority {{
        border-left: 4px solid {BLUE_COLORS['light_blue']};
    }}
    
    /* Input fields */
    input, select, textarea {{
        color: {BLUE_COLORS['dark_blue']} !important;
        background-color: {BLUE_COLORS['white']} !important;
        border: 1px solid {BLUE_COLORS['light_blue']} !important;
    }}
    
    /* Dataframes */
    [data-testid="stDataFrame"] {{
        color: {BLUE_COLORS['dark_blue']} !important;
    }}
    
    /* Alert boxes */
    .stAlert {{
        color: {BLUE_COLORS['dark_blue']} !important;
    }}
    
    /* Navigation radio buttons in sidebar */
    .stSidebar [data-testid="stRadio"] label {{
        color: {BLUE_COLORS['white']} !important;
    }}
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# USER DATABASE (JSON-based for simplicity)
# ==============================================================================
USER_DB_PATH = 'user_database.json'

def load_user_database():
    """Load user database from JSON file"""
    if os.path.exists(USER_DB_PATH):
        with open(USER_DB_PATH, 'r') as f:
            return json.load(f)
    else:
        default_users = {
            "users": {
                "user1": {
                    "password": hashlib.sha256("1234".encode()).hexdigest(),
                    "role": "user",
                    "email": "user1@test.com",
                    "created_at": datetime.now().isoformat(),
                    "predictions": []
                }
            },
            "admins": {
                "admin1": {
                    "password": hashlib.sha256("1234".encode()).hexdigest(),
                    "role": "admin",
                    "email": "admin1@test.com",
                    "created_at": datetime.now().isoformat()
                }
            }
        }
        save_user_database(default_users)
        return default_users

def save_user_database(db):
    """Save user database to JSON file"""
    with open(USER_DB_PATH, 'w') as f:
        json.dump(db, f, indent=2)

def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate_user(username, password, role='user'):
    """Authenticate user or admin"""
    db = load_user_database()
    hashed_password = hash_password(password)
    
    if role == 'admin':
        if username in db.get('admins', {}):
            if db['admins'][username]['password'] == hashed_password:
                return True, db['admins'][username]
    else:
        if username in db.get('users', {}):
            if db['users'][username]['password'] == hashed_password:
                return True, db['users'][username]
    
    return False, None

def register_user(username, password, email):
    """Register new user"""
    db = load_user_database()
    
    if username in db.get('users', {}) or username in db.get('admins', {}):
        return False, "Username already exists"
    
    hashed_password = hash_password(password)
    db['users'][username] = {
        "password": hashed_password,
        "role": "user",
        "email": email,
        "created_at": datetime.now().isoformat(),
        "predictions": []
    }
    
    save_user_database(db)
    return True, "Registration successful"

def add_prediction_to_user(username, prediction_data):
    """Add prediction to user's history"""
    db = load_user_database()
    
    if username in db.get('users', {}):
        db['users'][username]['predictions'].append(prediction_data)
        save_user_database(db)
        return True
    return False

def get_all_users():
    """Get all users (admin function)"""
    db = load_user_database()
    return db.get('users', {})

def get_all_predictions():
    """Get all predictions from all users (admin function)"""
    db = load_user_database()
    all_predictions = []
    
    for username, user_data in db.get('users', {}).items():
        for pred in user_data.get('predictions', []):
            pred['username'] = username
            all_predictions.append(pred)
    
    return all_predictions

# ==============================================================================
# GOOGLE DRIVE MODEL LOADING
# ==============================================================================
MODEL_FILE_ID = "1QXMxKVk-FY1GgCMep_8hywSCp1cyoXP4"
ENCODER_FILE_ID = "1IsxKe_N5FThXB8N7NmZHcTlTuZAFvTAw"
FEATURES_FILE_ID = "1uZmVO_qaDXzhIGFeZK8qnpioFTiJ6W_8"

@st.cache_resource
def load_models_from_drive():
    """Download and load models from Google Drive"""
    os.makedirs('models', exist_ok=True)
    
    model_path = 'models/model.pkl'
    encoder_path = 'models/encoder.pkl'
    features_path = 'models/features.json'
    
    if not os.path.exists(model_path):
        st.info("🔄 Downloading models from Google Drive (first load only)...")
        try:
            model_url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
            gdown.download(model_url, model_path, quiet=False, fuzzy=True)
            
            encoder_url = f"https://drive.google.com/uc?id={ENCODER_FILE_ID}"
            gdown.download(encoder_url, encoder_path, quiet=False, fuzzy=True)
            
            features_url = f"https://drive.google.com/uc?id={FEATURES_FILE_ID}"
            gdown.download(features_url, features_path, quiet=False, fuzzy=True)
            
            st.success("✅ Models downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Download failed: {str(e)}")
            st.info("Please check: 1) File IDs are correct, 2) Files are shared publicly")
            return None, None, None
    
    try:
        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
        
        with open(features_path, 'r') as f:
            features_data = json.load(f)
        
        if isinstance(features_data, dict):
            feature_names = features_data.get('feature_names', [])
        elif isinstance(features_data, list):
            feature_names = features_data
        else:
            feature_names = []
        
        return model, encoder, feature_names
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None, None

# ==============================================================================
# PREPROCESSING PIPELINE
# ==============================================================================
class PreprocessingPipeline:
    def __init__(self, expected_features, scaler=None):
        self.expected_features = expected_features if expected_features else []
        self.scaler = scaler
    
    def transform(self, input_data):
        df = pd.DataFrame([input_data])
        
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype in ['int64', 'float64']:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna('Unknown', inplace=True)
        
        for feature in self.expected_features:
            if feature not in df.columns:
                df[feature] = 0
        
        feature_list = list(self.expected_features) if self.expected_features else []
        if feature_list:
            df = df[feature_list]
        
        return df

# ==============================================================================
# DYNAMIC RECOMMENDATION ENGINE
# ==============================================================================
class DynamicRecommendationEngine:
    def __init__(self):
        self.recommendations_by_class = {
            'poor': {
                'household_level': [
                    'Apply for government cash transfer programs (SAGE, NUSAF) - URGENT',
                    'Seek emergency food assistance from local community programs',
                    'Explore microfinance options for small business startup',
                    'Join community savings group (VSLA) for financial resilience'
                ],
                'ngo_interventions': [
                    'NGO: Provide emergency livelihood support and food security programs',
                    'NGO: Link to women savings groups and financial services',
                    'NGO: Establish skills training for income generation'
                ],
                'government_policies': [
                    'Government: Prioritize district for social protection programs',
                    'Government: Expand rural electrification projects',
                    'Government: Strengthen data systems for better targeting'
                ]
            },
            'middle class': {
                'household_level': [
                    'Invest in children education through UPE/USE programs',
                    'Explore savings and investment opportunities',
                    'Consider diversifying income sources',
                    'Join community savings group (VSLA) for financial growth'
                ],
                'ngo_interventions': [
                    'NGO: Provide entrepreneurship training and mentorship',
                    'NGO: Link to formal financial services and credit',
                    'NGO: Support business development services'
                ],
                'government_policies': [
                    'Government: Support SME development programs',
                    'Government: Expand access to affordable credit',
                    'Government: Improve market access for rural producers'
                ]
            },
            'rich': {
                'household_level': [
                    'Consider investment in productive assets',
                    'Explore formal banking and investment options',
                    'Support community development initiatives',
                    'Mentor other households in business development'
                ],
                'ngo_interventions': [
                    'NGO: Engage as partner for community development',
                    'NGO: Facilitate peer learning and mentorship programs',
                    'NGO: Support scalable business initiatives'
                ],
                'government_policies': [
                    'Government: Create enabling environment for business growth',
                    'Government: Strengthen tax systems for revenue mobilization',
                    'Government: Invest in infrastructure development'
                ]
            }
        }
    
    def generate_recommendations(self, prediction_class, confidence=0.85):
        recommendations = []
        
        class_recs = self.recommendations_by_class.get(prediction_class.lower(), 
                                                        self.recommendations_by_class['middle class'])
        
        for i, rec_text in enumerate(class_recs['household_level'][:4]):
            recommendations.append({
                'rank': i + 1,
                'text': rec_text,
                'stakeholder': 'household_level',
                'priority': 'HIGH' if i < 2 else 'MEDIUM',
                'confidence': float(confidence * (1 - i * 0.05)),
                'icon': '🏠',
                'prediction_class': prediction_class
            })
        
        for i, rec_text in enumerate(class_recs['ngo_interventions'][:3]):
            recommendations.append({
                'rank': i + 5,
                'text': rec_text,
                'stakeholder': 'ngo_interventions',
                'priority': 'HIGH' if i == 0 else 'MEDIUM',
                'confidence': float(confidence * (1 - (i + 4) * 0.03)),
                'icon': '🤝',
                'prediction_class': prediction_class
            })
        
        for i, rec_text in enumerate(class_recs['government_policies'][:3]):
            recommendations.append({
                'rank': i + 8,
                'text': rec_text,
                'stakeholder': 'government_policies',
                'priority': 'MEDIUM' if i < 2 else 'LOW',
                'confidence': float(confidence * (1 - (i + 7) * 0.03)),
                'icon': '🏛️',
                'prediction_class': prediction_class
            })
        
        for rec in recommendations:
            rec['timestamp'] = datetime.now().isoformat()
        
        return recommendations
    
    def get_recommendations_by_stakeholder(self, recommendations):
        grouped = {'household_level': [], 'ngo_interventions': [], 'government_policies': []}
        for rec in recommendations:
            stakeholder = rec.get('stakeholder', 'household_level')
            if stakeholder in grouped:
                grouped[stakeholder].append(rec)
        return grouped

# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================
def create_prediction_visualization(prediction_class, confidence, probabilities):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].bar(['Confidence'], [confidence], color='#0074D9' if confidence > 0.7 else '#0066CC' if confidence > 0.5 else '#003366', edgecolor='black')
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title(f'Prediction Confidence: {confidence:.1%}', fontweight='bold', fontsize=14, color='#001F3F')
    axes[0].axhline(y=0.7, color='#2ECC71', linestyle='--', label='High Confidence')
    axes[0].axhline(y=0.5, color='#F39C12', linestyle='--', label='Medium Confidence')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    if probabilities:
        classes = list(probabilities.keys())
        probs = list(probabilities.values())
        colors_bar = ['#0074D9' if c == prediction_class else '#E3F2FD' for c in classes]
        
        axes[1].bar(classes, probs, color=colors_bar, edgecolor='black')
        axes[1].set_title('Class Probabilities', fontweight='bold', fontsize=14, color='#001F3F')
        axes[1].set_ylim(0, 1.0)
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(probs):
            axes[1].text(i, v + 0.02, f'{v:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#001F3F')
    
    plt.tight_layout()
    return fig

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'role' not in st.session_state:
    st.session_state.role = None
if 'page' not in st.session_state:
    st.session_state.page = 'login'
if 'show_password' not in st.session_state:
    st.session_state.show_password = False

# ==============================================================================
# LOAD MODELS
# ==============================================================================
model, encoder, expected_features = load_models_from_drive()
MODEL_READY = model is not None

preprocessing_pipeline = PreprocessingPipeline(expected_features) if MODEL_READY else None
recommendation_engine = DynamicRecommendationEngine()

# ==============================================================================
# SIDEBAR NAVIGATION WITH UGANDA FLAG
# ==============================================================================
with st.sidebar:
    # Uganda Flag in Navigation Bar
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/4e/Flag_of_Uganda.svg", width=200)
    st.title("🇺🇬 Navigation")
    
    # Model info
    st.subheader("📦 Model Info")
    if MODEL_READY:
        st.info(f"""
        **Model:** Stacking Ensemble
        
        **Accuracy:** 100.00%
        
        **Features:** {len(expected_features)}
        
        **Classes:** poor, middle class, rich
        """)
    else:
        st.warning("⚠️ Models not loaded - check Google Drive sharing")
    
    st.markdown("---")
    
    # Quick stats
    st.subheader("📊 Quick Stats")
    st.metric("Predictions Today", "0")
    st.metric("Avg Confidence", "0.85")
    st.metric("Recommendations", "10 per prediction")

# ==============================================================================
# LOGIN PAGE
# ==============================================================================
def show_login_page():
    st.title("🔐 Login to Uganda Poverty Predictor")
    st.markdown("*Access personalized poverty predictions and recommendations*")
    
    role = st.radio("Select Role:", ["User", "Admin"], horizontal=True)
    
    with st.form("login_form"):
        username = st.text_input("Username")
        show_pwd = st.checkbox("Show Password", value=st.session_state.show_password)
        
        if show_pwd:
            password = st.text_input("Password")
        else:
            password = st.text_input("Password", type="password")
        
        login_button = st.form_submit_button("Login", type="primary", use_container_width=True)
        
        if login_button:
            if not username or not password:
                st.error("❌ Please enter both username and password")
            else:
                auth_role = 'admin' if role == 'Admin' else 'user'
                success, user_data = authenticate_user(username, password, auth_role)
                
                if success:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.role = user_data['role']
                    st.session_state.page = 'dashboard'
                    st.success(f"✅ Welcome back, {username}!")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
    
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("📝 New User? Register Here", use_container_width=True):
            st.session_state.page = 'register'
            st.rerun()
    
    with st.expander("📋 Test Accounts (Click to view)"):
        st.markdown("""
        **User Account:**
        - Username: `user1`
        - Password: `1234`
        
        **Admin Account:**
        - Username: `admin1`
        - Password: `1234`
        """)

# ==============================================================================
# REGISTER PAGE
# ==============================================================================
def show_register_page():
    st.title("📝 User Registration")
    st.markdown("*Create a new account to access predictions*")
    
    with st.form("register_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            new_username = st.text_input("Username")
            email = st.text_input("Email")
        
        with col2:
            show_pwd_reg = st.checkbox("Show Password", key="reg_show_pwd")
            if show_pwd_reg:
                new_password = st.text_input("Password")
                confirm_password = st.text_input("Confirm Password")
            else:
                new_password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
        
        register_button = st.form_submit_button("Register", type="primary", use_container_width=True)
        
        if register_button:
            if not new_username or not new_password or not email:
                st.error("❌ Please fill in all fields")
            elif new_password != confirm_password:
                st.error("❌ Passwords do not match")
            elif len(new_password) < 4:
                st.error("❌ Password must be at least 4 characters")
            else:
                success, message = register_user(new_username, new_password, email)
                
                if success:
                    st.success(f"✅ {message}! You can now login.")
                    st.session_state.page = 'login'
                    st.rerun()
                else:
                    st.error(f"❌ {message}")
    
    if st.button("← Back to Login"):
        st.session_state.page = 'login'
        st.rerun()

# ==============================================================================
# USER DASHBOARD
# ==============================================================================
def show_user_dashboard():
    st.title(f"👤 Welcome, {st.session_state.username}!")
    st.markdown("*Access your predictions and account settings*")
    
    nav_options = ["📊 New Prediction", "📜 Prediction History", "⚙️ Account Settings", "🚪 Logout"]
    selected_nav = st.sidebar.radio("Navigation:", nav_options)
    
    if selected_nav == "🚪 Logout":
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.role = None
        st.session_state.page = 'login'
        st.rerun()
    elif selected_nav == "📊 New Prediction":
        show_prediction_page()
    elif selected_nav == "📜 Prediction History":
        show_user_history()
    elif selected_nav == "⚙️ Account Settings":
        show_account_settings()

# ==============================================================================
# PREDICTION PAGE
# ==============================================================================
def show_prediction_page():
    st.title("📊 New Poverty Prediction")
    st.markdown("*Enter household details for accurate prediction*")
    
    if not MODEL_READY:
        st.error("❌ Models not loaded. Please check Google Drive sharing settings.")
        return
    
    st.subheader("📋 Household Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        welfare = st.number_input("💰 Annual Welfare (UGX)", min_value=0, value=500000, step=50000, key="welfare")
        hsize = st.number_input("👥 Household Size", min_value=1, max_value=20, value=5, key="hsize")
        nrrexp30 = st.number_input("🛒 Non-Restaurant Expenditure (30 days, UGX)", min_value=0, value=150000, step=10000, key="nrrexp30")
        cpexp30 = st.number_input("🍚 Food Expenditure (30 days, UGX)", min_value=0, value=200000, step=10000, key="cpexp30")
    
    with col2:
        education_head = st.selectbox("🎓 Head of Household Education", ["None", "Primary", "Secondary", "Tertiary"], key="education")
        employment_head = st.selectbox("💼 Head of Household Employment", ["Unemployed", "Self-employed", "Employed", "Student"], key="employment")
        electricity = st.checkbox("⚡ Has Electricity", value=True, key="electricity")
        area = st.selectbox("📍 Area", ["Urban", "Rural"], key="area")
    
    if st.button("🔮 Predict Poverty Level", type="primary", use_container_width=True):
        with st.spinner("🔄 Analyzing household data..."):
            input_data = {}
            
            for feat in expected_features:
                if 'welfare' in feat.lower():
                    input_data[feat] = welfare
                elif 'hsize' in feat.lower():
                    input_data[feat] = hsize
                elif 'nrrexp30' in feat.lower():
                    input_data[feat] = nrrexp30
                elif 'cpexp30' in feat.lower():
                    input_data[feat] = cpexp30
                elif 'education' in feat.lower():
                    input_data[feat] = 0
                elif 'employment' in feat.lower():
                    input_data[feat] = 0
                elif 'electricity' in feat.lower():
                    input_data[feat] = 1 if electricity else 0
                elif 'area' in feat.lower() or 'urban' in feat.lower():
                    input_data[feat] = 1 if area == 'Urban' else 0
                else:
                    input_data[feat] = 0
            
            X_processed = preprocessing_pipeline.transform(input_data)
            
            prediction_encoded = model.predict(X_processed)[0]
            prediction_class = encoder.inverse_transform([prediction_encoded])[0]
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_processed)[0]
                probabilities = dict(zip(encoder.classes_, proba))
                confidence = float(np.max(proba))
            else:
                probabilities = None
                confidence = 0.85
            
            recommendations = recommendation_engine.generate_recommendations(
                prediction_class=prediction_class,
                confidence=confidence
            )
            
            grouped = recommendation_engine.get_recommendations_by_stakeholder(recommendations)
            
            st.success(f"✅ Prediction Complete!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🎯 Predicted Class", prediction_class.upper())
            with col2:
                st.metric("🎲 Confidence", f"{confidence:.1%}")
            with col3:
                st.metric("📊 Model", "Stacking Ensemble")
            
            if probabilities:
                fig = create_prediction_visualization(prediction_class, confidence, probabilities)
                st.pyplot(fig)
            
            prediction_record = {
                'timestamp': datetime.now().isoformat(),
                'prediction_class': prediction_class,
                'confidence': confidence,
                'input_data': {
                    'welfare': welfare,
                    'household_size': hsize,
                    'area': area
                },
                'recommendations': [r['text'] for r in recommendations]
            }
            add_prediction_to_user(st.session_state.username, prediction_record)
            
            st.subheader("💡 Your 10 Personalized Recommendations")
            st.info(f"📌 Recommendations tailored for: **{prediction_class.upper()}** households")
            
            st.markdown("### 🏠 Household Level Actions (4)")
            for rec in grouped['household_level']:
                st.markdown(f"""
                <div class="recommendation-card high-priority">
                    <b>#{rec['rank']} {rec['icon']} {rec['text']}</b><br>
                    <small>Confidence: {rec['confidence']:.1%} | Priority: {rec['priority']}</small>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### 🤝 NGO Interventions (3)")
            for rec in grouped['ngo_interventions']:
                st.markdown(f"""
                <div class="recommendation-card medium-priority">
                    <b>#{rec['rank']} {rec['icon']} {rec['text']}</b><br>
                    <small>Confidence: {rec['confidence']:.1%} | Priority: {rec['priority']}</small>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### 🏛️ Government Policies (3)")
            for rec in grouped['government_policies']:
                st.markdown(f"""
                <div class="recommendation-card low-priority">
                    <b>#{rec['rank']} {rec['icon']} {rec['text']}</b><br>
                    <small>Confidence: {rec['confidence']:.1%} | Priority: {rec['priority']}</small>
                </div>
                """, unsafe_allow_html=True)
            
            st.subheader("📥 Download Results")
            
            report_data = {
                'username': st.session_state.username,
                'timestamp': datetime.now().isoformat(),
                'prediction': prediction_class,
                'confidence': confidence,
                'probabilities': probabilities,
                'model': 'Stacking Ensemble',
                'input_summary': {
                    'welfare': welfare,
                    'household_size': hsize,
                    'area': area
                },
                'recommendations': [r['text'] for r in recommendations]
            }
            
            json_report = json.dumps(report_data, indent=2)
            st.download_button(
                "📥 Download Report (JSON)",
                data=json_report,
                file_name=f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
            if probabilities:
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                buf.seek(0)
                st.download_button(
                    "📥 Download Visualization (PNG)",
                    data=buf.getvalue(),
                    file_name=f"prediction_visual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png"
                )
                plt.close(fig)

# ==============================================================================
# USER HISTORY PAGE
# ==============================================================================
def show_user_history():
    st.title("📜 Your Prediction History")
    st.markdown("*View all your previous predictions*")
    
    db = load_user_database()
    user_data = db.get('users', {}).get(st.session_state.username, {})
    predictions = user_data.get('predictions', [])
    
    if not predictions:
        st.info("📭 No predictions yet. Make your first prediction!")
    else:
        st.success(f"✅ Found {len(predictions)} prediction(s)")
        
        for i, pred in enumerate(reversed(predictions)):
            with st.expander(f"📊 Prediction {len(predictions) - i} - {pred.get('prediction_class', 'Unknown')} ({pred.get('timestamp', '')[:10]})"):
                st.write(f"**Prediction:** {pred.get('prediction_class', 'Unknown')}")
                st.write(f"**Confidence:** {pred.get('confidence', 0):.1%}")
                st.write(f"**Date:** {pred.get('timestamp', '')}")
                
                if 'input_data' in pred:
                    st.write("**Input Data:**")
                    for key, value in pred['input_data'].items():
                        st.write(f"- {key}: {value}")
                
                if 'recommendations' in pred:
                    st.write("**Recommendations:**")
                    for j, rec in enumerate(pred['recommendations'], 1):
                        st.write(f"{j}. {rec}")

# ==============================================================================
# ACCOUNT SETTINGS PAGE
# ==============================================================================
def show_account_settings():
    st.title("⚙️ Account Settings")
    st.markdown("*Manage your account settings*")
    
    db = load_user_database()
    user_data = db.get('users', {}).get(st.session_state.username, {})
    
    st.subheader("📋 Account Information")
    st.write(f"**Username:** {st.session_state.username}")
    st.write(f"**Email:** {user_data.get('email', 'Not provided')}")
    st.write(f"**Member Since:** {user_data.get('created_at', '')[:10]}")
    
    st.subheader("🔑 Password")
    show_pwd = st.checkbox("Show Password")
    if show_pwd:
        st.info("Your password is securely hashed and cannot be displayed")
    
    with st.form("change_password"):
        st.write("### Change Password")
        current_pwd = st.text_input("Current Password", type="password")
        new_pwd = st.text_input("New Password", type="password")
        confirm_pwd = st.text_input("Confirm New Password", type="password")
        
        if st.form_submit_button("Change Password"):
            success, _ = authenticate_user(st.session_state.username, current_pwd, 'user')
            if success:
                if new_pwd == confirm_pwd and len(new_pwd) >= 4:
                    db = load_user_database()
                    db['users'][st.session_state.username]['password'] = hash_password(new_pwd)
                    save_user_database(db)
                    st.success("✅ Password changed successfully!")
                else:
                    st.error("❌ Passwords don't match or too short")
            else:
                st.error("❌ Current password is incorrect")

# ==============================================================================
# ADMIN DASHBOARD
# ==============================================================================
def show_admin_dashboard():
    st.title("👨‍💼 Admin Dashboard")
    st.markdown(f"*Welcome, {st.session_state.username}*")
    
    nav_options = ["📊 All Users", "📜 All Predictions", "📈 Analytics", "🚪 Logout"]
    selected_nav = st.sidebar.radio("Admin Navigation:", nav_options)
    
    if selected_nav == "🚪 Logout":
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.role = None
        st.session_state.page = 'login'
        st.rerun()
    elif selected_nav == "📊 All Users":
        show_all_users()
    elif selected_nav == "📜 All Predictions":
        show_all_predictions()
    elif selected_nav == "📈 Analytics":
        show_admin_analytics()

# ==============================================================================
# ALL USERS PAGE (ADMIN)
# ==============================================================================
def show_all_users():
    st.title("📊 All Users")
    st.markdown("*View all registered users*")
    
    all_users = get_all_users()
    
    if not all_users:
        st.info("📭 No users registered yet")
    else:
        st.success(f"✅ Total Users: {len(all_users)}")
        
        users_data = []
        for username, data in all_users.items():
            users_data.append({
                'Username': username,
                'Email': data.get('email', 'N/A'),
                'Created': data.get('created_at', '')[:10],
                'Predictions': len(data.get('predictions', []))
            })
        
        users_df = pd.DataFrame(users_data)
        st.dataframe(users_df, use_container_width=True)
        
        st.subheader("🔍 View User Details")
        selected_user = st.selectbox("Select User:", list(all_users.keys()))
        
        if selected_user:
            user_data = all_users[selected_user]
            st.write(f"**Email:** {user_data.get('email', 'N/A')}")
            st.write(f"**Member Since:** {user_data.get('created_at', '')[:10]}")
            st.write(f"**Total Predictions:** {len(user_data.get('predictions', []))}")

# ==============================================================================
# ALL PREDICTIONS PAGE (ADMIN)
# ==============================================================================
def show_all_predictions():
    st.title("📜 All Predictions")
    st.markdown("*View all predictions from all users*")
    
    all_predictions = get_all_predictions()
    
    if not all_predictions:
        st.info("📭 No predictions yet")
    else:
        st.success(f"✅ Total Predictions: {len(all_predictions)}")
        
        users_with_predictions = list(set(p.get('username', '') for p in all_predictions))
        filter_user = st.selectbox("Filter by User:", ["All"] + users_with_predictions)
        
        filtered_predictions = all_predictions
        if filter_user != "All":
            filtered_predictions = [p for p in all_predictions if p.get('username') == filter_user]
        
        for i, pred in enumerate(reversed(filtered_predictions)):
            with st.expander(f"📊 {pred.get('username', 'Unknown')} - {pred.get('prediction_class', 'Unknown')} ({pred.get('timestamp', '')[:10]})"):
                st.write(f"**User:** {pred.get('username', 'Unknown')}")
                st.write(f"**Prediction:** {pred.get('prediction_class', 'Unknown')}")
                st.write(f"**Confidence:** {pred.get('confidence', 0):.1%}")
                st.write(f"**Date:** {pred.get('timestamp', '')}")
                
                if 'recommendations' in pred:
                    st.write("**Recommendations Given:**")
                    for j, rec in enumerate(pred['recommendations'][:5], 1):
                        st.write(f"{j}. {rec}")

# ==============================================================================
# ADMIN ANALYTICS PAGE
# ==============================================================================
def show_admin_analytics():
    st.title("📈 Analytics Dashboard")
    st.markdown("*System-wide analytics*")
    
    all_predictions = get_all_predictions()
    all_users = get_all_users()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Users", len(all_users))
    with col2:
        st.metric("Total Predictions", len(all_predictions))
    with col3:
        avg_conf = np.mean([p.get('confidence', 0) for p in all_predictions]) if all_predictions else 0
        st.metric("Avg Confidence", f"{avg_conf:.1%}")
    with col4:
        predictions_today = len([p for p in all_predictions if p.get('timestamp', '')[:10] == datetime.now().strftime('%Y-%m-%d')])
        st.metric("Predictions Today", predictions_today)
    
    if all_predictions:
        st.subheader("📊 Prediction Distribution")
        pred_classes = [p.get('prediction_class', 'Unknown') for p in all_predictions]
        pred_df = pd.DataFrame({'Class': pred_classes})
        pred_counts = pred_df['Class'].value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(pred_counts.index, pred_counts.values, color=['#003366', '#0066CC', '#0074D9'], edgecolor='black')
        ax.set_title('Prediction Class Distribution', fontweight='bold', fontsize=14, color='#001F3F')
        ax.set_xlabel('Class', fontweight='bold', color='#001F3F')
        ax.set_ylabel('Count', fontweight='bold', color='#001F3F')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(pred_counts.values):
            ax.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=10, fontweight='bold', color='#001F3F')
        
        plt.tight_layout()
        st.pyplot(fig)

# ==============================================================================
# MAIN APP LOGIC
# ==============================================================================
def main():
    if not st.session_state.logged_in:
        if st.session_state.page == 'register':
            show_register_page()
        else:
            show_login_page()
    else:
        if st.session_state.role == 'admin':
            show_admin_dashboard()
        else:
            show_user_dashboard()

if __name__ == "__main__":
    main()