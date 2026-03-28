import streamlit as st   
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from PIL import Image


st.set_page_config(page_title="AI Crop Disease Diagnosis", layout="wide", initial_sidebar_state="collapsed")

def apply_global_styles():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Lexend:wght@300;400;700&display=swap');
    
            html, body, [data-testid="stAppViewContainer"] {
                overflow: hidden !important;
                height: 100% !important;
                margin: 0;
                padding: 0;
                background-color: #113425 !important;
            }

            [data-testid="stAppViewContainer"],
            [data-testid="stAppViewContainer"] > div,
            section.main,
            .block-container {
                overflow: hidden !important;
                height: 100vh !important;
                max-height: 100vh !important;
                padding: 0 !important;
            }

            [data-testid="stVerticalBlock"] {
                overflow: hidden !important;
            }

            [data-testid="stSidebar"] {
                display: none;
            }

            header, footer {
                visibility: hidden;
            }

            .content-wrapper {
                height: 100vh;
                overflow-y: auto;
                padding: 40px;
                box-sizing: border-box;
            }
            
            .navbar {
    width: 100%;
    display: flex;
    justify-content: flex-end;
    gap: 10px;
    padding: 20px 40px;
}
                
/* NAVBAR CONTAINER - Making buttons look like text */
.navbar-container div.stButton > button {
    background: none !important;
    border: none !important;
    color: #dfffb6 !important;
    font-size: 18px !important;
    font-family: 'Lexend' !important;
    font-weight: 400 !important; /* Normal text weight */
    padding: 0 !important;
    margin: 0 !important;
    box-shadow: none !important;
    text-decoration: none !important;
    transition: color 0.3s ease;
    width: auto !important;
    min-height: 0 !important;
    height: auto !important;
}

/* HOVER EFFECT */
.navbar-container div.stButton > button:hover {
    color: #f6ae1e !important; /* Changes color on hover */
    background: none !important;
    text-decoration: underline !important; /* Optional: adds underline on hover */
}

/* ACTIVE STATE - Prevents the 'click' gray background */
.navbar-container div.stButton > button:active {
    background: none !important;
    color: #f6ae1e !important;
}

            input[type="text"], input[type="password"] {
                border: 1px solid #dfffb6 !important;
                border-radius: 10px !important;
                padding: 10px !important;
                color: #113425 !important;
            }

            input[type="text"]:focus, input[type="password"]:focus {
                border: 1px solid #f6ae1e !important;
                outline: none !important;
                box-shadow: 0 0 5px #f6ae1e;
            }
                
            h1 {
                font-family: 'Lexend';
                color: #dfffb6;
            }

            p {
                font-family: 'Lexend';
                color: #dfffb6;
            }

            div.stButton > button {
                background-color: #f6ae1e !important;
                color: #113425 !important;
                border-radius: 25px !important;
                padding: 0.8rem 3rem !important;
                font-weight: 700 !important;
                margin-left: 1rem ;
            }
                
            div.stButton > button:hover {
                border:1px #dfffb6 solid;
                background-color: #f6ae1e !important;
            }
             [data-testid="stFileUploader"] {
                background-color: #113425  !important;
                border: 2px dashed #dfffb6 !important;
                border-radius: 15px;
                padding: 20px;
                }
                [data-testid="stFileUploader"] label {
                color: #dfffb6 !important;
                font-weight: bold;
                }
                [data-testid="stFileUploader"] button {
                background-color: #f6ae1e !important;
                color: #113425 !important;
                border-radius: 10px;
                }
                [data-testid="stFileUploaderFileName"] {
                color: #dfffb6 !important;
                }
                [data-testid="stFileUploaderDropzone"] {
                background-color: #113425 !important;
                border-radius: 10px;
                color: #dfffb6 !important;
                opacity: 1 !important;
                }
                [data-testid="stFileUploaderDropzone"] * {
                color: #dfffb6 !important;
                opacity: 1 !important;
                }
                [data-testid="stFileUploaderDropzone"] small {
                color: #f6ae1e !important;
                }
                
        </style>
    """, unsafe_allow_html=True)

CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 
    'Tomato___healthy'
]

TREATMENTS = {
    "Apple___Apple_scab": "Apply fungicides like Captan or Sulfur starting at green tip stage. Rake and destroy fallen leaves to prevent spores from overwintering in soil. Ensure proper pruning to improve air circulation within the tree canopy.",
    "Apple___Black_rot": "Prune out dead branches and remove 'mummy' fruit from the tree. Apply copper-based fungicides during the dormant season to kill spores. Avoid wounding the fruit during harvest as it creates entry points for rot.",
    "Apple___Cedar_apple_rust": "Remove nearby Juniper/Cedar trees which act as the alternate host for the fungus. Apply fungicides such as Myclobutanil when apple blossoms begin to open. Plant resistant varieties like Liberty or Freedom to avoid future infections.",
    "Cherry_(including_sour)___Powdery_mildew": "Apply horticultural oils or sulfur-based fungicides at the first sign of white spots. Prune the center of the tree to allow maximum sunlight and wind to dry out the leaves. Avoid overhead irrigation which keeps the foliage damp for long periods.",
    "Corn_(maize)___Common_rust_": "Use resistant hybrids specifically rated for rust resistance in your region. Apply foliar fungicides if pustules appear on more than 10% of the leaf surface area. Rotate crops with non-grass species to break the fungal life cycle in the field.",
    "Corn_(maize)___Northern_Leaf_Blight": "Practice deep plowing to bury infected crop residue from the previous season. Apply fungicides early if lesions appear before the silking stage of growth. Manage field drainage to reduce high humidity levels that favor blight development.",
    "Grape___Black_rot": "Ensure meticulous pruning to remove all infected canes and dried berries. Apply Mancozeb or Ziram fungicides from early bloom until the berries turn color. Space vines adequately to ensure rapid leaf drying after rain or heavy dew.",
    "Orange___Haunglongbing_(Citrus_greening)": "Aggressively control Asian Citrus Psyllid populations using systemic insecticides. Immediately remove and burn infected trees to prevent the bacteria from spreading to neighbors. Apply nutritional foliar sprays to boost the immune system of surrounding healthy trees.",
    "Peach___Bacterial_spot": "Avoid high-nitrogen fertilizers which promote succulent growth susceptible to bacteria. Apply zinc-sulfate or copper sprays during the late leaf-fall stage in autumn. Plant resistant peach cultivars and avoid overhead watering to keep fruit dry.",
    "Potato___Early_blight": "Maintain high soil fertility, especially nitrogen, to keep the plant vigorous and resistant. Use protective fungicides like Chlorothalonil or Mancozeb on a 7-10 day schedule. Practice a 3-year crop rotation and remove all volunteer potato plants from the field.",
    "Potato___Late_blight": "Destroy all cull piles and infected tubers as they are primary sources of outbreaks. Use systemic fungicides like Ridomil Gold if weather is consistently cool and wet. Monitor the field daily and harvest early if the disease begins to reach the tubers.",
    "Tomato___Bacterial_spot": "Treat seeds with hot water or dilute bleach before planting to ensure they are pathogen-free. Apply copper-based bactericides combined with Mancozeb every 7 days during wet weather. Keep the garden free of weeds like nightshade that can harbor the bacteria.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Install fine mesh screens in greenhouses and use silver reflective mulches to deter whiteflies. Remove infected plants immediately and place them in sealed bags to kill the insect vectors. Use imidacloprid-based insecticides for severe whitefly infestations.",
    "Tomato___Tomato_mosaic_virus": "Wash hands with soap and water before handling plants as the virus spreads via touch. Disinfect garden tools with a 10% bleach solution between working on different plants. Remove and burn all crop debris at the end of the season to prevent soil carryover.",
    "healthy": "The plant appears healthy and vibrant! Continue regular deep watering at the base of the plant rather than on the leaves. Monitor weekly for early signs of pests like aphids or mites to maintain health.",
    "default": "Diagnosis confirmed. Please consult your local agricultural extension office for region-specific chemical recommendations. Ensure you follow all safety labels when applying any treatment to food crops."
}



@st.cache_resource
def load_plant_model():
    
    try:
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
        ])

        model = keras.Sequential([
            data_augmentation,
            layers.Rescaling(1./255),
            layers.Conv2D(32, (3,3), activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, (3,3), activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(128, (3,3), activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(38, activation='softmax')
        ])

        model.build(input_shape=(None, 96, 96, 3))

        # Try full load first, fall back to weights-only
        try:
            model = tf.keras.models.load_model("plant_disease_model.keras", compile=False)
        except:
            model.load_weights("plant_disease_model.keras")
        
        return model

    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None
    
def model_prediction(test_image):
    model = load_plant_model()
    image = Image.open(test_image).resize((96, 96))
    img_arr = np.expand_dims(np.array(image), axis=0) 
    prediction = model.predict(img_arr)
    return np.argmax(prediction)

if 'user' not in st.session_state:
    st.session_state.user = None  

if 'page' not in st.session_state:
    st.session_state.page = 'Login'

if st.session_state.user is None and st.session_state.page not in ['Login', 'Register']:
    st.session_state.page = 'Login'
    st.stop()

if 'predicted_disease' not in st.session_state:
    st.session_state.predicted_disease = None

def navigate(target):
    st.session_state.page = target
    st.rerun()

def top_navbar():
    st.markdown('<div class="navbar-container">', unsafe_allow_html=True)

    # Adjusted column ratios for tighter text alignment
    cols = st.columns([4, 1.5, 1.5, 1.5, 1.5])

    with cols[1]:
        if st.button("Home"):
            navigate('Home')

    with cols[2]:
        if st.button("Predict"):
            navigate('Prediction')

    with cols[3]:
        if st.button("Treat it"):
            navigate('Treatment')

    with cols[4]:
        if st.button("Logout"):
            st.session_state.user = None
            navigate('Login')

    st.markdown('</div>', unsafe_allow_html=True)

apply_global_styles()
    
# ---------------- LOGIN ----------------
if st.session_state.page == 'Login':    
    l, m, n = st.columns([1,1,1])
    with m:
        st.markdown("<h1 style='color:#dfffb6;text-align: center;'>Login</h1>", unsafe_allow_html=True)
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            st.session_state.user = username    
            navigate('Home')

        if st.button("Register"):
            navigate('Register')

# ---------------- REGISTER ----------------
elif st.session_state.page == 'Register':    
    x, y, z = st.columns([1,1,1])
    with y:
        st.markdown("<h1 style='color:#dfffb6;text-align: center;'>Register</h1>", unsafe_allow_html=True)
        new_user = st.text_input("Create Username")
        new_pass = st.text_input("Create Password", type="password")

        if st.button("Register"):
            st.success("Registered successfully")

        if st.button("Login"):
            navigate('Login')

# ---------------- HOME ----------------
elif st.session_state.page == 'Home':
    top_navbar()
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        col_pad, col_img = st.columns([0.1, 1])

    with col_img:
        st.markdown("<div style='height:10vh;'></div>", unsafe_allow_html=True)
        st.image("Home Page.jpg", width=680)

    
    with right:
        st.markdown("<div style='height:11vh;'></div>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #dfffb6; '>Giving your garden a voice before the first leaf falls.</h4>", unsafe_allow_html=True)
        st.markdown("<h1 style='color: #dfffb6; font-size: 4rem; line-height: 1;'>AI Crop Disease<br>Diagnosis</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #dfffb6; font-size: 1.2rem; opacity: 0.9; '>We use advanced Computer Vision to help you identify 38 different plant diseases instantly. Simply Snap a photo, upload it and get an instant diagnosis.</p>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([0.05, 1, 1])
        with col2:
            if st.button("Scan Now"):
                navigate('Prediction')
        with col3:
            if st.button("Treatment"):
                navigate('Treatment')

# ---------------- PREDICTION ----------------
elif st.session_state.page == 'Prediction':
    top_navbar()
    if 'model_ready' not in st.session_state:
        with st.spinner("Loading AI model... "):
            load_plant_model()
        st.session_state.model_ready = True
    uploaded_file = None
    left, right = st.columns(2)

    with left:
        st.markdown("<div style='height:10vh;'></div>", unsafe_allow_html=True)
        st.markdown("<h1 style='color: #dfffb6; text-align: center;'>Let's Find Your Plant's Disease</h1>", unsafe_allow_html=True)
        l, m, n = st.columns([0.5, 3, 0.5])
        with m:
            uploaded_file = st.file_uploader("Upload leaf photo", type=["jpg","png","jpeg"])
            if st.session_state.predicted_disease:
                st.success(f"Result: {st.session_state.predicted_disease.replace('___', ' ')}")
                if st.button("Treatment"):
                    navigate('Treatment')
            
    with right:
        p, q, r = st.columns([0.5, 3, 0.5])
        with q:
            st.markdown("<div style='height:10vh;'></div>", unsafe_allow_html=True)
            if uploaded_file is None:
                st.markdown("""
                            <div style="
                                height: 330px;
                                width: 400px;
                                border: 2px dashed #dfffb6;
                                border-radius: 15px;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                color: #dfffb6;
                                font-size: 1.2rem;
                                text-align: center;
                                margin-bottom: 15px; 
                            ">
                                Uploaded image will appear here
                            </div>
                            """, unsafe_allow_html=True)

            else:
                st.image(uploaded_file, width=350)

                if st.button("Analyze Image"):
                        idx = model_prediction(uploaded_file)
                        st.session_state.predicted_disease = CLASS_NAMES[idx]

# ---------------- TREATMENT ----------------
elif st.session_state.page == 'Treatment':
    top_navbar()
    st.markdown("<style>.stApp { background-color: #113425; }</style>", unsafe_allow_html=True)

    t_left, t_right = st.columns([1, 1])

    with t_left:
        l, m, n = st.columns([0.5, 3, 0.5])
        with m:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("<h1 style='color: #dfffb6; '>Plant Treatment Plan</h1>", unsafe_allow_html=True)

            default_idx = 0
            if st.session_state.predicted_disease in CLASS_NAMES:
                default_idx = CLASS_NAMES.index(st.session_state.predicted_disease)

            selection = st.selectbox("Current Selected Disease:", options=CLASS_NAMES, index=default_idx)
            st.info("Note : This tool is designed to assist in early detection only. Always consult a Professional for critical crop decisions ")
    

    with t_right:
        st.markdown("<div style='height:5vh;'></div>", unsafe_allow_html=True)
        l, m, n = st.columns([0.2, 3.8, 0.2])
        with m:
            advice = TREATMENTS.get(selection, TREATMENTS["default"])
            if "healthy" in selection.lower():
                advice = TREATMENTS["healthy"]

            st.markdown(
                f"""
                 <div style='margin-bottom: 20px;background: #113425; border: 1px dashed #dfffb6; padding: 40px; border-radius: 10px; box-shadow: 5px 5px 15px rgba(0,0,0,0.05);'>
                    <h3 style='font-size: 1.6rem;color: #dfffb6;'>The Treatment Plan for {selection.replace("___", " ")} is :</h3>
                    <hr>
                    <p style='font-size: 1.2rem; line-height: 1.6;'>{advice}</p>
                </div>
                """,unsafe_allow_html=True)
