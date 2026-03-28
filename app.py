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

            html, body {
                margin: 0;
                padding: 0;
                background-color: #113425 !important;
                font-family: 'Lexend', sans-serif;
            }

            [data-testid="stAppViewContainer"],
            [data-testid="stAppViewContainer"] > div,
            section.main {
                background-color: #113425 !important;
            }

            /* Center + constrain content width */
            .block-container {
                background-color: #113425 !important;
                max-width: 1100px !important;
                margin: 0 auto !important;
                padding: 20px 32px !important;
            }

            [data-testid="stSidebar"] { display: none; }
            header, footer { visibility: hidden; }

            h1, h2, h3, h4 {
                font-family: 'Lexend', sans-serif;
                color: #dfffb6;
            }
            p { font-family: 'Lexend', sans-serif; color: #dfffb6; }

            /* ---- NAVBAR buttons ---- */
            .navbar-container div.stButton > button {
                background: none !important;
                border: none !important;
                color: #dfffb6 !important;
                font-size: clamp(13px, 2vw, 18px) !important;
                font-family: 'Lexend' !important;
                font-weight: 400 !important;
                padding: 0 !important;
                margin: 0 !important;
                box-shadow: none !important;
                transition: color 0.3s ease;
                width: auto !important;
                min-height: 0 !important;
                height: auto !important;
            }
            .navbar-container div.stButton > button:hover {
                color: #f6ae1e !important;
                background: none !important;
                text-decoration: underline !important;
            }
            .navbar-container div.stButton > button:active {
                background: none !important;
                color: #f6ae1e !important;
            }

            /* ---- Inputs ---- */
            input[type="text"], input[type="password"] {
                border: 1px solid #dfffb6 !important;
                border-radius: 10px !important;
                padding: 10px !important;
                color: #113425 !important;
                width: 100% !important;
                box-sizing: border-box !important;
            }
            input[type="text"]:focus, input[type="password"]:focus {
                border: 1px solid #f6ae1e !important;
                outline: none !important;
                box-shadow: 0 0 5px #f6ae1e;
            }

            /* ---- Buttons ---- */
            div.stButton > button {
                background-color: #f6ae1e !important;
                color: #113425 !important;
                border-radius: 25px !important;
                padding: 0.7rem 2rem !important;
                font-weight: 700 !important;
                font-family: 'Lexend' !important;
                width: 100% !important;
                margin-top: 8px !important;
                margin-left: 0 !important;
            }
            div.stButton > button:hover {
                border: 1px #dfffb6 solid;
                background-color: #f6ae1e !important;
            }

            /* ---- File uploader ---- */
            [data-testid="stFileUploader"] {
                background-color: #113425 !important;
                border: 2px dashed #dfffb6 !important;
                border-radius: 15px;
                padding: 20px;
            }
            [data-testid="stFileUploader"] label { color: #dfffb6 !important; font-weight: bold; }
            [data-testid="stFileUploader"] button {
                background-color: #f6ae1e !important;
                color: #113425 !important;
                border-radius: 10px;
            }
            [data-testid="stFileUploaderFileName"] { color: #dfffb6 !important; }
            [data-testid="stFileUploaderDropzone"] {
                background-color: #113425 !important;
                border-radius: 10px;
                color: #dfffb6 !important;
            }
            [data-testid="stFileUploaderDropzone"] * { color: #dfffb6 !important; }
            [data-testid="stFileUploaderDropzone"] small { color: #f6ae1e !important; }

            /* ---- Selectbox ---- */
            [data-testid="stSelectbox"] label { color: #dfffb6 !important; }

            /* ---- MOBILE ---- */
            @media (max-width: 768px) {
                .block-container { padding: 12px 12px !important; }

                [data-testid="stHorizontalBlock"] { flex-direction: column !important; }
                [data-testid="stHorizontalBlock"] > div {
                    width: 100% !important;
                    min-width: 100% !important;
                    flex: 1 1 100% !important;
                }

                /* Navbar stays as a row */
                .navbar-container [data-testid="stHorizontalBlock"] {
                    flex-direction: row !important;
                    flex-wrap: wrap !important;
                    justify-content: center !important;
                    gap: 8px !important;
                }
                .navbar-container [data-testid="stHorizontalBlock"] > div {
                    width: auto !important;
                    min-width: auto !important;
                    flex: 0 0 auto !important;
                }

                h1 { font-size: 1.8rem !important; }
                h4 { font-size: 1rem !important; }
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
    cols = st.columns([4, 1.5, 1.5, 1.5, 1.5])
    with cols[1]:
        if st.button("Home"):    navigate('Home')
    with cols[2]:
        if st.button("Predict"): navigate('Prediction')
    with cols[3]:
        if st.button("Treat it"): navigate('Treatment')
    with cols[4]:
        if st.button("Logout"):
            st.session_state.user = None
            navigate('Login')
    st.markdown('</div>', unsafe_allow_html=True)


apply_global_styles()

# ─────────────────────── LOGIN ───────────────────────
if st.session_state.page == 'Login':
    st.markdown("<div style='height:6vh'></div>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.markdown("<h1 style='color:#dfffb6; text-align:center;'>Login</h1>", unsafe_allow_html=True)
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("Login"):
            st.session_state.user = username
            navigate('Home')
        if st.button("Register"):
            navigate('Register')

# ─────────────────────── REGISTER ───────────────────────
elif st.session_state.page == 'Register':
    st.markdown("<div style='height:6vh'></div>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.markdown("<h1 style='color:#dfffb6; text-align:center;'>Register</h1>", unsafe_allow_html=True)
        new_user = st.text_input("Create Username")
        new_pass = st.text_input("Create Password", type="password")
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("Register"):
            st.success("Registered successfully!")
        if st.button("Back to Login"):
            navigate('Login')

# ─────────────────────── HOME ───────────────────────
elif st.session_state.page == 'Home':
    top_navbar()
    st.markdown("<div style='height:3vh'></div>", unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.image("Home Page.jpg", use_container_width=True)
    with right:
        st.markdown("<div style='height:2vh'></div>", unsafe_allow_html=True)
        st.markdown(
            "<h4 style='color:#dfffb6; font-size:clamp(0.9rem,1.5vw,1.1rem);'>"
            "Giving your garden a voice before the first leaf falls.</h4>",
            unsafe_allow_html=True)
        st.markdown(
            "<h1 style='color:#dfffb6; font-size:clamp(2rem,4vw,3.5rem); line-height:1.15;'>"
            "AI Crop Disease<br>Diagnosis</h1>",
            unsafe_allow_html=True)
        st.markdown(
            "<p style='color:#dfffb6; font-size:clamp(0.85rem,1.5vw,1.05rem); opacity:0.9; margin-bottom:24px;'>"
            "We use advanced Computer Vision to help you identify 38 different plant diseases instantly. "
            "Simply snap a photo, upload it and get an instant diagnosis.</p>",
            unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Scan Now"):  navigate('Prediction')
        with c2:
            if st.button("Treatment"): navigate('Treatment')

# ─────────────────────── PREDICTION ───────────────────────
elif st.session_state.page == 'Prediction':
    top_navbar()

    if 'model_ready' not in st.session_state:
            load_plant_model()
    st.session_state.model_ready = True

    st.markdown("<div style='height:2vh'></div>", unsafe_allow_html=True)
    st.markdown(
        "<h1 style='color:#dfffb6; text-align:center; font-size:clamp(1.4rem,3vw,2.2rem);'>"
        "Let's Find Your Plant's Disease</h1>",
        unsafe_allow_html=True)
    st.markdown("<div style='height:1vh'></div>", unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")

    with left:
        uploaded_file = st.file_uploader("Upload leaf photo", type=["jpg", "png", "jpeg"])
        if st.session_state.predicted_disease:
            st.success(f"Result: {st.session_state.predicted_disease.replace('___', ' ')}")
            if st.button("View Treatment"):
                navigate('Treatment')

    with right:
        if uploaded_file is None:
            st.markdown("""
                <div style="
                    min-height: 260px;
                    width: 100%;
                    border: 2px dashed #dfffb6;
                    border-radius: 15px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: #dfffb6;
                    font-size: 1.1rem;
                    text-align: center;
                    padding: 20px;
                    box-sizing: border-box;
                ">Uploaded image will appear here</div>
            """, unsafe_allow_html=True)
        else:
            # Centered, fixed-size image preview
            st.markdown(
                "<div style='display:flex; justify-content:center;'>",
                unsafe_allow_html=True)
            st.image(uploaded_file, width=250)
            st.markdown("</div>", unsafe_allow_html=True)

            if st.button("Analyze Image"):
                with st.spinner("Analyzing..."):
                    idx = model_prediction(uploaded_file)
                    st.session_state.predicted_disease = CLASS_NAMES[idx]

# ─────────────────────── TREATMENT ───────────────────────
elif st.session_state.page == 'Treatment':
    top_navbar()
    st.markdown("<div style='height:2vh'></div>", unsafe_allow_html=True)

    left, right = st.columns([1, 1.4], gap="large")

    with left:
        st.markdown("<h1 style='color:#dfffb6;'>Plant Treatment Plan</h1>", unsafe_allow_html=True)
        default_idx = 0
        if st.session_state.predicted_disease in CLASS_NAMES:
            default_idx = CLASS_NAMES.index(st.session_state.predicted_disease)
        selection = st.selectbox("Current Selected Disease:", options=CLASS_NAMES, index=default_idx)
        st.info("Note: This tool is designed to assist in early detection only. Always consult a Professional for critical crop decisions.")

    with right:
        advice = TREATMENTS.get(selection, TREATMENTS["default"])
        if "healthy" in selection.lower():
            advice = TREATMENTS["healthy"]
        st.markdown(
            f"""
            <div style='background:#113425; border:1px dashed #dfffb6; padding:28px;
                        border-radius:12px; box-shadow:5px 5px 15px rgba(0,0,0,0.1); margin-top:8px;'>
                <h3 style='font-size:clamp(1rem,2vw,1.4rem); color:#dfffb6; margin-top:0;'>
                    Treatment Plan for <span style="color:#f6ae1e;">{selection.replace("___", " ")}</span>:
                </h3>
                <hr style='border-color:#dfffb6; opacity:0.4;'>
                <p style='font-size:clamp(0.85rem,1.5vw,1.05rem); line-height:1.7; margin-bottom:0;'>{advice}</p>
            </div>
            """, unsafe_allow_html=True)
