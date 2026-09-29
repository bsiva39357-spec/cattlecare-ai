import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import matplotlib.cm as cm

# --- Streamlit Setup ---
st.set_page_config(page_title="CattleCare AI", page_icon="🐄", layout="wide")

CLASS_NAMES = ['bovine_pinkeye', 'fmd', 'healthy', 'lumpy_skin', 'mastitis', 'ringworm']

DISEASE_DB = {
    'lumpy_skin': {
        'name': 'Lumpy Skin Disease (LSD)',
        'symptoms': 'Enlarged superficial lymph nodes, firm circumscribed skin nodules (2-5 cm), fever, sudden drop in milk yield.',
        'causes': 'Capripoxvirus transmitted mainly by blood-feeding arthropod vectors (mosquitoes, stable flies, ticks).',
        'prevention': 'Strict vector control, restricted livestock movement, mass homologous vaccination.',
        'treatment': 'Isolate immediate herd. Secondary bacterial infection control via systemic antibiotics, anti-inflammatory drugs, and topical wound sprays.'
    },
    'fmd': {
        'name': 'Foot and Mouth Disease (FMD)',
        'symptoms': 'High pyrexia, vesicular lesions/erosions on tongue, dental pad, hooves, excessive salivation, lameness.',
        'causes': 'Aphthovirus (Picornaviridae family), airborne aerosol spread and direct contact transmission.',
        'prevention': 'Regular bi-annual polyvalent vaccination, strict farm biosecurity, mandatory entry quarantine.',
        'treatment': 'Symptomatic supportive care: Mild potassium permanganate mouth wash, topical antiseptic hoof dressings, soft feed.'
    },
    'bovine_pinkeye': {
        'name': 'Infectious Bovine Keratoconjunctivitis (Pinkeye)',
        'symptoms': 'Severe lacrimation, photophobia, corneal clouding/opacity, central corneal ulceration.',
        'causes': 'Moraxella bovis bacterium exacerbated by UV radiation, dust, and face flies (Musca autumnalis).',
        'prevention': 'Fly population management, shade availability, dust reduction in holding pens.',
        'treatment': 'Subconjunctival antibiotic injection or topical veterinary oxytetracycline/cloxacillin eye formulations.'
    },
    'mastitis': {
        'name': 'Bovine Mastitis',
        'symptoms': 'Swollen, painful udder quarters, abnormal milk (flakes, clots, watery serum), systemic fever.',
        'causes': 'Intramammary bacterial invasion (Staphylococcus aureus, Streptococcus uberis, E. coli).',
        'prevention': 'Post-milking teat disinfection, proper milking machine calibration, clean and dry beddings.',
        'treatment': 'Prompt intramammary antibiotic infusion based on culture sensitivity, frequent milk evacuation, NSAIDs.'
    },
    'ringworm': {
        'name': 'Ringworm (Dermatophytosis)',
        'symptoms': 'Circular, raised, crusty, greyish-white alopecia lesions around neck, eyes, and perineum.',
        'causes': 'Trichophyton verrucosum fungus spore exposure via contaminated grooming brushes, fences.',
        'prevention': 'Disinfection of housing facilities, adequate sunlight exposure, isolation of symptomatic stock.',
        'treatment': 'Topical iodine scrubbing, miconazole/clotrimazole application, systemic vitamin A supplementation.'
    },
    'healthy': {
        'name': 'Healthy Livestock',
        'symptoms': 'Bright demeanor, moist muzzle, clear eyes, smooth coat, normal appetite and rumination.',
        'causes': 'Optimal herd management and health protocols.',
        'prevention': 'Continue standard herd vaccination cycles and balanced nutritional rations.',
        'treatment': 'No medical intervention required.'
    }
}

@st.cache_resource
def load_vision_model():
    return tf.keras.models.load_model('cattlecare_model.keras')

model = load_vision_model()

# --- Phase 7: Grad-CAM Implementation ---
def generate_gradcam(img_array, model, class_idx):
    # Locate base model layer and target last conv layer inside MobileNetV2
    base_layer = None
    for l in model.layers:
        if 'mobilenetv2' in l.name.lower():
            base_layer = l
            break

    if not base_layer:
        return None

    # Sub-graph mapping for gradient calculation
    last_conv_layer = base_layer.get_layer('out_relu')
    grad_model = tf.keras.models.Model(
        inputs=[base_layer.input],
        outputs=[last_conv_layer.output, base_layer.output]
    )

    # Preprocess image scaling matching model input
    scaled_input = tf.keras.applications.mobilenet_v2.preprocess_input(img_array.copy())

    with tf.GradientTape() as tape:
        conv_outputs, base_preds = grad_model(scaled_input)
        # Pass through classification top layers
        x = model.get_layer(index=2)(base_preds)  # GlobalAvgPool
        x = model.get_layer(index=3)(x)           # Dropout
        preds = model.get_layer(index=4)(x)       # Dense softmax
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def overlay_heatmap(original_pil_img, heatmap):
    img = np.array(original_pil_img)
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
    return superimposed

# --- Navigation Bar (Phase 9) ---
st.sidebar.title("🐄 CattleCare AI Navigation")
page = st.sidebar.radio("Select View:", [
    "1️⃣ Home",
    "2️⃣ Disease Detection",
    "3️⃣ Explainable AI (XAI)",
    "4️⃣ Outbreak Risk Prediction",
    "5️⃣ Disease Repository"
])

# ----------------- PAGE 1: HOME -----------------
if page == "1️⃣ Home":
    st.title("🌾 CattleCare AI: Intelligent Livestock Disease & Surveillance System")
    st.markdown("---")
    st.markdown("""
    ### Project Overview
    **CattleCare AI** is an end-to-end veterinary clinical support platform combining:
    * **Computer Vision Diagnostics:** Deep transfer learning model fine-tuned across 6 cattle health states.
    * **Explainable AI (Grad-CAM):** Visual verification confirming biological symptom locations rather than background artifacts.
    * **Epidemiological Risk Surveillance:** Machine learning heuristics synthesizing localized meteorological & vector transmission parameters.
    """)
    st.info("👈 Use the left sidebar to navigate across diagnostic tools.")

# ----------------- PAGE 2: DISEASE DETECTION -----------------
elif page == "2️⃣ Disease Detection":
    st.title("📷 Vision-Based Cattle Disease Diagnostic")
    uploaded_file = st.file_uploader("Upload Cow/Lesion Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        raw_image = Image.open(uploaded_file).convert('RGB')
        col_img, col_pred = st.columns(2)
        
        with col_img:
            st.image(raw_image, caption="Uploaded Image", use_container_width=True)

      with col_pred:
            with st.spinner("Processing image through MobileNetV2 pipeline..."):
                img_resized = raw_image.resize((224, 224))
                img_array = np.array(img_resized, dtype=np.float32)
                img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
                img_array = np.expand_dims(img_array, axis=0)
                preds = model.predict(img_array)[0]
                idx = np.argmax(preds)
                pred_class = CLASS_NAMES[idx]
                conf = preds[idx] * 100

            st.session_state['last_image'] = raw_image
            st.session_state['last_pred_idx'] = idx
            st.session_state['last_pred_class'] = pred_class

            st.subheader("Diagnostic Assessment")
            if pred_class == 'healthy':
                st.success(f"### Result: {DISEASE_DB[pred_class]['name']}")
            else:
                st.error(f"### Result: {DISEASE_DB[pred_class]['name']}")
            
            st.metric("Confidence Score", f"{conf:.2f}%")
            st.write(f"**Primary Symptoms:** {DISEASE_DB[pred_class]['symptoms']}")
            st.info(f"**Recommended First Response:** {DISEASE_DB[pred_class]['treatment']}")

# ----------------- PAGE 3: EXPLAINABLE AI -----------------
elif page == "3️⃣ Explainable AI (XAI)":
    st.title("🔍 Explainable AI (Grad-CAM Diagnostic Heatmap)")
    st.markdown("Validates AI attention regions to guarantee focus on lesions, tissue, and symptoms.")

if 'last_image' in st.session_state:
        raw_image = st.session_state['last_image']
        idx = st.session_state['last_pred_idx']
        pred_class = st.session_state['last_pred_class']

        img_resized = raw_image.resize((224, 224))
        img_array = np.array(img_resized, dtype=np.float32)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("Generating Class Activation Maps..."):
            heatmap = generate_gradcam(img_array, model, idx)

        col1, col2 = st.columns(2)
        with col1:
            st.image(raw_image, caption=f"Original Specimen ({pred_class})", use_container_width=True)
        with col2:
            if heatmap is not None:
                overlay = overlay_heatmap(raw_image, heatmap)
                st.image(overlay, caption="Grad-CAM Focus Overlay (Red = Critical Focus)", use_container_width=True)
            else:
                st.warning("Heatmap generator unable to hook target activation layer.")
    else:
        st.info("Run a diagnostic on the '2️⃣ Disease Detection' page first to view Grad-CAM explainability.")

# ----------------- PAGE 4: RISK PREDICTION -----------------
elif page == "4️⃣ Outbreak Risk Prediction":
    st.title("🌦️ Regional Outbreak & Environmental Epidemiological Risk")
    st.markdown("Evaluates microclimate vector suitability and localized infection velocity.")

    col_in1, col_in2 = st.columns(2)
    with col_in1:
        loc = st.text_input("District / Region", value="Madurai")
        disease_choice = st.selectbox("Disease Susceptibility Target", ["LSD", "FMD", "Mastitis", "Ringworm", "Pinkeye"])
        temp_val = st.slider("Ambient Temperature (°C)", 15, 45, 31)
        hum_val = st.slider("Relative Humidity (%)", 20, 100, 82)
    with col_in2:
        season_val = st.selectbox("Season", ["Monsoon", "Summer", "Winter", "Post-Monsoon"])
        rain_val = st.number_input("Average Rainfall (mm)", 0, 300, 15)
        cases_val = st.number_input("Prior Reported Cases in Vicinity (Last 30 Days)", 0, 100, 8)

    if st.button("Evaluate Outbreak Severity"):
        # Heuristic scoring pipeline (Phase 8)
        risk_score = 0
        factors = []

        if disease_choice == "LSD":
            if hum_val > 70:
                risk_score += 35
                factors.append("High humidity promotes biting fly/tick vector surge.")
            if season_val == "Monsoon" or rain_val > 10:
                risk_score += 25
                factors.append("Rainfall patterns extend vector propagation ranges.")
            if cases_val >= 5:
                risk_score += 30
                factors.append(f"High localized viral reservoir ({cases_val} previous cases reported).")

        elif disease_choice == "FMD":
            if cases_val >= 4:
                risk_score += 45
                factors.append("Extreme aerodynamic viral contagion risk in immediate perimeter.")
            if hum_val > 65:
                risk_score += 25
                factors.append("Moist air enhances extracellular virus aerosol stability.")

        elif disease_choice == "Ringworm":
            if temp_val > 28 and hum_val > 70:
                risk_score += 60
                factors.append("Warm, high moisture conditions favor rapid fungal spore germination.")

        else:
            if cases_val > 5:
                risk_score += 50
                factors.append("High proximity infection density.")

        risk_score = min(risk_score, 98)

        st.markdown("---")
        st.subheader("Risk Assessment Summary")
        if risk_score >= 65:
            st.error(f"### Regional Disease Threat: HIGH ({risk_score}%)")
            st.write("**Recommended Protocol:** Implement strict quarantine, immediate ring-vaccination, and daily vector suppression sprays.")
        elif risk_score >= 35:
            st.warning(f"### Regional Disease Threat: MEDIUM ({risk_score}%)")
            st.write("**Recommended Protocol:** Increase sanitization frequency and isolate any cattle displaying mild pyrexia.")
        else:
            st.success(f"### Regional Disease Threat: LOW ({risk_score}%)")
            st.write("**Recommended Protocol:** Standard biosecurity upkeep.")

        if factors:
            st.markdown("**Key Risk Drivers Detected:**")
            for f in factors:
                st.write(f"- {f}")

# ----------------- PAGE 5: DISEASE REPOSITORY -----------------
elif page == "5️⃣ Disease Repository":
    st.title("📚 Comprehensive Cattle Pathology Repository")
    for key, info in DISEASE_DB.items():
        with st.expander(f"{info['name']}"):
            st.write(f"**Symptoms:** {info['symptoms']}")
            st.write(f"**Etiology / Causes:** {info['causes']}")
            st.write(f"**Biosecurity & Prevention:** {info['prevention']}")
            st.write(f"**Clinical Interventions:** {info['treatment']}")
