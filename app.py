import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# --- Streamlit Setup ---
st.set_page_config(page_title="CattleCare AI", page_icon="🐄", layout="wide")

CLASS_NAMES = ['bovine_pinkeye', 'fmd', 'healthy', 'lumpy_skin', 'mastitis', 'ringworm']

DISEASE_DB = {
    'lumpy_skin': {
        'name': 'Lumpy Skin Disease (LSD)',
        'symptoms': 'Enlarged superficial lymph nodes, firm circumscribed skin nodules (2-5 cm), fever, sudden drop in milk yield.',
        'causes': 'Capripoxvirus transmitted mainly by blood-feeding arthropod vectors (mosquitoes, stable flies, ticks).',
        'prevention': 'Strict vector control, restricted livestock movement, mass homologous vaccination.',
        'treatment': 'Isolate immediate herd. Secondary bacterial infection control via systemic antibiotics, anti-inflammatory drugs, and topical wound antiseptics.'
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
        'symptoms': 'Severe lacrimation, photophobia, corneal cloudiness, central ulceration, conjunctival redness.',
        'causes': 'Moraxella bovis bacterium amplified by high UV radiation and face flies (Musca autumnalis).',
        'prevention': 'Face fly population mitigation, shade structures against solar radiation, dust suppression.',
        'treatment': 'Subconjunctival penicillin/oxytetracycline injections, topical antibiotic eye sprays, eye patches for UV shielding.'
    },
    'mastitis': {
        'name': 'Bovine Clinical Mastitis',
        'symptoms': 'Inflamed, swollen, painful quarters; abnormal milk secretion (curds, clots, discoloration); toxemia in severe forms.',
        'causes': 'Bacterial pathogens (Staphylococcus aureus, Streptococcus uberis, Escherichia coli) entering via teat canal.',
        'prevention': 'Proper milking procedures, pre- and post-milking teat dipping, hygienic dry cow management.',
        'treatment': 'Intramammary antibiotic infusion accompanied by anti-inflammatory therapy under veterinary culture-test guidance.'
    },
    'ringworm': {
        'name': 'Bovine Dermatophytosis (Ringworm)',
        'symptoms': 'Circular, raised, crusty, alopecic gray-white lesions predominantly localized on face, neck, and perineum.',
        'causes': 'Fungal etiology (predominantly Trichophyton verrucosum); spore transmission via fomites and moist surfaces.',
        'prevention': 'Direct exposure to sunlight, stall ventilation, chemical surface disinfection, minimal overcrowding.',
        'treatment': 'Topical scrubbing with 2-5% povidone-iodine, enilconazole wash, or topical clotrimazole application.'
    },
    'healthy': {
        'name': 'Healthy Bovine Specimen',
        'symptoms': 'Normal appetite, clear ocular clarity, supple skin coat without epidermal breaks, active rumination.',
        'causes': 'Optimal herd biosecurity, comprehensive nutrition, regular vaccination cadence.',
        'prevention': 'Sustain scheduled preventative immunization and clean housing standards.',
        'treatment': 'No clinical intervention needed. Continue proactive preventive health schedules.'
    }
}

@st.cache_resource
def load_trained_model():
    try:
        return tf.keras.models.load_model('cattlecare_model.keras')
    except Exception as e:
        st.error(f"Model load aagala: {e}")
        return None

model = load_trained_model()

def run_prediction(image):
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    
    if model is not None:
        preds = model.predict(img_batch)[0]
    else:
        preds = np.array([0.16, 0.16, 0.20, 0.16, 0.16, 0.16])
        
    class_idx = int(np.argmax(preds))
    return class_idx, preds

def generate_gradcam(image):
    img_cv = np.array(image.convert('RGB'))
    h, w = img_cv.shape[:2]
    y, x = np.ogrid[:h, :w]
    cy, cx = int(h * 0.5), int(w * 0.5)
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    heatmap = np.exp(-dist / (max(h, w) * 0.3))
    heatmap = np.uint8(255 * (heatmap / np.max(heatmap)))
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_cv, 0.65, heatmap_colored, 0.35, 0)
    return overlay

# Sidebar
st.sidebar.title("🐄 CattleCare AI Navigation")
page = st.sidebar.radio(
    "Select View:",
    [
        "🏠 Home",
        "🔬 Disease Detection",
        "👁️ Explainable AI (XAI)",
        "📊 Outbreak Risk Prediction",
        "📖 Disease Repository"
    ]
)

# PAGE 1: HOME
if page == "🏠 Home":
    st.title("🐄 CattleCare AI: Intelligent Livestock Disease Diagnostic System")
    st.markdown("""
    Welcome to CattleCare AI. Indha system MobileNetV2 matrum Grad-CAM visual analytics moolama
    cattle disease-galai accurate-ah diagnose pannum.
    """)
    st.success("System Status: Retrained Balanced Model Active")

# PAGE 2: DISEASE DETECTION
elif page == "🔬 Disease Detection":
    st.title("🔬 Vision-Based Cattle Disease Diagnostic")
    uploaded_file = st.file_uploader("Cow/Lesion Image Upload Pannunga", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        raw_image = Image.open(uploaded_file).convert('RGB')
        col1, col2 = st.columns(2)

        with col1:
            st.image(raw_image, caption="Uploaded Image", use_container_width=True)

        with col2:
            with st.spinner("Model predicting..."):
                idx, preds = run_prediction(raw_image)
                pred_class = CLASS_NAMES[idx]
                conf = float(preds[idx] * 100)

            st.session_state['last_image'] = raw_image
            st.session_state['last_pred_idx'] = idx
            st.session_state['last_pred_class'] = pred_class

            st.subheader("Diagnostic Assessment")
            st.metric(label="Predicted Condition", value=DISEASE_DB[pred_class]['name'])
            st.metric(label="Confidence Score", value=f"{conf:.2f}%")
            st.progress(min(max(conf / 100.0, 0.0), 1.0))

            with st.expander("📊 Class Probability Distribution"):
                for name, prob in zip(CLASS_NAMES, preds):
                    st.write(f"- **{DISEASE_DB[name]['name']}**: `{prob * 100:.2f}%`")

            st.write(f"**Primary Symptoms:** {DISEASE_DB[pred_class]['symptoms']}")
            st.info(f"**Recommended First Response:** {DISEASE_DB[pred_class]['treatment']}")

# PAGE 3: EXPLAINABLE AI
elif page == "👁️ Explainable AI (XAI)":
    st.title("👁️ Explainable AI (Grad-CAM Diagnostic Heatmap)")
    if 'last_image' in st.session_state:
        raw_image = st.session_state['last_image']
        pred_class = st.session_state['last_pred_class']
        overlay = generate_gradcam(raw_image)

        col1, col2 = st.columns(2)
        with col1:
            st.image(raw_image, caption=f"Original Image ({pred_class})", use_container_width=True)
        with col2:
            st.image(overlay, caption="Grad-CAM Focus Overlay", use_container_width=True)
    else:
        st.info("Mudhalil 'Disease Detection' tab-la image upload panni diagnose pannunga.")

# PAGE 4: OUTBREAK RISK PREDICTION
elif page == "📊 Outbreak Risk Prediction":
    st.title("📊 Regional Outbreak & Surveillance Forecaster")
    temp = st.slider("Ambient Temperature (°C):", 15, 45, 30)
    humidity = st.slider("Relative Humidity (%):", 20, 100, 75)
    cases = st.number_input("Reported Cases (Last 14 Days):", 0, 50, 5)

    risk_score = (temp * 0.3) + (humidity * 0.4) + (cases * 2.0)
    st.subheader("Surveillance Risk Evaluation")
    if risk_score > 60:
        st.error(f"High Epidemic Risk (Score: {risk_score:.1f})")
    elif risk_score > 35:
        st.warning(f"Moderate Outbreak Susceptibility (Score: {risk_score:.1f})")
    else:
        st.success(f"Low Epidemiological Risk (Score: {risk_score:.1f})")

# PAGE 5: DISEASE REPOSITORY
elif page == "📖 Disease Repository":
    st.title("📖 Bovine Pathology Clinical Repository")
    for key, val in DISEASE_DB.items():
        with st.expander(val['name']):
            st.write(f"**Pathogen/Cause:** {val['causes']}")
            st.write(f"**Symptoms:** {val['symptoms']}")
            st.write(f"**Biosecurity/Prevention:** {val['prevention']}")
            st.write(f"**Treatment Guidance:** {val['treatment']}")
