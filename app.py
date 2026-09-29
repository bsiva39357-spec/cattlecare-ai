import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page Configuration
st.set_page_config(page_title="CattleCare AI", page_icon="🐄", layout="wide")

CLASS_NAMES = ['bovine_pinkeye', 'fmd', 'healthy', 'lumpy_skin', 'mastitis', 'ringworm']

DISEASE_DB = {
    'lumpy_skin': {
        'name': 'Lumpy Skin Disease (LSD)',
        'symptoms': 'Enlarged superficial lymph nodes, firm circumscribed skin nodules (2-5 cm), fever, sudden drop in milk yield.',
        'causes': 'Capripoxvirus transmitted mainly by blood-feeding arthropod vectors (mosquitoes, stable flies, ticks).',
        'prevention': 'Strict vector control, restricted livestock movement, mass homologous vaccination.',
        'treatment': 'Isolate immediate herd. Secondary bacterial infection control via systemic antibiotics and topical antiseptics.'
    },
    'fmd': {
        'name': 'Foot and Mouth Disease (FMD)',
        'symptoms': 'High pyrexia, vesicular lesions/erosions on tongue, dental pad, hooves, excessive salivation, lameness.',
        'causes': 'Aphthovirus (Picornaviridae family), airborne aerosol spread and direct contact transmission.',
        'prevention': 'Regular bi-annual polyvalent vaccination, strict farm biosecurity, mandatory entry quarantine.',
        'treatment': 'Symptomatic supportive care: Mild potassium permanganate mouth wash, antiseptic hoof dressings.'
    },
    'bovine_pinkeye': {
        'name': 'Infectious Bovine Keratoconjunctivitis (Pinkeye)',
        'symptoms': 'Severe lacrimation, photophobia, corneal cloudiness, central ulceration, conjunctival redness.',
        'causes': 'Moraxella bovis bacterium amplified by high UV radiation and face flies.',
        'prevention': 'Face fly population mitigation, shade structures against solar radiation, dust suppression.',
        'treatment': 'Subconjunctival antibiotic injections, topical antibiotic eye sprays, eye patches.'
    },
    'mastitis': {
        'name': 'Bovine Clinical Mastitis',
        'symptoms': 'Inflamed, swollen, painful quarters; abnormal milk secretion (curds, clots, discoloration); fever.',
        'causes': 'Bacterial pathogens (Staphylococcus aureus, Streptococcus uberis, E. coli) entering via teat canal.',
        'prevention': 'Proper milking procedures, post-milking teat dipping, hygienic dry cow management.',
        'treatment': 'Intramammary antibiotic infusion accompanied by veterinary anti-inflammatory therapy.'
    },
    'ringworm': {
        'name': 'Bovine Dermatophytosis (Ringworm)',
        'symptoms': 'Circular, raised, crusty, alopecic gray-white lesions predominantly localized on face, neck, and perineum.',
        'causes': 'Fungal etiology (predominantly Trichophyton verrucosum); spore transmission via surfaces.',
        'prevention': 'Direct exposure to sunlight, stall ventilation, chemical surface disinfection.',
        'treatment': 'Topical scrubbing with 2-5% povidone-iodine, enilconazole wash, or topical clotrimazole application.'
    },
    'healthy': {
        'name': 'Healthy Bovine Specimen',
        'symptoms': 'Normal appetite, clear ocular clarity, supple skin coat without lesions, active rumination.',
        'causes': 'Optimal herd biosecurity, comprehensive nutrition, regular vaccination cadence.',
        'prevention': 'Sustain scheduled preventative immunization and clean housing standards.',
        'treatment': 'No clinical intervention needed. Continue proactive preventive health schedules.'
    }
}

@st.cache_resource
def load_cattle_model():
    return tf.keras.models.load_model('cattlecare_model.keras')

try:
    model = load_cattle_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading model: {e}")

# Sidebar
st.sidebar.title("🐄 CattleCare AI Navigation")
page = st.sidebar.radio(
    "Select View:",
    [
        "🏠 Home",
        "🔬 Disease Detection",
        "📊 Outbreak Risk Prediction",
        "📖 Disease Repository"
    ]
)

if page == "🏠 Home":
    st.title("🐄 CattleCare AI: Intelligent Livestock Disease & Surveillance System")
    st.markdown("""
    CattleCare AI uses Deep Learning (MobileNetV2) to deliver fast and reliable veterinary screening for cattle conditions.
    """)
    if model_loaded:
        st.success("✅ Model Loaded and Operational!")
    else:
        st.warning("⚠️ Model file not found or failed to initialize.")

elif page == "🔬 Disease Detection":
    st.title("🔬 Vision-Based Cattle Disease Diagnostic")
    uploaded_file = st.file_uploader("Upload Cow/Lesion Image", type=["jpg", "jpeg", "png"])

    if uploaded_file and model_loaded:
        raw_image = Image.open(uploaded_file).convert('RGB')
        col_img, col_pred = st.columns(2)

        with col_img:
            st.image(raw_image, caption="Uploaded Image", use_container_width=True)

        with col_pred:
            with st.spinner("Analyzing image..."):
                img_resized = raw_image.resize((224, 224))
                img_array = np.array(img_resized, dtype=np.float32)

                # Process through MobileNetV2 native preprocessing
                processed_input = tf.keras.applications.mobilenet_v2.preprocess_input(img_array.copy())
                processed_input = np.expand_dims(processed_input, axis=0)

                preds = model.predict(processed_input, verbose=0)[0]
                idx = int(np.argmax(preds))
                pred_class = CLASS_NAMES[idx]
                conf = float(preds[idx] * 100)

            st.subheader("Diagnostic Assessment")
            st.metric(label="Predicted Condition", value=DISEASE_DB[pred_class]['name'])
            st.metric(label="Confidence Score", value=f"{conf:.2f}%")
            st.progress(min(max(conf / 100.0, 0.0), 1.0))

            with st.expander("📊 View All Disease Confidence Scores"):
                for name, prob in zip(CLASS_NAMES, preds):
                    st.write(f"- **{DISEASE_DB[name]['name']}**: `{prob * 100:.2f}%`")

            st.write(f"**Primary Symptoms:** {DISEASE_DB[pred_class]['symptoms']}")
            st.info(f"**Recommended First Response:** {DISEASE_DB[pred_class]['treatment']}")

elif page == "📊 Outbreak Risk Prediction":
    st.title("📊 Regional Outbreak & Surveillance Forecaster")
    temp = st.slider("Ambient Temperature (°C):", 15, 45, 30)
    humidity = st.slider("Relative Humidity (%):", 20, 100, 75)
    reported_cases = st.number_input("Existing Cluster Reports (Last 14 Days):", 0, 50, 5)

    risk_score = (temp * 0.3) + (humidity * 0.4) + (reported_cases * 2.0)
    st.subheader("Surveillance Risk Evaluation")
    if risk_score > 60:
        st.error(f"High Epidemic Risk (Score: {risk_score:.1f}) - Vector intervention required.")
    elif risk_score > 35:
        st.warning(f"Moderate Outbreak Susceptibility (Score: {risk_score:.1f}) - Monitor sanitation.")
    else:
        st.success(f"Low Epidemiological Risk (Score: {risk_score:.1f}) - Routine conditions.")

elif page == "📖 Disease Repository":
    st.title("📖 Bovine Pathology Clinical Repository")
    for key, val in DISEASE_DB.items():
        with st.expander(val['name']):
            st.write(f"**Pathogen/Cause:** {val['causes']}")
            st.write(f"**Symptoms:** {val['symptoms']}")
            st.write(f"**Biosecurity/Prevention:** {val['prevention']}")
            st.write(f"**Treatment Guidance:** {val['treatment']}")
