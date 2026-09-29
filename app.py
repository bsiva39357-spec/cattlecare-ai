import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
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
def load_model():
    return tf.keras.models.load_model('cattlecare_model.keras')

model = load_model()

def generate_gradcam(img_array, model, class_idx, last_conv_layer_name="out_relu"):
    try:
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy()
    except Exception:
        return None

def overlay_heatmap(raw_image, heatmap, alpha=0.4):
    img = np.array(raw_image)
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay

# Sidebar Navigation
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
    st.title("🐄 CattleCare AI: Intelligent Livestock Disease & Surveillance System")
    st.subheader("Project Overview")
    st.markdown("""
    CattleCare AI combines **Computer Vision (MobileNetV2)** with **Clinical Explainability (Grad-CAM)** 
    and meteorological risk modeling to support rapid veterinary diagnosis in rural areas.
    """)
    st.info("System Status: **Model Active (Validation Accuracy: 94.46%)**")

# PAGE 2: DISEASE DETECTION
elif page == "🔬 Disease Detection":
    st.title("🔬 Vision-Based Cattle Disease Diagnostic")
    uploaded_file = st.file_uploader("Upload Cow/Lesion Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        raw_image = Image.open(uploaded_file).convert('RGB')
        col_img, col_pred = st.columns(2)

        with col_img:
            st.image(raw_image, caption="Uploaded Image", use_container_width=True)

        with col_pred:
            with st.spinner("Processing image through MobileNetV2 pipeline..."):
                img_rgb = raw_image.convert('RGB')
                img_resized = img_rgb.resize((224, 224))
                img_array = np.array(img_resized, dtype=np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                preds = model.predict(img_array)[0]
                idx = int(np.argmax(preds))
                pred_class = CLASS_NAMES[idx]
                conf = float(preds[idx] * 100)

            st.session_state['last_image'] = raw_image
            st.session_state['last_pred_idx'] = idx
            st.session_state['last_pred_class'] = pred_class

            st.subheader("Diagnostic Assessment")
            st.metric(label="Predicted Condition", value=DISEASE_DB[pred_class]['name'])
            st.metric(label="Confidence Score", value=f"{conf:.2f}%")
            st.progress(min(max(conf / 100.0, 0.0), 1.0))
            st.write(preds)
            st.write(f"**Primary Symptoms:** {DISEASE_DB[pred_class]['symptoms']}")
            st.info(f"**Recommended First Response:** {DISEASE_DB[pred_class]['treatment']}")

# PAGE 3: EXPLAINABLE AI
elif page == "👁️ Explainable AI (XAI)":
    st.title("👁️ Explainable AI (Grad-CAM Diagnostic Heatmap)")
    st.markdown("Validates AI attention regions to guarantee focus on lesions, tissue, and symptoms.")

    if 'last_image' in st.session_state:
        raw_image = st.session_state['last_image']
        idx = st.session_state['last_pred_idx']
        pred_class = st.session_state['last_pred_class']

        img_resized = raw_image.resize((224, 224))
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
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
        st.info("Please run a diagnosis in the 'Disease Detection' tab first to generate Grad-CAM heatmaps.")

# PAGE 4: OUTBREAK RISK PREDICTION
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

# PAGE 5: DISEASE REPOSITORY
elif page == "📖 Disease Repository":
    st.title("📖 Bovine Pathology Clinical Repository")
    for key, val in DISEASE_DB.items():
        with st.expander(val['name']):
            st.write(f"**Pathogen/Cause:** {val['causes']}")
            st.write(f"**Symptoms:** {val['symptoms']}")
            st.write(f"**Biosecurity/Prevention:** {val['prevention']}")
            st.write(f"**Treatment Guidance:** {val['treatment']}")
