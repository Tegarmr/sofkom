import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Student Study Duration Predictor",
    page_icon="🎓",
    layout="wide"
)

# Title and description
st.title("🎓 Student Study Duration Predictor")
st.markdown("### Predict student graduation timeline using CNN + Fuzzy Logic")

# Initialize fuzzy system (same as in your notebook)
@st.cache_resource
def initialize_fuzzy_system():
    # Define universe variables
    fluktuasi = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'fluktuasi')
    sks_tidak_lulus = ctrl.Antecedent(np.arange(0, 16, 1), 'sks_tidak_lulus')
    kehadiran = ctrl.Antecedent(np.arange(0, 101, 1), 'kehadiran')
    tugas = ctrl.Antecedent(np.arange(0, 101, 1), 'tugas')
    risiko = ctrl.Consequent(np.arange(0, 101, 1), 'risiko')
    
    # Membership functions
    fluktuasi['stabil'] = fuzz.trimf(fluktuasi.universe, [0, 0, 0.3])
    fluktuasi['fluktuatif'] = fuzz.trimf(fluktuasi.universe, [0.2, 0.5, 0.7])
    fluktuasi['menurun'] = fuzz.trimf(fluktuasi.universe, [0.6, 1, 1])
    
    sks_tidak_lulus['sedikit'] = fuzz.trimf(sks_tidak_lulus.universe, [0, 0, 4])
    sks_tidak_lulus['sedang'] = fuzz.trimf(sks_tidak_lulus.universe, [2, 6, 10])
    sks_tidak_lulus['banyak'] = fuzz.trimf(sks_tidak_lulus.universe, [8, 15, 15])
    
    kehadiran['rendah'] = fuzz.trimf(kehadiran.universe, [0, 0, 60])
    kehadiran['sedang'] = fuzz.trimf(kehadiran.universe, [50, 70, 85])
    kehadiran['tinggi'] = fuzz.trimf(kehadiran.universe, [80, 100, 100])
    
    tugas['sering_telat'] = fuzz.trimf(tugas.universe, [0, 0, 50])
    tugas['tepat_waktu'] = fuzz.trimf(tugas.universe, [40, 100, 100])
    
    risiko['rendah'] = fuzz.trimf(risiko.universe, [0, 0, 35])
    risiko['sedang'] = fuzz.trimf(risiko.universe, [30, 50, 70])
    risiko['tinggi'] = fuzz.trimf(risiko.universe, [60, 100, 100])
    
    # Rules
    rule1 = ctrl.Rule(
        fluktuasi['stabil'] & sks_tidak_lulus['sedikit'] & kehadiran['tinggi'] & tugas['tepat_waktu'],
        risiko['rendah']
    )
    rule2 = ctrl.Rule(
        fluktuasi['fluktuatif'] & sks_tidak_lulus['sedang'] & (kehadiran['sedang'] | kehadiran['tinggi']) & tugas['tepat_waktu'],
        risiko['sedang']
    )
    rule3 = ctrl.Rule(
        (fluktuasi['menurun'] | fluktuasi['fluktuatif']) &
        (sks_tidak_lulus['sedang'] | sks_tidak_lulus['banyak']) &
        (kehadiran['rendah'] | kehadiran['sedang']) &
        tugas['sering_telat'],
        risiko['tinggi']
    )
    rule4 = ctrl.Rule(
        fluktuasi['menurun'] & sks_tidak_lulus['banyak'] & kehadiran['rendah'] & tugas['sering_telat'],
        risiko['tinggi']
    )
    rule5 = ctrl.Rule(fluktuasi['stabil'], risiko['rendah'])
    rule6 = ctrl.Rule(fluktuasi['fluktuatif'], risiko['sedang'])
    rule7 = ctrl.Rule(fluktuasi['menurun'], risiko['tinggi'])
    
    risk_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])
    risk_sim = ctrl.ControlSystemSimulation(risk_ctrl)
    
    return risk_sim

# Mapping functions
def map_fluktuasi(value):
    mapping = {'Stabil': 0.1, 'Fluktuatif': 0.5, 'Menurun': 0.9}
    return mapping.get(value, 0.5)

def map_sks_tidak_lulus(value):
    mapping = {'Sedikit': 2, 'Sedang': 6, 'Banyak': 12}
    return mapping.get(value, 6)

def map_kehadiran(value):
    mapping = {'Rendah': 30, 'Sedang': 70, 'Tinggi': 90}
    return mapping.get(value, 70)

def map_tugas(value):
    mapping = {'Sering telat': 20, 'Tepat waktu': 80}
    return mapping.get(value, 50)

# Calculate fuzzy risk
def calculate_fuzzy_risk(risk_sim, fluktuasi_val, sks_val, kehadiran_val, tugas_val):
    try:
        risk_sim.input['fluktuasi'] = fluktuasi_val
        risk_sim.input['sks_tidak_lulus'] = sks_val
        risk_sim.input['kehadiran'] = kehadiran_val
        risk_sim.input['tugas'] = tugas_val
        risk_sim.compute()
        return risk_sim.output['risiko']
    except Exception as e:
        st.error(f"Error in fuzzy calculation: {e}")
        return 50.0  # Default value

# Main app
def main():
    # Initialize fuzzy system
    risk_sim = initialize_fuzzy_system()
    
    # File upload for model
    st.sidebar.header("📁 Model File")
    model_file = st.sidebar.file_uploader("Upload your .h5 model file", type=['h5'])
    
    if model_file is not None:
        # Save uploaded file temporarily
        with open("temp_model.h5", "wb") as f:
            f.write(model_file.getbuffer())
        
        try:
            # Load model
            model = load_model("temp_model.h5")
            st.sidebar.success("✅ Model loaded successfully!")
            
            # Input form
            st.header("📊 Input Student Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Academic Performance (IP per Semester)")
                ip_sem = []
                for i in range(1, 7):
                    ip = st.number_input(f"IP Semester {i}", min_value=0.0, max_value=4.0, value=3.0, step=0.1, key=f"ip_{i}")
                    ip_sem.append(ip)
                
                st.subheader("Failed Courses (MK Ulang per Semester)")
                mk_ulang = []
                for i in range(1, 7):
                    mk = st.number_input(f"MK Ulang Semester {i}", min_value=0, max_value=10, value=0, key=f"mk_{i}")
                    mk_ulang.append(mk)
            
            with col2:
                st.subheader("Completed Credits (SKS per Semester)")
                sks_selesai = []
                for i in range(1, 7):
                    sks = st.number_input(f"Total SKS Semester {i}", min_value=0, max_value=30, value=20, key=f"sks_{i}")
                    sks_selesai.append(sks)
                
                st.subheader("Behavioral Factors")
                fluktuasi_ips = st.selectbox("Fluktuasi IPS", ['Stabil', 'Fluktuatif', 'Menurun'])
                sks_tidak_lulus = st.selectbox("SKS Tidak Lulus", ['Sedikit', 'Sedang', 'Banyak'])
                kehadiran = st.selectbox("Kehadiran", ['Rendah', 'Sedang', 'Tinggi'])
                tugas = st.selectbox("Tugas", ['Sering telat', 'Tepat waktu'])
            
            # Prediction button
            if st.button("🔮 Predict Study Duration", type="primary"):
                try:
                    # Prepare sequential features (same structure as training)
                    seq_features = []
                    for i in range(6):
                        seq_features.extend([ip_sem[i], mk_ulang[i], sks_selesai[i]])
                    
                    seq_features = np.array(seq_features).reshape(1, 6, 3)
                    
                    # Standard scaling (you might want to save and load the actual scalers from training)
                    # For now, using basic normalization
                    seq_features_scaled = (seq_features - seq_features.mean()) / (seq_features.std() + 1e-8)
                    
                    # Calculate fuzzy risk
                    fluktuasi_val = map_fluktuasi(fluktuasi_ips)
                    sks_val = map_sks_tidak_lulus(sks_tidak_lulus)
                    kehadiran_val = map_kehadiran(kehadiran)
                    tugas_val = map_tugas(tugas)
                    
                    risiko_fuzzy = calculate_fuzzy_risk(risk_sim, fluktuasi_val, sks_val, kehadiran_val, tugas_val)
                    risiko_scaled = np.array([[risiko_fuzzy]])
                    
                    # Make prediction
                    prediction_probs = model.predict({
                        'input_seq': seq_features_scaled,
                        'input_fuzzy': risiko_scaled
                    })
                    
                    # Assuming the classes are ['Lulus Cepat', 'Tepat Waktu', 'Terlambat/DO']
                    classes = ['Lulus Cepat', 'Tepat Waktu', 'Terlambat/DO']
                    predicted_class = classes[np.argmax(prediction_probs[0])]
                    confidence = np.max(prediction_probs[0]) * 100
                    
                    # Display results
                    st.header("🎯 Prediction Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Predicted Category", predicted_class)
                    
                    with col2:
                        st.metric("Confidence", f"{confidence:.1f}%")
                    
                    with col3:
                        st.metric("Fuzzy Risk Score", f"{risiko_fuzzy:.1f}")
                    
                    # Probability breakdown
                    st.subheader("📈 Probability Breakdown")
                    prob_df = pd.DataFrame({
                        'Category': classes,
                        'Probability': prediction_probs[0] * 100
                    })
                    
                    st.bar_chart(prob_df.set_index('Category'))
                    
                    # Risk interpretation
                    st.subheader("🔍 Risk Analysis")
                    if risiko_fuzzy < 40:
                        risk_level = "Low Risk"
                        risk_color = "green"
                        risk_desc = "Student shows good academic performance and behavior patterns."
                    elif risiko_fuzzy < 60:
                        risk_level = "Medium Risk"
                        risk_color = "orange"
                        risk_desc = "Student needs attention and support to stay on track."
                    else:
                        risk_level = "High Risk"
                        risk_color = "red"
                        risk_desc = "Student requires immediate intervention and support."
                    
                    st.markdown(f"**Risk Level:** :{risk_color}[{risk_level}]")
                    st.write(risk_desc)
                    
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
                    st.error("Please check your model file and input data.")
        
        except Exception as e:
            st.sidebar.error(f"❌ Error loading model: {e}")
            st.sidebar.error("Please ensure you uploaded a valid .h5 model file.")
    
    else:
        st.info("👆 Please upload your trained .h5 model file in the sidebar to start making predictions.")
        
        # Show sample data format
        st.header("📋 Expected Input Format")
        st.write("Your model expects the following inputs:")
        
        sample_data = {
            "Sequential Features (6 semesters)": ["IP_Semester_1-6", "MK_Ulang_Semester_1-6", "Total_SKS_Selesai_Semester_1-6"],
            "Behavioral Factors": ["Fluktuasi_IPS", "SKS_Tidak_Lulus", "Kehadiran", "Tugas"],
            "Output Classes": ["Lulus Cepat", "Tepat Waktu", "Terlambat/DO"]
        }
        
        for key, value in sample_data.items():
            st.write(f"**{key}:** {', '.join(value)}")

if __name__ == "__main__":
    main()