import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Lama Studi Mahasiswa",
    page_icon="🎓",
    layout="wide"
)

# Judul aplikasi
st.title("🎓 Prediksi Lama Studi Mahasiswa")
st.markdown("Aplikasi ini menggunakan CNN + Fuzzy Logic untuk memprediksi kategori lama studi mahasiswa")

# Inisialisasi session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.model = None
    st.session_state.scaler_seq = None
    st.session_state.scaler_fuzzy = None
    st.session_state.label_encoder_classes = None

# Fungsi untuk setup fuzzy system
@st.cache_resource
def setup_fuzzy_system():
    # Definisikan universe variabel fuzzy
    fluktuasi = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'fluktuasi')
    sks_tidak_lulus = ctrl.Antecedent(np.arange(0, 16, 1), 'sks_tidak_lulus')
    kehadiran = ctrl.Antecedent(np.arange(0, 101, 1), 'kehadiran')
    tugas = ctrl.Antecedent(np.arange(0, 101, 1), 'tugas')
    risiko = ctrl.Consequent(np.arange(0, 101, 1), 'risiko')
    
    # Fungsi keanggotaan
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
    
    # Aturan fuzzy
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

# Fungsi mapping untuk fuzzy
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

# Fungsi untuk menghitung risiko fuzzy
def calculate_fuzzy_risk(fluktuasi_val, sks_val, kehadiran_val, tugas_val, risk_sim):
    risk_sim.input['fluktuasi'] = map_fluktuasi(fluktuasi_val)
    risk_sim.input['sks_tidak_lulus'] = map_sks_tidak_lulus(sks_val)
    risk_sim.input['kehadiran'] = map_kehadiran(kehadiran_val)
    risk_sim.input['tugas'] = map_tugas(tugas_val)
    risk_sim.compute()
    return risk_sim.output['risiko']

# Load model dan preprocessing tools
@st.cache_resource
def load_model_and_tools():
    try:
        # Load model
        model = load_model('best_model.h5')
        
        # Buat scaler dummy (seharusnya load dari file pickle jika ada)
        scaler_seq = StandardScaler()
        scaler_fuzzy = StandardScaler()
        
        # Label classes (sesuaikan dengan dataset Anda)
        label_classes = ['Lulus Cepat', 'Tepat Waktu', 'Terlambat/DO']
        
        return model, scaler_seq, scaler_fuzzy, label_classes
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None

# Sidebar untuk upload model
st.sidebar.title("📁 Model Management")

# Check if model file exists
if os.path.exists('best_model.h5'):
    if st.sidebar.button("Load Model"):
        with st.spinner("Loading model..."):
            model, scaler_seq, scaler_fuzzy, label_classes = load_model_and_tools()
            if model is not None:
                st.session_state.model = model
                st.session_state.scaler_seq = scaler_seq
                st.session_state.scaler_fuzzy = scaler_fuzzy
                st.session_state.label_encoder_classes = label_classes
                st.session_state.model_loaded = True
                st.sidebar.success("Model loaded successfully!")
            else:
                st.sidebar.error("Failed to load model!")
else:
    st.sidebar.error("Model file 'best_model.h5' not found!")
    uploaded_file = st.sidebar.file_uploader("Upload your trained model (.h5)", type=['h5'])
    if uploaded_file is not None:
        with open('best_model.h5', 'wb') as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success("Model uploaded successfully! Click 'Load Model' to use it.")

# Setup fuzzy system
risk_sim = setup_fuzzy_system()

# Main interface
if st.session_state.model_loaded:
    st.success("✅ Model ready for prediction!")
    
    # Input form
    st.header("📝 Input Data Mahasiswa")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Akademik per Semester")
        
        # Input IP dan data per semester
        ip_data = []
        mk_ulang_data = []
        sks_data = []
        
        for i in range(1, 7):
            st.write(f"**Semester {i}:**")
            ip = st.number_input(f"IP Semester {i}", min_value=0.0, max_value=4.0, value=3.0, step=0.01, key=f"ip_{i}")
            mk_ulang = st.number_input(f"MK Ulang Semester {i}", min_value=0, max_value=20, value=0, key=f"mk_{i}")
            sks = st.number_input(f"Total SKS Selesai Semester {i}", min_value=0, max_value=30, value=20, key=f"sks_{i}")
            
            ip_data.append(ip)
            mk_ulang_data.append(mk_ulang)
            sks_data.append(sks)
    
    with col2:
        st.subheader("Data Perilaku dan Karakteristik")
        
        fluktuasi_ips = st.selectbox("Fluktuasi IPS", ['Stabil', 'Fluktuatif', 'Menurun'])
        sks_tidak_lulus = st.selectbox("SKS Tidak Lulus", ['Sedikit', 'Sedang', 'Banyak'])
        kehadiran = st.selectbox("Kehadiran", ['Rendah', 'Sedang', 'Tinggi'])
        tugas = st.selectbox("Tugas", ['Sering telat', 'Tepat waktu'])
    
    # Tombol prediksi
    if st.button("🔮 Prediksi Lama Studi", type="primary"):
        try:
            # Hitung risiko fuzzy
            risiko_fuzzy = calculate_fuzzy_risk(fluktuasi_ips, sks_tidak_lulus, kehadiran, tugas, risk_sim)
            
            # Persiapan data sequence untuk CNN
            seq_data = []
            for i in range(6):
                seq_data.extend([ip_data[i], mk_ulang_data[i], sks_data[i]])
            
            seq_features = np.array(seq_data).reshape(1, 6, 3)
            
            # Normalize data (menggunakan fit_transform dummy untuk demo)
            # Dalam implementasi nyata, gunakan scaler yang sudah difit dari training data
            seq_features_scaled = seq_features  # Simplified untuk demo
            risiko_scaled = np.array([[risiko_fuzzy / 100]])  # Normalize to 0-1
            
            # Prediksi
            prediction = st.session_state.model.predict({
                'input_seq': seq_features_scaled,
                'input_fuzzy': risiko_scaled
            })
            
            predicted_class = np.argmax(prediction[0])
            predicted_prob = prediction[0][predicted_class]
            predicted_label = st.session_state.label_encoder_classes[predicted_class]
            
            # Tampilkan hasil
            st.header("📊 Hasil Prediksi")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Risiko Fuzzy", f"{risiko_fuzzy:.2f}")
            
            with col2:
                st.metric("Prediksi Kategori", predicted_label)
            
            with col3:
                st.metric("Confidence", f"{predicted_prob*100:.1f}%")
            
            # Visualisasi probabilitas
            st.subheader("Distribusi Probabilitas")
            prob_df = pd.DataFrame({
                'Kategori': st.session_state.label_encoder_classes,
                'Probabilitas': prediction[0]
            })
            
            st.bar_chart(prob_df.set_index('Kategori'))
            
            # Interpretasi hasil
            st.subheader("💡 Interpretasi Hasil")
            
            if predicted_label == "Lulus Cepat":
                st.success("🎉 Mahasiswa ini memiliki potensi untuk lulus lebih cepat dari waktu normal!")
            elif predicted_label == "Tepat Waktu":
                st.info("✅ Mahasiswa ini diprediksi akan lulus tepat waktu.")
            else:
                st.warning("⚠️ Mahasiswa ini berisiko mengalami keterlambatan studi atau dropout. Perlu perhatian khusus!")
            
            # Rekomendasi
            st.subheader("📋 Rekomendasi")
            
            if risiko_fuzzy > 60:
                st.write("**Rekomendasi Tinggi:**")
                st.write("- Lakukan konseling akademik intensif")
                st.write("- Berikan bimbingan khusus untuk mata kuliah yang sulit")
                st.write("- Monitor kehadiran dan tugas lebih ketat")
                st.write("- Pertimbangkan program remedial")
            elif risiko_fuzzy > 40:
                st.write("**Rekomendasi Sedang:**")
                st.write("- Lakukan monitoring berkala")
                st.write("- Berikan motivasi dan dukungan akademik")
                st.write("- Tingkatkan kualitas belajar")
            else:
                st.write("**Rekomendasi Rendah:**")
                st.write("- Pertahankan performa akademik")
                st.write("- Dorong untuk berpartisipasi dalam kegiatan akademik lainnya")
                st.write("- Jadikan contoh untuk mahasiswa lain")
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.error("Pastikan model dan data input sudah benar!")

else:
    st.info("📋 Silakan load model terlebih dahulu menggunakan sidebar di sebelah kiri.")
    
    # Tampilkan informasi tentang aplikasi
    st.header("ℹ️ Tentang Aplikasi")
    st.write("""
    Aplikasi ini menggunakan kombinasi CNN (Convolutional Neural Network) dan Fuzzy Logic untuk memprediksi 
    kategori lama studi mahasiswa berdasarkan:
    
    **Input Data:**
    - Data akademik per semester (IP, MK Ulang, SKS Selesai)
    - Karakteristik perilaku (Fluktuasi IPS, SKS Tidak Lulus, Kehadiran, Tugas)
    
    **Output:**
    - Prediksi kategori: Lulus Cepat, Tepat Waktu, atau Terlambat/DO
    - Tingkat kepercayaan prediksi
    - Rekomendasi tindakan
    """)

# Footer
st.markdown("---")
st.markdown("**Dikembangkan menggunakan Streamlit + TensorFlow + Scikit-Fuzzy**")