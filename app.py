import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Air Quality Jogja",
    layout="wide" # Menggunakan layout lebar agar lebih lega
)

# --- 2. LOAD MODEL ---
@st.cache_resource # Agar model tidak di-load berulang kali (biar cepat)
def load_model():
    try:
        return joblib.load('model_kualitas_udara.pkl')
    except FileNotFoundError:
        return None

model = load_model()

if model is None:
    st.error("File 'model_kualitas_udara.pkl' tidak ditemukan. Pastikan file ada di folder yang sama.")
    st.stop()

# --- 3. SIDEBAR (Input & Edukasi) ---
st.sidebar.title("Panel Kontrol")
st.sidebar.write("Masukkan nilai konsentrasi polutan:")

def user_input_features():
    PM10 = st.sidebar.slider('PM10 (Partikel Debu)', 0, 150, 20)
    SO2  = st.sidebar.slider('SO2 (Sulfur Dioksida)', 0, 150, 15)
    CO   = st.sidebar.slider('CO (Karbon Monoksida)', 0, 150, 10)
    O3   = st.sidebar.slider('O3 (Ozon)', 0, 150, 30)
    NO2  = st.sidebar.slider('NO2 (Nitrogen Dioksida)', 0, 150, 5)
    
    data = {'PM10': [PM10], 'SO2': [SO2], 'CO': [CO], 'O3': [O3], 'NO2': [NO2]}
    return pd.DataFrame(data)

input_df = user_input_features()

# Fitur Edukasi di Sidebar (Expander)
st.sidebar.markdown("---")
with st.sidebar.expander("Apa arti istilah ini?"):
    st.write("""
    - **PM10:** Partikel debu halus (asap kendaraan, debu jalan).
    - **SO2:** Gas buang industri/kendaraan diesel.
    - **CO:** Gas beracun dari pembakaran tidak sempurna (knalpot).
    - **O3:** Ozon permukaan (reaksi sinar matahari & polusi).
    - **NO2:** Gas dari pembakaran bahan bakar fosil.
    """)

# --- 4. MAIN PAGE ---
st.title("Prediksi Kualitas Udara Yogyakarta")
st.markdown("Aplikasi berbasis **Machine Learning** untuk memantau kesehatan udara.")
st.markdown("---")

# Layout Kolom (Kiri: Hasil, Kanan: Grafik)
col1, col2 = st.columns([2, 3]) # Kolom kanan lebih lebar sedikit

with col1:
    st.subheader("Hasil Prediksi")
    
    if st.button("Analisis Sekarang", use_container_width=True):
        # Prediksi Kategori
        prediction = model.predict(input_df)[0]
        # Prediksi Probabilitas (Keyakinan Model)
        prediction_proba = model.predict_proba(input_df)
        
        # Mapping Probabilitas ke Label
        classes = model.classes_
        proba_df = pd.DataFrame(prediction_proba, columns=classes)
        
        # Tampilkan Hasil Utama
        if prediction == 'Good':
            st.success(f"### {prediction} (Baik)")
            st.info("**Saran:** Udara sangat segar! Sangat baik untuk olahraga outdoor, bersepeda, atau jalan-jalan.")
        elif prediction == 'Moderate':
            st.warning(f"### {prediction} (Sedang)")
            st.info("**Saran:** Kelompok sensitif (asma, lansia, anak-anak) sebaiknya kurangi aktivitas berat di luar.")
        else:
            st.error(f"### {prediction} (Tidak Sehat)")
            st.info("**Saran:** WAJIB pakai masker. Tutup ventilasi rumah. Hindari keluar jika tidak mendesak.")

        # Simpan data untuk grafik di kolom sebelah
        st.session_state['proba_df'] = proba_df
        st.session_state['predicted'] = True
        st.session_state['prediction'] = prediction

with col2:
    st.subheader("Analisis Detail")
    
    # Cek apakah tombol sudah ditekan
    if 'predicted' in st.session_state:
        proba_df = st.session_state['proba_df']
        
        # Membuat Grafik Batang dengan Plotly
        # Transpose dataframe agar mudah di-plot
        proba_melted = proba_df.melt(var_name='Kategori', value_name='Probabilitas')
        
        fig = go.Figure(data=[go.Bar(
            x=proba_melted['Kategori'],
            y=proba_melted['Probabilitas'],
            marker_color=['#2ecc71', '#f1c40f', '#e74c3c'] # Hijau, Kuning, Merah
        )])
        
        fig.update_layout(
            title="Seberapa Yakin Model?",
            yaxis_title="Tingkat Keyakinan (0-1)",
            xaxis_title="Kategori",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Menampilkan Data Input User dalam Tabel Kecil
        st.write("Data yang Anda Masukkan:")
        st.dataframe(input_df, hide_index=True)

# Footer
st.markdown("---")

st.caption("Dikembangkan dengan Random Forest Algorithm | Data: Pollutant Standards Index Jogja 2020")


