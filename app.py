import pickle
import streamlit as st
import numpy as np
import pandas as pd
from anfis_model import ANFIS, ANFISBagging

# Load model dan statistik
@st.cache_resource
def load_model():
    with open('anfis_bagging_model_skenario1-11.pkl', 'rb') as f:
        model = pickle.load(f)
    
    stats = np.load('normalization_params.npz')
    return model, stats

bagging, stats = load_model()
feature_names = stats['feature_names']
original_mean = stats['mean'][:4]
original_std = stats['std'][:4]
target_mean = stats['mean'][4]
target_std = stats['std'][4]

# Fungsi denormalisasi
def denormalize(y_norm):
    return y_norm * target_std + target_mean

# Fungsi normalisasi
def normalize_features(X, original_mean, original_std):
    return (X - original_mean) / original_std

# ==================== TAMPILAN STREAMLIT ====================
st.title('🌧️ Prediksi Curah Hujan')

# Hanya menyediakan input manual
st.header("📝 Input Data Manual")
num_days = st.slider("Jumlah Hari yang Akan Diprediksi", 1, 30, 1)

input_data = []
for day in range(num_days):
    st.markdown(f"### Hari {day+1}")
    cols = st.columns(4)
    with cols[0]:
        tavg = st.number_input(f"Tavg (°C)", key=f"tavg_{day}")
    with cols[1]:
        rh_avg = st.number_input(f"RH_avg (%)", key=f"rh_avg_{day}")
    with cols[2]:
        ss = st.number_input(f"ss (jam)", key=f"ss_{day}")
    with cols[3]:
        ff_avg = st.number_input(f"ff_avg (km/jam)", key=f"ff_avg_{day}")
    input_data.append([tavg, rh_avg, ss, ff_avg])

if st.button("Proses Prediksi"):
    # Konversi ke array
    X_input = np.array(input_data)
    
    # Normalisasi
    X_norm = normalize_features(
        X_input,
        original_mean,
        original_std
    )
    
    # Prediksi
    y_norm = bagging.predict(X_norm)
    predictions = denormalize(y_norm)
    
    # Buat DataFrame hasil
    result_df = pd.DataFrame({
        'Hari': [f"Hari {i+1}" for i in range(num_days)],
        'Tavg (°C)': [data[0] for data in input_data],
        'RH_avg (%)': [data[1] for data in input_data],
        'ss (jam)': [data[2] for data in input_data],
        'ff_avg (km/jam)': [data[3] for data in input_data],
        'Prediksi Curah Hujan (mm)': np.round(predictions, 2)
    })
    
    # Tampilkan hasil
    st.subheader("Hasil Prediksi")
    st.dataframe(result_df)
    
    # Grafik prediksi
    #st.subheader("Grafik Prediksi Curah Hujan")
    #st.bar_chart(result_df.set_index('Hari')['Prediksi Curah Hujan (mm)'])