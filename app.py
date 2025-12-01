import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# ================================
# Page Config
# ================================
st.set_page_config(
    page_title="Prediksi Nilai Siswa",
    page_icon="🎓",
    layout="wide"
)

# ================================
# Custom CSS
# ================================
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Sidebar Title */
    [data-testid="stSidebar"] h1 {
        color: white !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 1.5rem !important;
        border-bottom: 2px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Sidebar Text */
    [data-testid="stSidebar"] p {
        color: #e0e0e0 !important;
        font-size: 0.95rem;
    }
    
    /* Radio Button Container */
    [data-testid="stSidebar"] .stRadio {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Radio Button Label */
    [data-testid="stSidebar"] .stRadio label {
        color: white !important;
        font-weight: 600 !important;
        margin-bottom: 0.8rem !important;
    }
    
    /* Radio Button Options */
    [data-testid="stSidebar"] .stRadio > div {
        gap: 0.5rem;
    }
    
    [data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] {
        background: rgba(255, 255, 255, 0.1);
        padding: 0.8rem 1rem;
        border-radius: 8px;
        border: 2px solid transparent;
        transition: all 0.3s ease;
        margin: 0.3rem 0 !important;
    }
    
    [data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:hover {
        background: rgba(255, 255, 255, 0.2);
        border-color: rgba(255, 255, 255, 0.5);
        transform: translateX(5px);
    }
    
    [data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] span {
        color: white !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
    }
    
    /* Selected Radio Button */
    [data-testid="stSidebar"] .stRadio label[data-baseweb="radio"][data-checked="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-color: #ffffff;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        transform: translateX(5px);
    }
    
    /* Main Content */
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.75rem;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# Header
# ================================
st.title("🎓 Prediksi Nilai Akhir Siswa (G3)")
st.markdown("---")
st.info("📊 Aplikasi ini menggunakan 2 algoritma Machine Learning: **Artificial Neural Network (ANN)** & **Random Forest**")

# ================================
# Load Models
# ================================
@st.cache_resource
def load_models():
    ann = load_model("model/model_ann_student_mat.h5", compile=False)
    ann.compile(optimizer='adam', loss='mse', metrics=['mae'])
    rf = joblib.load("model/random_forest_student_mat.pkl")
    scaler_obj = joblib.load("model/scaler.pkl")
    return ann, rf, scaler_obj

# ================================
# Load Models
# ================================
@st.cache_resource
def load_models():
    ann = load_model("model/model_ann_student_mat.h5", compile=False)
    ann.compile(optimizer='adam', loss='mse', metrics=['mae'])
    rf = joblib.load("model/random_forest_student_mat.pkl")
    scaler_obj = joblib.load("model/scaler.pkl")
    return ann, rf, scaler_obj

ann_model, rf_model, scaler = load_models()

# ================================
# Load and Prepare Data for Evaluation
# ================================
@st.cache_data
def load_data():
    df = pd.read_csv("data/student-mat.csv", sep=";")
    df_encoded = pd.get_dummies(df, drop_first=True)
    X = df_encoded.drop("G3", axis=1)
    y = df_encoded["G3"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_data()

# ================================
# ================================
# Sidebar Navigation
# ================================
st.sidebar.markdown("""
<div style='text-align: center; padding: 1rem 0 1.5rem 0;'>
    <h1 style='font-size: 2.5rem; margin: 0;'>🎓</h1>
    <p style='color: white; font-size: 0.9rem; margin: 0.5rem 0 0 0; opacity: 0.9;'>Student Performance Prediction</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("🎯 Menu Navigasi")
st.sidebar.markdown("<p style='color: #e0e0e0; font-size: 0.85rem; margin-bottom: 1rem;'>Pilih menu untuk memulai</p>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Pilih Halaman:",
    ["🏠 Dashboard", "🔢 Prediksi Data Numerik", "✅ Prediksi Data Kategorikal", "📊 Analisis Data Numerik", "📋 Analisis Data Kategorikal", "📈 Evaluasi Model"],
    label_visibility="collapsed"
)

# Sidebar Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; padding: 1rem 0; color: rgba(255, 255, 255, 0.7);'>
    <p style='font-size: 0.8rem; margin: 0;'>Powered by</p>
    <p style='font-size: 0.9rem; font-weight: 600; margin: 0.3rem 0;'>ANN & Random Forest</p>
    <p style='font-size: 0.75rem; margin: 0.5rem 0 0 0;'>© 2024 ML Project</p>
</div>
""", unsafe_allow_html=True)

# ================================
# Page: Dashboard
# ================================
if page == "🏠 Dashboard":
    st.title("🏠 Dashboard - Sistem Prediksi Nilai Siswa")
    st.markdown("---")
    
    # Header
    st.markdown("""
    <div style='padding: 1.5rem; border-radius: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; margin-bottom: 2rem;'>
        <h2 style='color: white; margin: 0;'>📚 Prediksi Nilai Akhir Siswa (G3)</h2>
        <p style='color: white; margin: 0.5rem 0 0 0;'>Menggunakan Artificial Neural Network (ANN) & Random Forest</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Overview
    st.markdown("### 📖 Tentang Sistem")
    st.write("""
    Sistem ini menggunakan **Machine Learning** dan **Deep Learning** untuk memprediksi nilai akhir siswa (G3) berdasarkan berbagai faktor 
    seperti demografi, kondisi sosial ekonomi, dan performa akademik sebelumnya. Dataset yang digunakan adalah 
    **Student Performance Dataset** dengan 41 fitur input.
    
    - **Machine Learning**: Random Forest (Ensemble Learning)
    - **Deep Learning**: Artificial Neural Network (ANN)
    """)
    
    st.markdown("---")
    
    # Algoritma Section
    st.markdown("## 🤖 Cara Kerja Algoritma")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='padding: 1.5rem; border-radius: 10px; background-color: #e3f2fd; height: 100%;'>
            <h3 style='color: #1976d2;'>🧠 Artificial Neural Network (ANN)</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 📋 Arsitektur Model:")
        st.write("""
        - **Input Layer**: 41 fitur (numerik + kategorikal)
        - **Hidden Layer 1**: 128 neuron + ReLU activation + Dropout (0.2)
        - **Hidden Layer 2**: 64 neuron + ReLU activation + Dropout (0.2)
        - **Hidden Layer 3**: 32 neuron + ReLU activation
        - **Output Layer**: 1 neuron (prediksi nilai G3)
        """)
        
        st.markdown("#### ⚙️ Alur Kerja:")
        st.write("""
        1. **Input Processing**: Data dinormalisasi menggunakan StandardScaler
        2. **Forward Propagation**: 
           - Data melewati setiap layer
           - Setiap neuron melakukan operasi: output = activation(weights × input + bias)
        3. **Activation**: Fungsi ReLU untuk non-linearitas
        4. **Regularization**: Dropout untuk mencegah overfitting
        5. **Output**: Nilai prediksi G3
        """)
        
        st.markdown("#### 📊 Training:")
        st.write("""
        - **Loss Function**: Mean Squared Error (MSE)
        - **Optimizer**: Adam
        - **Metrics**: Mean Absolute Error (MAE)
        - **Early Stopping**: Monitoring validation loss
        - **Epochs**: Maksimal 300 dengan patience 15
        """)
        
        st.info("✅ **Keunggulan ANN**: Mampu menangkap pola non-linear yang kompleks dalam data")
    
    with col2:
        st.markdown("""
        <div style='padding: 1.5rem; border-radius: 10px; background-color: #e8f5e9; height: 100%;'>
            <h3 style='color: #388e3c;'>🌲 Random Forest</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 📋 Konfigurasi Model:")
        st.write("""
        - **N Estimators**: 300 pohon keputusan
        - **Max Depth**: None (unlimited)
        - **Task**: Regression
        - **Random State**: 42 (reproducibility)
        """)
        
        st.markdown("#### ⚙️ Alur Kerja:")
        st.write("""
        1. **Bootstrap Sampling**: 
           - Membuat 300 subset data secara random dengan replacement
        2. **Decision Tree Building**:
           - Setiap tree dibangun dengan subset fitur random
           - Split berdasarkan MSE minimization
        3. **Tree Growing**:
           - Nodes di-split hingga pure atau max depth
           - Setiap leaf menyimpan nilai prediksi
        4. **Ensemble Prediction**:
           - Prediksi dari 300 trees di-aggregate
           - Output = rata-rata dari semua predictions
        5. **Feature Importance**: Menghitung kontribusi setiap fitur
        """)
        
        st.markdown("#### 📊 Karakteristik:")
        st.write("""
        - **Bagging**: Mengurangi variance dengan averaging
        - **Feature Randomness**: Decorrelation antar trees
        - **Robust**: Tahan terhadap outliers
        - **No Scaling Required**: Bekerja dengan data asli
        """)
        
        st.success("✅ **Keunggulan Random Forest**: Robust, interpretable, dan handling missing values")
    
    st.markdown("---")
    
    # Comparison
    st.markdown("## 📊 Perbandingan Algoritma")
    
    comparison_data = {
        "Aspek": ["Preprocessing", "Training Time", "Interpretability", "Overfitting Risk", "Feature Scaling", "Handling Non-linear"],
        "ANN": ["Perlu normalisasi", "Lebih lama", "Low (Black box)", "Medium-High", "Required", "Excellent"],
        "Random Forest": ["Minimal", "Medium", "High (Feature importance)", "Low", "Not required", "Good"]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Dataset Info
    st.markdown("## 📁 Informasi Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Sampel", "395")
    with col2:
        st.metric("Jumlah Fitur", "41")
    with col3:
        st.metric("Fitur Numerik", "15")
    with col4:
        st.metric("Fitur Kategorikal", "26")
    
    st.markdown("#### 📝 Fitur Input:")
    
    feature_col1, feature_col2 = st.columns(2)
    
    with feature_col1:
        st.markdown("**🔢 Fitur Numerik:**")
        st.write("""
        - Demografi: age
        - Pendidikan Orang Tua: Medu, Fedu
        - Akademik: studytime, failures, absences, G1, G2
        - Sosial: famrel, freetime, goout, Dalc, Walc
        - Lainnya: traveltime, health
        """)
    
    with feature_col2:
        st.markdown("**✅ Fitur Kategorikal:**")
        st.write("""
        - Sekolah & Demografi: school, sex, address, famsize
        - Keluarga: Pstatus, Mjob, Fjob, guardian
        - Dukungan: schoolsup, famsup, paid, activities
        - Aspirasi: higher, nursery, internet
        - Lainnya: reason, romantic
        """)
    
    st.markdown("---")
    
    # Navigation Guide
    st.markdown("## 🧭 Panduan Penggunaan")
    
    st.write("""
    1. **🔢 Prediksi Data Numerik**: Input hanya data numerik, fitur kategorikal diasumsikan 0
    2. **✅ Prediksi Data Kategorikal**: Input lengkap (numerik + kategorikal) untuk prediksi akurat
    3. **📊 Analisis Data Numerik**: Eksplorasi distribusi dan korelasi fitur numerik
    4. **📋 Analisis Data Kategorikal**: Eksplorasi distribusi dan pengaruh fitur kategorikal
    5. **📈 Evaluasi Model**: Melihat performa model dengan berbagai metrik dan visualisasi
    """)
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background-color: #f5f5f5; border-radius: 10px;'>
        <h4>🎯 Mulai Prediksi Sekarang!</h4>
        <p>Pilih menu di sidebar untuk melakukan prediksi atau analisis data</p>
    </div>
    """, unsafe_allow_html=True)

# ================================
# Page: Prediksi Data Numerik
# ================================
elif page == "🔢 Prediksi Data Numerik":
    st.title("🔢 Prediksi Berdasarkan Data Numerik")
    st.markdown("---")
    st.info("📊 Halaman ini fokus pada prediksi menggunakan fitur numerik dengan asumsi fitur kategorikal = 0")
    
    st.markdown("### 📝 Input Data Numerik Siswa")
    
    numeric_features = {
        'age': ('Umur', 15, 22),
        'Medu': ('Pendidikan Ibu (0-4)', 0, 4),
        'Fedu': ('Pendidikan Ayah (0-4)', 0, 4),
        'traveltime': ('Waktu Perjalanan (1-4)', 1, 4),
        'studytime': ('Waktu Belajar (1-4)', 1, 4),
        'failures': ('Jumlah Kegagalan (0-4)', 0, 4),
        'famrel': ('Hubungan Keluarga (1-5)', 1, 5),
        'freetime': ('Waktu Luang (1-5)', 1, 5),
        'goout': ('Keluar bersama Teman (1-5)', 1, 5),
        'Dalc': ('Konsumsi Alkohol Hari Kerja (1-5)', 1, 5),
        'Walc': ('Konsumsi Alkohol Akhir Pekan (1-5)', 1, 5),
        'health': ('Status Kesehatan (1-5)', 1, 5),
        'absences': ('Jumlah Absen (0-93)', 0, 93),
        'G1': ('Nilai Periode 1 (0-20)', 0, 20),
        'G2': ('Nilai Periode 2 (0-20)', 0, 20)
    }
    
    inputs_numeric = []
    
    cols = st.columns(3)
    for idx, (key, (label, min_val, max_val)) in enumerate(numeric_features.items()):
        with cols[idx % 3]:
            val = st.number_input(label, min_value=min_val, max_value=max_val, value=min_val, step=1, key=f"pred_num_{key}")
            inputs_numeric.append(float(val))
    
    # Add zeros for categorical features (26 features)
    categorical_features_count = 26
    for i in range(categorical_features_count):
        inputs_numeric.append(0.0)
    
    inputs_array = np.array(inputs_numeric).reshape(1, -1)
    
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🤖 Pilih Algoritma Prediksi")
        algo = st.selectbox(
            "Algoritma:",
            ["Artificial Neural Network (ANN)", "Random Forest"],
            label_visibility="collapsed",
            key="numeric_algo"
        )
        
        predict_btn = st.button("🚀 PREDIKSI NILAI", use_container_width=True, key="numeric_predict")
    
    if predict_btn:
        with st.spinner('⏳ Sedang memproses prediksi...'):
            if algo == "Artificial Neural Network (ANN)":
                X_scaled = scaler.transform(inputs_array)
                pred = ann_model.predict(X_scaled, verbose=0)[0][0]
            else:
                pred = rf_model.predict(inputs_array)[0]
        
        st.markdown(f"""
        <div class="prediction-box">
            🎯 Hasil Prediksi Nilai G3<br>
            <span style="font-size: 3rem;">{pred:.2f}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Interpretasi hasil
        if pred >= 16:
            st.success("🌟 **Excellent!** Prediksi menunjukkan nilai sangat baik!")
        elif pred >= 14:
            st.success("✨ **Very Good!** Prediksi menunjukkan nilai baik!")
        elif pred >= 12:
            st.info("👍 **Good!** Prediksi menunjukkan nilai cukup baik!")
        elif pred >= 10:
            st.warning("⚠️ **Fair!** Prediksi menunjukkan nilai cukup, perlu peningkatan!")
        else:
            st.error("🔻 **Needs Improvement!** Prediksi menunjukkan nilai kurang, butuh usaha lebih!")
        
        st.markdown("---")
        st.info("💡 **Catatan:** Prediksi ini menggunakan asumsi semua fitur kategorikal bernilai 0 (default)")

# ================================
# Page: Prediksi Data Kategorikal
# ================================
elif page == "✅ Prediksi Data Kategorikal":
    st.title("✅ Prediksi Berdasarkan Data Lengkap")
    st.markdown("---")
    st.info("📊 Halaman ini memerlukan input data numerik DAN kategorikal untuk prediksi yang akurat")
    
    st.markdown("### 📝 Input Data Siswa")
    
    # Numeric Features
    st.subheader("🔢 Data Numerik")
    
    numeric_features = {
        'age': ('Umur', 15, 22),
        'Medu': ('Pendidikan Ibu (0-4)', 0, 4),
        'Fedu': ('Pendidikan Ayah (0-4)', 0, 4),
        'traveltime': ('Waktu Perjalanan (1-4)', 1, 4),
        'studytime': ('Waktu Belajar (1-4)', 1, 4),
        'failures': ('Jumlah Kegagalan (0-4)', 0, 4),
        'famrel': ('Hubungan Keluarga (1-5)', 1, 5),
        'freetime': ('Waktu Luang (1-5)', 1, 5),
        'goout': ('Keluar bersama Teman (1-5)', 1, 5),
        'Dalc': ('Konsumsi Alkohol Hari Kerja (1-5)', 1, 5),
        'Walc': ('Konsumsi Alkohol Akhir Pekan (1-5)', 1, 5),
        'health': ('Status Kesehatan (1-5)', 1, 5),
        'absences': ('Jumlah Absen (0-93)', 0, 93),
        'G1': ('Nilai Periode 1 (0-20)', 0, 20),
        'G2': ('Nilai Periode 2 (0-20)', 0, 20)
    }
    
    inputs_full = []
    
    cols = st.columns(4)
    for idx, (key, (label, min_val, max_val)) in enumerate(numeric_features.items()):
        with cols[idx % 4]:
            val = st.number_input(label, min_value=min_val, max_value=max_val, value=min_val, step=1, key=f"pred_cat_num_{key}")
            inputs_full.append(float(val))
    
    # Categorical Features
    st.markdown("---")
    st.subheader("✅ Data Kategorikal")
    st.write("**Centang fitur yang sesuai dengan kondisi siswa**")
    
    categorical_features = [
        'school_MS', 'sex_M', 'address_U', 'famsize_LE3', 'Pstatus_T',
        'Mjob_health', 'Mjob_other', 'Mjob_services', 'Mjob_teacher',
        'Fjob_health', 'Fjob_other', 'Fjob_services', 'Fjob_teacher',
        'reason_home', 'reason_other', 'reason_reputation',
        'guardian_mother', 'guardian_other',
        'schoolsup_yes', 'famsup_yes', 'paid_yes', 'activities_yes', 'nursery_yes',
        'higher_yes', 'internet_yes', 'romantic_yes'
    ]
    
    cols = st.columns(4)
    for idx, feature in enumerate(categorical_features):
        with cols[idx % 4]:
            val = st.checkbox(feature.replace('_', ' ').title(), value=False, key=f"pred_cat_{feature}")
            inputs_full.append(1.0 if val else 0.0)
    
    inputs_array = np.array(inputs_full).reshape(1, -1)
    
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🤖 Pilih Algoritma Prediksi")
        algo = st.selectbox(
            "Algoritma:",
            ["Artificial Neural Network (ANN)", "Random Forest"],
            label_visibility="collapsed",
            key="categorical_algo"
        )
        
        predict_btn = st.button("🚀 PREDIKSI NILAI", use_container_width=True, key="categorical_predict")
    
    if predict_btn:
        with st.spinner('⏳ Sedang memproses prediksi...'):
            if algo == "Artificial Neural Network (ANN)":
                X_scaled = scaler.transform(inputs_array)
                pred = ann_model.predict(X_scaled, verbose=0)[0][0]
            else:
                pred = rf_model.predict(inputs_array)[0]
        
        st.markdown(f"""
        <div class="prediction-box">
            🎯 Hasil Prediksi Nilai G3<br>
            <span style="font-size: 3rem;">{pred:.2f}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Interpretasi hasil
        if pred >= 16:
            st.success("🌟 **Excellent!** Prediksi menunjukkan nilai sangat baik!")
        elif pred >= 14:
            st.success("✨ **Very Good!** Prediksi menunjukkan nilai baik!")
        elif pred >= 12:
            st.info("👍 **Good!** Prediksi menunjukkan nilai cukup baik!")
        elif pred >= 10:
            st.warning("⚠️ **Fair!** Prediksi menunjukkan nilai cukup, perlu peningkatan!")
        else:
            st.error("🔻 **Needs Improvement!** Prediksi menunjukkan nilai kurang, butuh usaha lebih!")
        
        st.markdown("---")
        st.success("✅ **Prediksi Lengkap:** Prediksi ini menggunakan semua fitur (numerik + kategorikal)")

# ================================
# Page: Analisis Data Numerik
# ================================
elif page == "📊 Analisis Data Numerik":
    st.title("📊 Analisis Data Numerik")
    st.markdown("---")
    
    # Load raw data
    df = pd.read_csv("data/student-mat.csv", sep=";")
    
    numeric_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 
                    'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']
    
    st.subheader("📋 Statistik Deskriptif")
    st.dataframe(df[numeric_cols].describe(), use_container_width=True)
    
    st.markdown("---")
    st.subheader("📈 Distribusi Data Numerik")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_feature = st.selectbox("Pilih Fitur untuk Visualisasi:", numeric_cols)
    
    with col2:
        chart_type = st.selectbox("Jenis Chart:", ["Histogram", "Box Plot"])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if chart_type == "Histogram":
        ax.hist(df[selected_feature], bins=20, color='skyblue', edgecolor='black')
        ax.set_xlabel(selected_feature)
        ax.set_ylabel('Frekuensi')
        ax.set_title(f'Distribusi {selected_feature}')
    else:
        ax.boxplot(df[selected_feature], vert=True)
        ax.set_ylabel(selected_feature)
        ax.set_title(f'Box Plot {selected_feature}')
    
    st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("🔗 Korelasi dengan Nilai G3")
    
    correlations = df[numeric_cols].corr()['G3'].sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    correlations.drop('G3').plot(kind='barh', ax=ax, color='coral')
    ax.set_xlabel('Korelasi dengan G3')
    ax.set_title('Korelasi Fitur Numerik dengan Nilai G3')
    ax.grid(axis='x', alpha=0.3)
    
    st.pyplot(fig)

# ================================
# Page: Analisis Data Kategorikal
# ================================
elif page == "📋 Analisis Data Kategorikal":
    st.title("📋 Analisis Data Kategorikal")
    st.markdown("---")
    
    # Load raw data
    df = pd.read_csv("data/student-mat.csv", sep=";")
    
    categorical_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 
                        'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 
                        'nursery', 'higher', 'internet', 'romantic']
    
    st.subheader("📊 Distribusi Data Kategorikal")
    
    selected_cat = st.selectbox("Pilih Fitur Kategorikal:", categorical_cols)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Distribusi {selected_cat}**")
        value_counts = df[selected_cat].value_counts()
        st.dataframe(value_counts.reset_index().rename(columns={'index': selected_cat, selected_cat: 'Count'}), 
                     use_container_width=True)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        value_counts.plot(kind='bar', ax=ax, color='steelblue')
        ax.set_title(f'Distribusi {selected_cat}')
        ax.set_xlabel(selected_cat)
        ax.set_ylabel('Jumlah')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("📈 Rata-rata G3 Berdasarkan Kategori")
    
    avg_g3 = df.groupby(selected_cat)['G3'].mean().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_g3.plot(kind='barh', ax=ax, color='lightcoral')
    ax.set_xlabel('Rata-rata Nilai G3')
    ax.set_title(f'Rata-rata G3 Berdasarkan {selected_cat}')
    ax.grid(axis='x', alpha=0.3)
    
    st.pyplot(fig)

# ================================
# Page: Evaluasi Model
# ================================
elif page == "📈 Evaluasi Model":
    st.title("📈 Evaluasi Performa Model")
    st.markdown("---")
    
    # Prepare scaled data for ANN
    scaler_eval = StandardScaler()
    X_train_scaled = scaler_eval.fit_transform(X_train)
    X_test_scaled = scaler_eval.transform(X_test)
    
    # Predictions
    with st.spinner('⏳ Menghitung prediksi...'):
        y_pred_ann = ann_model.predict(X_test_scaled, verbose=0).flatten()
        y_pred_rf = rf_model.predict(X_test)
    
    # Calculate metrics
    def evaluate_model(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return mae, mse, rmse, r2
    
    mae_ann, mse_ann, rmse_ann, r2_ann = evaluate_model(y_test, y_pred_ann)
    mae_rf, mse_rf, rmse_rf, r2_rf = evaluate_model(y_test, y_pred_rf)
    
    # Display metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='padding: 1.5rem; border-radius: 10px; background-color: #e3f2fd; margin-bottom: 1rem;'>
            <h3 style='color: #1976d2; text-align: center;'>🧠 Artificial Neural Network (ANN)</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style='padding: 1rem; border-radius: 8px; background-color: #f5f5f5; margin-bottom: 0.5rem;'>
            <p style='margin: 0;'><strong>MAE:</strong> {mae_ann:.4f}</p>
        </div>
        <div style='padding: 1rem; border-radius: 8px; background-color: #f5f5f5; margin-bottom: 0.5rem;'>
            <p style='margin: 0;'><strong>MSE:</strong> {mse_ann:.4f}</p>
        </div>
        <div style='padding: 1rem; border-radius: 8px; background-color: #f5f5f5; margin-bottom: 0.5rem;'>
            <p style='margin: 0;'><strong>RMSE:</strong> {rmse_ann:.4f}</p>
        </div>
        <div style='padding: 1rem; border-radius: 8px; background-color: #f5f5f5; margin-bottom: 0.5rem;'>
            <p style='margin: 0;'><strong>R²:</strong> {r2_ann:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='padding: 1.5rem; border-radius: 10px; background-color: #e8f5e9; margin-bottom: 1rem;'>
            <h3 style='color: #388e3c; text-align: center;'>🌲 Random Forest</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style='padding: 1rem; border-radius: 8px; background-color: #f5f5f5; margin-bottom: 0.5rem;'>
            <p style='margin: 0;'><strong>MAE:</strong> {mae_rf:.4f}</p>
        </div>
        <div style='padding: 1rem; border-radius: 8px; background-color: #f5f5f5; margin-bottom: 0.5rem;'>
            <p style='margin: 0;'><strong>MSE:</strong> {mse_rf:.4f}</p>
        </div>
        <div style='padding: 1rem; border-radius: 8px; background-color: #f5f5f5; margin-bottom: 0.5rem;'>
            <p style='margin: 0;'><strong>RMSE:</strong> {rmse_rf:.4f}</p>
        </div>
        <div style='padding: 1rem; border-radius: 8px; background-color: #f5f5f5; margin-bottom: 0.5rem;'>
            <p style='margin: 0;'><strong>R²:</strong> {r2_rf:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Comparison
    st.markdown("---")
    st.subheader("📊 Perbandingan Performa Model")
    
    metrics_df = pd.DataFrame({
        'Metric': ['MAE', 'MSE', 'RMSE', 'R²'],
        'ANN': [mae_ann, mse_ann, rmse_ann, r2_ann],
        'Random Forest': [mae_rf, mse_rf, rmse_rf, r2_rf]
    })
    
    st.dataframe(metrics_df.set_index('Metric'), use_container_width=True)
    
    # Visualization
    st.markdown("---")
    st.subheader("📈 Visualisasi Hasil Prediksi")
    
    tab1, tab2, tab3 = st.tabs(["Actual vs Predicted", "Residual Plot", "Error Distribution"])
    
    with tab1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # ANN
        axes[0].scatter(y_test, y_pred_ann, alpha=0.5, color='blue')
        axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual G3')
        axes[0].set_ylabel('Predicted G3')
        axes[0].set_title(f'ANN (R²={r2_ann:.4f})')
        axes[0].grid(True, alpha=0.3)
        
        # Random Forest
        axes[1].scatter(y_test, y_pred_rf, alpha=0.5, color='green')
        axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1].set_xlabel('Actual G3')
        axes[1].set_ylabel('Predicted G3')
        axes[1].set_title(f'Random Forest (R²={r2_rf:.4f})')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # ANN Residuals
        residuals_ann = y_test - y_pred_ann
        axes[0].scatter(y_pred_ann, residuals_ann, alpha=0.5, color='blue')
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Predicted G3')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('ANN - Residual Plot')
        axes[0].grid(True, alpha=0.3)
        
        # Random Forest Residuals
        residuals_rf = y_test - y_pred_rf
        axes[1].scatter(y_pred_rf, residuals_rf, alpha=0.5, color='green')
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted G3')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Random Forest - Residual Plot')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab3:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # ANN Error Distribution
        axes[0].hist(residuals_ann, bins=20, color='blue', alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Prediction Error')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('ANN - Error Distribution')
        axes[0].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[0].grid(True, alpha=0.3)
        
        # Random Forest Error Distribution
        axes[1].hist(residuals_rf, bins=20, color='green', alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Prediction Error')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Random Forest - Error Distribution')
        axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # ================================
    # Best Model Section
    # ================================
    st.markdown("---")
    st.markdown("## 🏆 Model Terbaik")
    
    # Determine best model based on multiple metrics
    metrics_comparison = {
        'MAE': (mae_ann, mae_rf, 'lower'),
        'MSE': (mse_ann, mse_rf, 'lower'),
        'RMSE': (rmse_ann, rmse_rf, 'lower'),
        'R²': (r2_ann, r2_rf, 'higher')
    }
    
    ann_wins = 0
    rf_wins = 0
    
    for metric, (ann_val, rf_val, better) in metrics_comparison.items():
        if better == 'lower':
            if ann_val < rf_val:
                ann_wins += 1
            else:
                rf_wins += 1
        else:  # higher is better (R²)
            if ann_val > rf_val:
                ann_wins += 1
            else:
                rf_wins += 1
    
    # Determine winner
    if ann_wins > rf_wins:
        best_model = "ANN"
        best_color = "#1976d2"
        best_bg = "#e3f2fd"
        best_icon = "🧠"
        best_name = "Artificial Neural Network"
        best_mae = mae_ann
        best_mse = mse_ann
        best_rmse = rmse_ann
        best_r2 = r2_ann
        win_count = ann_wins
    else:
        best_model = "Random Forest"
        best_color = "#388e3c"
        best_bg = "#e8f5e9"
        best_icon = "🌲"
        best_name = "Random Forest"
        best_mae = mae_rf
        best_mse = mse_rf
        best_rmse = rmse_rf
        best_r2 = r2_rf
        win_count = rf_wins
    
    # Display best model with attractive UI
    st.markdown(f"""
    <div style='padding: 2rem; border-radius: 15px; background: linear-gradient(135deg, {best_color} 0%, {best_color}dd 100%); color: white; margin-bottom: 2rem; box-shadow: 0 8px 16px rgba(0,0,0,0.2);'>
        <div style='text-align: center;'>
            <h1 style='color: white; margin: 0; font-size: 3rem;'>{best_icon}</h1>
            <h2 style='color: white; margin: 0.5rem 0;'>{best_name}</h2>
            <p style='color: white; font-size: 1.2rem; margin: 0;'>Model dengan Performa Terbaik</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Performance Summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style='padding: 1.5rem; border-radius: 10px; background-color: {best_bg}; text-align: center; border: 3px solid {best_color};'>
            <h3 style='color: {best_color}; margin: 0;'>📊 Skor Kemenangan</h3>
            <h1 style='color: {best_color}; margin: 0.5rem 0; font-size: 3rem;'>{win_count}/4</h1>
            <p style='color: {best_color}; margin: 0;'>Metrik Terbaik</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='padding: 1.5rem; border-radius: 10px; background-color: {best_bg}; text-align: center; border: 3px solid {best_color};'>
            <h3 style='color: {best_color}; margin: 0;'>🎯 Akurasi (R²)</h3>
            <h1 style='color: {best_color}; margin: 0.5rem 0; font-size: 3rem;'>{best_r2:.3f}</h1>
            <p style='color: {best_color}; margin: 0;'>Coefficient of Determination</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style='padding: 1.5rem; border-radius: 10px; background-color: {best_bg}; text-align: center; border: 3px solid {best_color};'>
            <h3 style='color: {best_color}; margin: 0;'>📉 Error Rata-rata</h3>
            <h1 style='color: {best_color}; margin: 0.5rem 0; font-size: 3rem;'>{best_mae:.3f}</h1>
            <p style='color: {best_color}; margin: 0;'>Mean Absolute Error</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Detailed Comparison Table
    st.markdown(f"""
    <div style='padding: 1.5rem; border-radius: 10px; background-color: #f8f9fa; border-left: 5px solid {best_color};'>
        <h3 style='color: {best_color}; margin-top: 0;'>📋 Ringkasan Metrik Model Terbaik</h3>
    </div>
    """, unsafe_allow_html=True)
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("MAE (Lower is Better)", f"{best_mae:.4f}", 
                  delta=f"{abs(mae_ann - mae_rf):.4f}" if best_model == "ANN" else f"-{abs(mae_ann - mae_rf):.4f}",
                  delta_color="inverse")
    
    with metric_col2:
        st.metric("MSE (Lower is Better)", f"{best_mse:.4f}",
                  delta=f"{abs(mse_ann - mse_rf):.4f}" if best_model == "ANN" else f"-{abs(mse_ann - mse_rf):.4f}",
                  delta_color="inverse")
    
    with metric_col3:
        st.metric("RMSE (Lower is Better)", f"{best_rmse:.4f}",
                  delta=f"{abs(rmse_ann - rmse_rf):.4f}" if best_model == "ANN" else f"-{abs(rmse_ann - rmse_rf):.4f}",
                  delta_color="inverse")
    
    with metric_col4:
        st.metric("R² (Higher is Better)", f"{best_r2:.4f}",
                  delta=f"{abs(r2_ann - r2_rf):.4f}" if best_model == "ANN" else f"{abs(r2_ann - r2_rf):.4f}",
                  delta_color="normal")
    
    # Recommendation
    st.markdown("---")
    st.markdown(f"""
    <div style='padding: 2rem; border-radius: 10px; background-color: {best_bg}; border: 2px solid {best_color};'>
        <h3 style='color: {best_color};'>💡 Rekomendasi</h3>
        <p style='font-size: 1.1rem; line-height: 1.8;'>
            Berdasarkan evaluasi komprehensif terhadap 4 metrik utama (MAE, MSE, RMSE, R²), 
            <strong style='color: {best_color};'>{best_name}</strong> menunjukkan performa terbaik dengan memenangkan 
            <strong style='color: {best_color};'>{win_count} dari 4</strong> metrik evaluasi.
            <br><br>
            Model ini direkomendasikan untuk digunakan dalam prediksi nilai akhir siswa (G3) karena 
            memberikan prediksi yang lebih akurat dengan error yang lebih kecil.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ================================
# Footer
# ================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>📚 Sistem Prediksi Nilai Siswa menggunakan Machine Learning</p>
    <p>Algoritma: ANN & Random Forest | Data: Student Performance Dataset</p>
</div>
""", unsafe_allow_html=True)
