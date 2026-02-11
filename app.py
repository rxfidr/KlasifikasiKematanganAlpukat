import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import graycomatrix, graycoprops

# 1. Konfigurasi Halaman
st.set_page_config(
    page_title="Prediksi Kematangan Alpukat",
    page_icon="🥑",
    layout="centered"
)

# 2. Load DUA Model (Klasifikasi & Outlier Detector)
@st.cache_resource
def load_models():
    try:
        # Load Model Klasifikasi (Random Forest)
        classifier = joblib.load('model_rf_alpukat.pkl') 
        
        # Load Model Satpam (Isolation Forest)
        # Pastikan file ini ada di folder Anda!
        outlier_detector = joblib.load('model_outlier_alpukat.pkl') 
        
        return classifier, outlier_detector
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None

rf_model, iso_model = load_models()

# 3. Fungsi Preprocessing & Ekstraksi

def preprocess_image(img_array):
    if img_array is None:
        raise ValueError("Gambar tidak terbaca")

    img = cv2.resize(img_array, (224, 224))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = np.uint8(img) 
    return img

def extract_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    features = []
    for i in range(3):
        features.append(np.mean(hsv[:,:,i]))
        features.append(np.std(hsv[:,:,i]))
    return features

def extract_glcm(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    return [
        graycoprops(glcm, 'contrast')[0,0],
        graycoprops(glcm, 'dissimilarity')[0,0],
        graycoprops(glcm, 'homogeneity')[0,0],
        graycoprops(glcm, 'energy')[0,0],
        graycoprops(glcm, 'correlation')[0,0]
    ]

def extract_edge(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    edge_pixels = np.sum(edges > 0)
    edge_ratio = edge_pixels / edges.size

    return [edge_pixels, edge_ratio]

# 4. Tampilan Web Utama
st.title("🥑 Cek Kematangan Alpukat")
st.write("Sistem Klasifikasi Random Forest dengan Validasi Objek")

uploaded_file = st.file_uploader("Upload foto alpukat", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # --- A. Baca File dari Upload ---
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, 1) # 1 = Color

    st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), 
             caption='Gambar yang diupload', width=300)

    # Tombol Prediksi
    if st.button("Prediksi Sekarang"):
        # Pastikan kedua model berhasil dimuat
        if rf_model is not None and iso_model is not None:
            with st.spinner('Sedang memproses fitur...'):
                try:
                    # 1. Preprocess
                    processed_img = preprocess_image(original_image)
                    
                    # 2. Ekstraksi Fitur
                    hsv_feat  = extract_hsv(processed_img)
                    glcm_feat = extract_glcm(processed_img)
                    edge_feat = extract_edge(processed_img)

                    # 3. Gabung Fitur
                    feature_vector = np.array(
                        hsv_feat + glcm_feat + edge_feat
                    ).reshape(1, -1)

                    # --- 4. CEK VALIDITAS OBJEK (ISOLATION FOREST) ---
                    # Predict return: 1 (Normal/Alpukat), -1 (Outlier/Bukan)
                    is_avocado = iso_model.predict(feature_vector)[0]

                    if is_avocado == -1:
                        # JIKA TERDETEKSI BUKAN ALPUKAT
                        st.error("⛔ Objek Tidak Dikenali / Bukan Alpukat")
                        st.warning("Sistem mendeteksi tekstur atau warna yang tidak sesuai dengan karakteristik alpukat.")
                        st.info("Jika ini benar alpukat, cobalah foto ulang dengan pencahayaan yang lebih baik atau latar belakang polos.")
                    
                    else:
                        # JIKA LOLOS VALIDASI -> LANJUT KE KLASIFIKASI RF
                        prediction_index = rf_model.predict(feature_vector)[0]

                        # Mapping Label
                        class_map = {
                            0: "Underripe (Belum Matang)",
                            1: "Ripe (Matang)",
                            2: "Overripe (Terlalu Matang/Busuk)"
                        }
                        
                        result_label = class_map.get(prediction_index, "Unknown")
                        st.success(f"Hasil Analisis: **{result_label}**")

                    # (Opsional) Tampilkan detail fitur untuk debugging
                    with st.expander("Lihat Detail Ekstraksi Fitur"):
                        st.write(f"Status Validasi: {'✅ Alpukat' if is_avocado == 1 else '❌ Outlier'}")
                        st.write("HSV Features:", hsv_feat)
                        st.write("GLCM Features:", glcm_feat)
                        st.write("Edge Features:", edge_feat)

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses: {e}")
        else:
            st.error("Model tidak lengkap. Pastikan file 'model_rf_alpukat.pkl' DAN 'model_outlier_alpukat.pkl' ada di folder.")
