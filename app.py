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

# 2. Load Model (Cache agar tidak load berulang kali)
@st.cache_resource
def load_model():
    # Pastikan nama file model sesuai dengan yang Anda simpan
    try:
        model = joblib.load('model_rf_alpukat.pkl') 
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# 3. Fungsi Preprocessing & Ekstraksi (Diadaptasi dari kode Anda)

def preprocess_image(img_array):
    # img_array sudah berupa gambar (bukan path), jadi tidak perlu imread lagi
    if img_array is None:
        raise ValueError("Gambar tidak terbaca")

    img = cv2.resize(img_array, (224, 224))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Note: Normalize di OpenCV biasanya outputnya float jika tidak di-cast, 
    # tapi agar aman untuk GLCM (yang butuh int), kita pastikan formatnya uint8
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = np.uint8(img) # Memastikan tipe data integer untuk GLCM
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

    # Scikit-image graycomatrix
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
st.write("Sistem Klasifikasi Random Forest (HSV + GLCM + Edge)")

uploaded_file = st.file_uploader("Upload foto alpukat", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # --- A. Baca File dari Upload ---
    # Convert file buffer ke format array yang bisa dibaca OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, 1) # 1 = Color

    # Tampilkan gambar (Streamlit butuh format RGB, OpenCV default BGR)
    st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), 
             caption='Gambar yang diupload', width=300)

    # Tombol Prediksi
    if st.button("Prediksi Sekarang"):
        if model is not None:
            with st.spinner('Sedang memproses fitur...'):
                try:
                    # 1. Preprocess
                    processed_img = preprocess_image(original_image)
                    
                    # 2. Ekstraksi Fitur
                    hsv_feat  = extract_hsv(processed_img)
                    glcm_feat = extract_glcm(processed_img)
                    edge_feat = extract_edge(processed_img)

                    # 3. Gabung Fitur (Concatenate)
                    # Total fitur harus sama persis dengan saat training
                    feature_vector = np.array(
                        hsv_feat + glcm_feat + edge_feat
                    ).reshape(1, -1)

                    # 4. Prediksi
                    prediction_index = model.predict(feature_vector)[0]

                    # 5. Mapping Label
                    class_map = {
                        0: "Underripe (Belum Matang)",
                        1: "Ripe (Matang)",
                        2: "Overripe (Terlalu Matang/Busuk)"
                    }
                    
                    result_label = class_map.get(prediction_index, "Unknown")

                    # Tampilkan Hasil
                    st.success(f"Hasil Analisis: **{result_label}**")
                    
                    # (Opsional) Tampilkan detail fitur jika ingin debug
                    with st.expander("Lihat Detail Ekstraksi Fitur"):
                        st.write("HSV Features:", hsv_feat)
                        st.write("GLCM Features:", glcm_feat)
                        st.write("Edge Features:", edge_feat)

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses: {e}")
        else:
            st.error("Model belum dimuat. Pastikan file .pkl ada.")
