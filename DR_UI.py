import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd
import cv2  # Library OpenCV untuk pemrosesan citra

# -------------------------------------------------------------------------
# 1. KONFIGURASI HALAMAN
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Deteksi Diabetic Retinopathy (Ben Graham)",
    page_icon="👁️",
    layout="wide"
)

# -------------------------------------------------------------------------
# 2. DEFINISI KELAS & FUNGSI USER (BEN GRAHAM)
# -------------------------------------------------------------------------
CLASS_NAMES = {
    0: "No DR (Tidak Ada Retinopati)",
    1: "Mild (Ringan)",
    2: "Moderate (Sedang)",
    3: "Severe (Parah)",
    4: "Proliferative (Proliferatif)"
}

def crop_image_from_gray(img, tol=7):
    """Memotong area hitam di sekitar retina"""
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:,:,0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0): return img 
        img1 = img[:,:,0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:,:,1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:,:,2][np.ix_(mask.any(1), mask.any(0))]
        return np.stack([img1, img2, img3], axis=-1)

def ben_graham_process(img, size=300):
    """
    Standarisasi pencahayaan dan tekstur (Ben Graham Method).
    Parameter 'size' disesuaikan dengan input shape model Anda.
    """
    # 1. Auto Crop
    img = crop_image_from_gray(img)
    
    # 2. Resize
    # Menggunakan Inter_Area lebih baik untuk pengecilan gambar medis
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    
    # 3. Gaussian Blur Blending
    # Ini membuat pembuluh darah menonjol & background seragam
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), size/30), -4, 128)
    return img

# -------------------------------------------------------------------------
# 3. FUNGSI LOAD MODEL & WRAPPER PREPROCESSING
# -------------------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        # Ganti dengan path model Anda
        model = tf.keras.models.load_model('best_model_dr_final_extended.keras')
        return model
    except Exception as e:
        return None

def preprocess_and_prepare(image_pil, target_size=300):
    """
    Mengubah PIL Image -> Numpy -> Ben Graham Process -> Model Input
    """
    # Convert PIL ke Numpy Array (RGB)
    img_array = np.array(image_pil)
    
    # Terapkan Ben Graham Processing
    # Pastikan target_size sesuai dengan input model (misal 224, 299, 300)
    processed_img = ben_graham_process(img_array, size=target_size)
    
    # Normalisasi (Biasanya model DR dilatih dengan rescaling 1/255)
    model_input = processed_img.astype(np.float32) / 255.0
    
    # Tambah dimensi batch (1, H, W, C)
    model_input = np.expand_dims(model_input, axis=0)
    
    return model_input, processed_img

# -------------------------------------------------------------------------
# 4. HALAMAN UTAMA & LOGIKA
# -------------------------------------------------------------------------
st.sidebar.title("DR Detector AI")
st.sidebar.info("Model menggunakan preprocessing **Ben Graham** untuk standarisasi pencahayaan citra.")

st.title("👁️ Analisis Risiko Diabetic Retinopathy")
st.markdown("Deteksi tingkat keparahan dengan *Image Processing Enhancement*.")

uploaded_file = st.file_uploader("Unggah gambar fundus...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Buka gambar dengan PIL
    image_source = Image.open(uploaded_file).convert("RGB")
    
    # Load Model
    model = load_model()
    
    # --- PREPROCESSING ---
    # Ubah target_size di sini sesuai arsitektur model (misal: EfficientNetB0=224, B3=300)
    INPUT_SIZE = 300
    final_input, displayed_processed_img = preprocess_and_prepare(image_source, target_size=INPUT_SIZE)

    # --- TAMPILAN VISUAL (ORIGINAL vs PROCESSED) ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Citra Asli")
        st.image(image_source, use_column_width=True)
    
    with col2:
        st.subheader("2. Input Model (Ben Graham)")
        st.image(displayed_processed_img, caption="Crop + Resize + Gaussian Blending", use_column_width=True)

    # --- PREDIKSI ---
    st.markdown("---")
    st.subheader("Hasil Diagnosa AI")
    
    if model is None:
        st.error("Model tidak ditemukan. Pastikan file .h5 ada.")
        # Dummy prediction untuk demo UI (Hapus saat production)
        prediction = np.random.rand(1, 5)
        prediction = prediction / np.sum(prediction)
    else:
        with st.spinner('Sedang melakukan inferensi...'):
            prediction = model.predict(final_input)

    # --- INTERPRETASI HASIL ---
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    class_label = CLASS_NAMES[class_idx]
    
    # Styling hasil
    result_color = "green" if class_idx == 0 else ("orange" if class_idx <= 2 else "red")
    st.markdown(f"""
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid {result_color};">
        <h2 style="color: #31333F; margin:0;">Prediksi: {class_label}</h2>
        <p style="margin:0; font-size: 18px;">Tingkat Kepercayaan: <strong>{confidence:.2f}%</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chart
    st.write("")
    st.write("**Distribusi Probabilitas Kelas:**")
    prob_df = pd.DataFrame({'Kondisi': list(CLASS_NAMES.values()), 'Probabilitas': prediction[0]})
    st.bar_chart(prob_df.set_index('Kondisi'))

else:
    # Placeholder
    st.info("Menunggu input gambar...")