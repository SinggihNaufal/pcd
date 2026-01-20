import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Deteksi Penyakit Daun", page_icon="🌿", layout="wide")


# --- 2. DEFINISI METRIK (Wajib ada untuk load model, meski tidak dipakai hitung) ---
def dice_coef(y_true, y_pred):
    return 0


def iou_score(y_true, y_pred):
    return 0


def pixel_accuracy(y_true, y_pred):
    return 0


# --- 3. LOAD MODEL (Hanya ResNet) ---
@st.cache_resource
def load_model():
    model_path = "unet_leaf_model.h5"

    if not os.path.exists(model_path):
        st.error(
            f"❌ File '{model_path}' tidak ditemukan! Pastikan file ada di folder yang sama."
        )
        return None

    custom_objects = {
        "dice_coef": dice_coef,
        "iou_score": iou_score,
        "pixel_accuracy": pixel_accuracy,
        "pixel_acc": pixel_accuracy,
    }

    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        return model
    except Exception as e:
        st.error(f"Error memuat model: {e}")
        return None


model = load_model()


# --- 4. PREPROCESSING ---
def preprocess_image(image_pil):
    # Convert ke RGB (cegah error jika upload PNG transparan)
    image_pil = image_pil.convert("RGB")
    img_array = np.array(image_pil)

    # Resize ke ukuran input model (256x256)
    img_resized = cv2.resize(img_array, (256, 256))

    # Normalisasi (0-1)
    img_norm = img_resized.astype("float32") / 255.0

    # Tambah dimensi batch
    img_input = np.expand_dims(img_norm, axis=0)

    return img_array, img_input


# --- 5. TAMPILAN GUI ---
st.title("🌿 Deteksi Penyakit Daun")
st.markdown(
    "Aplikasi berbasis **ResNet50-UNet** untuk mendeteksi area penyakit pada daun."
)

# Sidebar Input
with st.sidebar:
    st.header("1. Upload Gambar")
    uploaded_file = st.file_uploader("Format: JPG, PNG", type=["jpg", "jpeg", "png"])

    st.divider()
    st.header("2. Pengaturan")
    threshold = st.slider("Sensitivitas", 0.1, 0.9, 0.5, 0.05)
    st.caption(
        "Geser ke kiri jika penyakit tidak terdeteksi (kurang sensitif). Geser ke kanan jika terlalu banyak noise."
    )

# Logika Utama
if uploaded_file is not None and model is not None:

    # Proses Gambar
    image_pil = Image.open(uploaded_file)
    original_img, input_tensor = preprocess_image(image_pil)

    # Buat 2 Kolom Berdampingan
    col1, col2 = st.columns(2)

    # --- KOLOM KIRI: GAMBAR ASLI ---
    with col1:
        st.subheader("📸 Gambar Asli")
        st.image(original_img, use_container_width=True)

    # --- KOLOM KANAN: HASIL PREDIKSI ---
    with col2:
        st.subheader("🔍 Hasil Deteksi (ResNet)")

        # Tampilkan loading saat proses
        with st.spinner("Sedang memindai daun..."):

            # 1. Prediksi
            pred_raw = model.predict(input_tensor)[0]

            # 2. Buat Masker Binary
            mask_uint8 = (pred_raw > threshold).astype(np.uint8) * 255

            # 3. Resize masker agar sama dengan ukuran gambar asli user
            # (Penting agar overlay pas, meskipun gambar user ukurannya besar)
            mask_resized = cv2.resize(
                mask_uint8, (original_img.shape[1], original_img.shape[0])
            )

            # 4. Buat Overlay Warna Merah
            overlay = original_img.copy()
            # Warnai area penyakit (putih di mask) menjadi Merah [255, 0, 0]
            overlay[mask_resized > 0] = [255, 0, 0]

            # 5. Gabungkan (Blending)
            final_view = cv2.addWeighted(original_img, 0.7, overlay, 0.3, 0)

            # Tampilkan
            st.image(
                final_view,
                caption="Area Penyakit Ditandai Merah",
                use_container_width=True,
            )

else:
    # Tampilan awal kosong
    st.info("👈 Silakan upload gambar daun di menu sebelah kiri.")
