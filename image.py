import streamlit as st
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import cv2
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="🛰️ Satellite Image Processor", layout="centered")
st.title("🛰️ Satellite Image Analysis with Streamlit")

uploaded_file = st.file_uploader("Upload a multi-band GeoTIFF image", type=["tif", "tiff"])

if uploaded_file:
    st.success("File uploaded successfully!")

    with rasterio.open(uploaded_file) as src:
        bands = src.count
        width, height = src.width, src.height
        profile = src.profile
        st.write(f"Image size: {width}x{height}, Bands: {bands}")

        # Read bands
        try:
            red = src.read(3).astype(float)  # Band 3: Red
            nir = src.read(4).astype(float)  # Band 4: NIR
            green = src.read(2).astype(float)  # Band 2: Green
        except IndexError:
            st.error("This image does not contain enough bands (need at least 4).")
            st.stop()

        # Calculate NDVI
        ndvi = (nir - red) / (nir + red + 1e-5)
        ndvi_image = np.clip((ndvi + 1) / 2, 0, 1)

        # Calculate NDWI
        ndwi = (green - nir) / (green + nir + 1e-5)
        ndwi_image = np.clip((ndwi + 1) / 2, 0, 1)

        # Show NDVI
        st.subheader("🌿 NDVI (Vegetation Index)")
        fig1, ax1 = plt.subplots()
        ndvi_plot = ax1.imshow(ndvi_image, cmap='RdYlGn')
        plt.colorbar(ndvi_plot, ax=ax1, fraction=0.04)
        st.pyplot(fig1)

        # Show NDWI
        st.subheader("💧 NDWI (Water Index)")
        fig2, ax2 = plt.subplots()
        ndwi_plot = ax2.imshow(ndwi_image, cmap='Blues')
        plt.colorbar(ndwi_plot, ax=ax2, fraction=0.04)
        st.pyplot(fig2)

        # Convert to 8-bit for OpenCV edge detection
        ndvi_uint8 = (ndvi_image * 255).astype(np.uint8)

        # Edge detection
        st.subheader("⚙️ Edge Detection (on NDVI)")
        low_thresh = st.slider("Low Threshold", 50, 200, 100)
        high_thresh = st.slider("High Threshold", 100, 300, 200)
        edges = cv2.Canny(ndvi_uint8, low_thresh, high_thresh)

        fig3, ax3 = plt.subplots()
        ax3.imshow(edges, cmap='gray')
        ax3.set_title("Canny Edge Detection")
        ax3.axis('off')
        st.pyplot(fig3)

        # Download edge image as PNG
        st.subheader("📥 Download Edge Image")
        im_pil = Image.fromarray(edges)
        buf = BytesIO()
        im_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button("Download Edges as PNG", data=byte_im, file_name="edges.png", mime="image/png")
