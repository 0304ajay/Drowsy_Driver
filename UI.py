import streamlit as st
from PIL import Image
import numpy as np

# Set page title
st.set_page_config(page_title='Upload and Display Image')

# Main Streamlit application
def main():
    st.title('Upload and Display Image')

    # File uploader widget
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Optional: Convert the image to a NumPy array
        img_array = np.array(image)
        st.write(f'Image shape: {img_array.shape}')  # Display the shape of the image array

# Run the app
if __name__ == '__main__':
    main()
