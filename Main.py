import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from keras.models import load_model
import time

def hide_github_icon():
  """Hides the entire Streamlit main menu including the GitHub icon."""
  style = """
  MainMenu {
    display: none;
  }
  """
  st.set_page_config(page_title=None, layout="wide", initial_sidebar_state="collapsed", menu_items={})
  st.write('<style>'+ style +'</style>', unsafe_allow_html=True)

# Call the function to hide the menu
hide_github_icon()

# Function to load and predict using a model
def load_and_predict(model_path, class_indices, title):
    # Load the trained model
    model = load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Create the Streamlit app
    st.title(title)

    # CSS Styling
    st.markdown("""
        <style>
            .prediction-container {
                background-color: #f0f0f0;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0px 0px 10px 0px rgba(0,0,0,0.1);
                margin-top: 20px;
            }
            .prediction-label {
                font-size: 24px;
                color: #2ecc71; /* Light Green */
                margin-bottom: 10px;
            }
            .confidence {
                font-size: 20px;
                color: #666;
            }
            .center {
                display: flex;
                justify-content: center;
            }
            .spinner-container {
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100px;
            }
            .spinner {
                border: 8px solid #f3f3f3; /* Light grey */
                border-top: 8px solid #3498db; /* Blue */
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 2s linear infinite;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = image.convert("RGB")
        st.write("<div class='center'>", unsafe_allow_html=True)
        st.image(image, caption='Uploaded Image.', width=300)  # Set the width to 300 pixels
        st.write("</div>", unsafe_allow_html=True)
        st.write("")
        with st.spinner(text='Classifying...'):
            time.sleep(2)  # Simulate some processing time
            label, confidence = predict(image, model, class_indices)
            st.write("""
                <div class='prediction-container'>
                    <div class='prediction-label'>Prediction: {}</div>
                    <div class='confidence'>Confidence: {}%</div>
                </div>
            """.format(label, confidence), unsafe_allow_html=True)

# Function to make predictions
def predict(image, model, class_indices):
    img_array = np.array(image)
    img_array = tf.image.resize(img_array, (256, 256))
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_indices[predicted_class_index]
    confidence = round(100 * (predictions[0][predicted_class_index]), 1)
    return predicted_class, confidence

# Define class names and indices for both models
potato_class_indices = {0: 'Early Blight', 1: 'Late Blight', 2: 'Healthy'}
mango_class_indices = {0: 'Anthracnose', 1: 'Bacterial Canker', 2: 'Cutting Weevil',3:'Die Back',4: 'Gall Midge',5: 'Healthy',6: 'Powdery Mildew',7: 'Sooty Mould'}

# Sidebar
st.sidebar.title('Navigator')
navigation = st.sidebar.selectbox('', ('Home', 'Leaf Disease Classifier'))

if navigation == 'Home':
    st.image('leaf.png', use_column_width=True)
    st.write('Welcome to the Leaf Disease Classifier website! This website helps you classify leaf diseases of both potatoes and mangoes. Select the "Leaf Disease Classifier" option from the sidebar to start classifying images of potato or mango leaves.')

    # Display the User Guide video
    st.title('User Guide')
    st.write("- Please make sure you are familiar with the type of leaf (mango or potato) before using the Leaf Disease Classifier.")
    st.video("UserGuide.mp4")

    
    st.header('About Potato Leaf Diseases:')
    st.write('Potato Leaf Disease Classifier consists of 3 classes:')
    st.write('- Early Blight')
    st.write('- Healthy')
    st.write('- Late Blight')

    st.header('About Mango Leaf Diseases:')
    st.write('Mango Leaf Disease Classifier consists of 8 classes:')
    st.write('- Anthracnose')
    st.write('- Bacterial Canker')
    st.write('- Cutting Weevil')
    st.write('- Die Back')
    st.write('- Gall Midge')
    st.write('- Healthy')
    st.write('- Powdery Mildew')
    st.write('- Sooty Mould')

else:
    # Main heading
    st.title('Leaf Disease Classifier')

    # Select option for classification
    option = st.selectbox('Select Leaf Type for Classification', ('Potato', 'Mango'))

    if option == 'Potato':
        load_and_predict('Potato.h5', potato_class_indices, 'Potato Leaf Disease Classifier')
    elif option == 'Mango':
        load_and_predict('Mango.h5', mango_class_indices, 'Mango Leaf Disease Classifier')
