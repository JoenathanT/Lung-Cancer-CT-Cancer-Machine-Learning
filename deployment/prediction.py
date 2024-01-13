import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import h5py

def run():
    st.header('CT SCAN PREDICTION')
    st.image('https://www.thetimes.co.uk/imageserver/image/%2Fmethode%2Fsundaytimes%2Fprod%2Fweb%2Fbin%2F20549920-bb0f-11e7-8b2e-f28d30e9c9fd.jpg?crop=2667%2C1500%2C0%2C0',
             caption='hello... my name is DEATH, i can di diagnose your disease by your CT SCAN photos')
    st.write('this model can predict: 1. normal, 2.squamous.cell.carcinoma, 3.large.cell.carcinoma, 4.adenocarcinoma ')
    st.write('*note* you can download image as the inputer image for this files')
    st.write('')
    with h5py.File('best_model.h5', 'r') as f:
        model = load_model(f, compile=False)

    uploaded_files = st.file_uploader('Upload files images', type=['png', 'jpg', 'jpeg'])
    if uploaded_files is not None:
        image = Image.open(uploaded_files)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        #preprocessing images
        resized_image = image.resize((100, 100))  
        normalized_image = np.array(resized_image) / 255.0  
        input_image = np.expand_dims(normalized_image, axis=0)  

        # Perform image classification
        prediction = model.predict(input_image)
        predicted_class = np.argmax(prediction)

        class_names = ['normal', 'adenocarcinoma', 'large.cell.carcinoma','squamous.cell.carcinoma']
        predicted_label = class_names[predicted_class]
        st.write('Predicted Class:', predicted_label)


if __name__ == '__main__':
    run()



    
