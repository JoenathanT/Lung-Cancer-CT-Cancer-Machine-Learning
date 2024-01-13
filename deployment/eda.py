import os
import glob
import random
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image


st.set_page_config(
    page_title = 'Exploratorty Data Analysis - CT SCAN ',
    layout = 'wide',
    initial_sidebar_state = 'expanded'
)

def run() :

    # Header 
    st.subheader('## This EDA is to understand the data for each label: 1.normal, 2.squamous cell carcinoma, 3. large cell carcinoma, 4. adenocarcinoma')


    # create variables for path
    main_path = 'Data'

    # train
    train_normal =os.path.join(main_path, 'train', 'normal')
    tarin_adenocarcinoma =os.path.join(main_path, 'train', 'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib')
    Train_squamous_cell_carcinoma =os.path.join(main_path, 'train', 'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa')
    train_large_cell_carcinoma =os.path.join(main_path, 'train', 'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa')

    # function to visualize each class
    def create_dataframe(list_of_images):
        data = []
        for image in list_of_images:
            data.append((image, image.split('\\')[-2]))  
        return pd.DataFrame(data, columns=['images', 'label'])

    train_df = create_dataframe(
        glob.glob(os.path.join(train_normal, '*.jpg'))+
        glob.glob(os.path.join(tarin_adenocarcinoma, '*.jpg'))+
        glob.glob(os.path.join(Train_squamous_cell_carcinoma, '*.jpg'))+
        glob.glob(os.path.join(train_large_cell_carcinoma, '*.jpg'))
    )
    train_df = train_df.sample(frac=1, random_state=7).reset_index(drop=True)

    def visualize_samples_by_label(df, label, num_samples=5):
        samples = df[df['label'] == label]['images'].iloc[:num_samples].tolist()
        print(f"Number of samples: {len(samples)}") 

        num_cols = min(num_samples, 5)
        num_rows = (num_samples - 1) // num_cols + 1
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(10, 2 * num_rows))
        count = 0
        for i in range(num_rows):
            for j in range(num_cols):
                if count < len(samples):
                    sample = samples[count]
                    img = Image.open(sample)
                    ax = axes[i, j]
                    ax.imshow(img)
                    ax.axis('off')
                    count += 1
                else:
                    break
        plt.tight_layout()
        st.pyplot(fig)
        

    # visualize 'normal' label 
    st.write(' 1. Normal Chest')
    visualize_samples_by_label(train_df, 'normal', num_samples=10)
    st.write('*Characteristics*')
    st.write('this is normal type of chest in CT SCAN, the size of lung is normal and there is no lump detected')

    st.markdown('----')

    # visualize 'adenocarcinoma' class
    st.write(' 2. Adenocarcinoma')
    visualize_samples_by_label(train_df, 'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib', num_samples=10)
    st.write('*Characteristics*')
    st.write('there is shrinkage or enlargement of the lungs and Ground-glass opacities may indicate early-stage adenocarcinomas.')

    st.markdown('----')

    # visualize 'squamous' class
    st.write(' 3. Squamous cell carcinoma')
    visualize_samples_by_label(train_df, 'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa', num_samples=10)
    st.write('*Characteristics*')
    st.write('They may present as a large, solitary, or dominant mass and The density may be heterogeneous, and areas of necrosis or hemorrhage may be present.')

    st.markdown('----')

    # visualize 'large cell' class
    st.write(' 4. Large cell carcinoma')
    visualize_samples_by_label(train_df, 'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa', num_samples=10)
    st.write('*Characteristics*')
    st.write('The density may be heterogeneous, and areas of necrosis or hemorrhage may be present and Large cell carcinomas often present as large, centrally located masses within the lung.')

    st.markdown('----')



if __name__ == '__main__':
    run()