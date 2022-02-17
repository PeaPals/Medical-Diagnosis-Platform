"""

File : app.py
File_Info : This file contains python streamlit code for main app
Author : CRP
License : None

"""

# Dependencies

import streamlit as st
import json

from Scripts.model_runner import predict
from Scripts.model_disease_info import model_disease_info


# setting streamlit page configurations

st.set_page_config(
    page_title="Medical Diagnosis Platform",
    layout="wide",
    page_icon=None,
    initial_sidebar_state="auto",
    menu_items=None
)

# page_icon=":smiley:"



#@st.cache(ttl=24*60*60)
#@st.cache    # no need for ttl ... its static data
def load_database():
    with open("./database.json") as f:
        database = json.load(f)

    disease_model_relation = database["disease_model_relation"]
    model_info = database["model_info"]
    segmented_output_provider = database["segmented_output_models"]

    return disease_model_relation, model_info, segmented_output_provider


# loading database
disease_model_relation, model_info, segmented_output_provider = load_database()
available_diseases = disease_model_relation.keys()



def main():

    # making page structure
    st.title("Medical Diagnosis Platform")

    selected_disease = st.selectbox(
        label="Select a disease diagnosis task",
        options=available_diseases,
        help="Select a disease diagnosis task"
    )

    model_disease_info_box = st.expander("Click here to see model and disease info")

    # Model Info Expander
    selected_model = disease_model_relation[selected_disease]
    
    model_disease_info(
        model = selected_model,
        disease = selected_disease,
        expander_box = model_disease_info_box
    )

    #Prediction Subheading
    st.subheader("Prediction")
    output_box = st.empty()


    if selected_model in segmented_output_provider:
        viewer_columns = st.columns([0.1, 1, 1])
        #viewer_columns[1].empty().image("Media/human-lung-disease-prediction-icon.jpg")

        buffer = st.file_uploader(
            label = "Upload test file here for disease prediction",
            type = [".jpg", ".png"],
            accept_multiple_files = False,
            on_change = None
        )

        output_box.info("No file uploaded")

        if buffer != None:
            viewer_columns[1].image(buffer, caption="original uploaded image")
            output_box.warning("Your file is uploaded... Please wait while we predict the output")
            
            predict(
                model = selected_model,
                input_buffer = buffer,
                output_box = output_box,
                segmented_image_viewer = viewer_columns[2]
            )

            # TODO : Add a download button for segment image download


    else:
        viewer_columns = st.columns([0.1, 2, 1])
        #viewer_columns[1].empty().image("Media/human-lung-disease-prediction-icon.jpg")

        buffer = viewer_columns[2].file_uploader(
            label = "Upload test file here for disease prediction",
            type = [".jpg", ".png"],
            accept_multiple_files = False,
            on_change = None
        )

        output_box.info("No file uploaded")

        if buffer != None:
            viewer_columns[1].image(buffer, caption="original uploaded image")
            output_box.warning("Your file is uploaded... Please wait while we predict the output")
            
            predict(
                model = selected_model,
                input_buffer = buffer,
                output_box = output_box
            )


main()