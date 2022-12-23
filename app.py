# Dependencies

import streamlit as st
import json

from Scripts.model_runner import predict
from Scripts.show_chart import show_charts, show_acc_charts


# setting streamlit page configurations

st.set_page_config(
    page_title="Medical Diagnosis Platform",
    layout="wide",
    page_icon=None,
    initial_sidebar_state="auto",
    menu_items=None
)



def load_database():
    with open("./database.json") as f:
        database = json.load(f)

    return database



# loading database
database = load_database()


def main():

    # making page structure
    st.title("Medical Diagnosis Platform") 
    st.subheader("Currently Supporting 14 different disease diagnosis")
    st.caption("Use dark theme using three horizontal bars on top-right for better experience")

    st.info("""PLEASE NOTE :- Artificial Intelligence Models for disease prediction maybe be helpful in diagnosing diseases up to some extent but you should not completely rely on these predictions. Please consult your doctor for the best possible advice. This project is made to predict the possibility of these diseases at earlier stages, but these predictions are not 100% accurate.""")

    #Prediction Subheading
    st.subheader("Prediction")
    info_box = st.empty()


    viewer_columns = st.columns([0.1, 2, 1])
    #viewer_columns[1].empty().image("Media/human-lung-disease-prediction-icon.jpg")

    buffer = viewer_columns[2].file_uploader(
        label = "Upload test file here for disease prediction",
        type = [".jpg", ".jpeg", ".png"],
        accept_multiple_files = False,
        on_change = None
    )

    info_box.info("No file uploaded")

    if buffer != None:
        viewer_columns[1].image(buffer, caption="original uploaded image")
        info_box.warning("Your file is uploaded... Please wait while we predict the output")
        
        graph_data = predict(
            input_buffer = buffer,
            database = database
        )

        info_box.info("Predictions are ready... Please look below")

        show_charts(st.container(), graph_data)
        show_acc_charts(st.container(), graph_data)


main()
