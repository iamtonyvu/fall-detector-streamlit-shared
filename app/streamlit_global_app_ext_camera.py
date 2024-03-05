
import streamlit as st
from ultralytics import YOLO
from utils_ext_camera import play_webcam
from PIL import Image


st.set_page_config(page_title="Fall Detector with streamlit",
                   page_icon='camera_icon.png'
                   )
st.header('Fall Detector - Model Explanation', divider='rainbow')

# Load the image for the title
title_image = Image.open("1457450.jpg")

# Display the title image
# st.image(title_image, width=300)

model = YOLO('best.pt')

st.subheader(':camera: Live camera')

st.info(':red[falling : red]  \n :orange[sitting : orange]  \n :pink[standing : pink]')


play_webcam(conf=0.5, model=model)
