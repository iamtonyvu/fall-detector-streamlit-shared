
import streamlit as st
from ultralytics import YOLO
from utils_ext_camera import play_webcam
from PIL import Image
import click


@click.command()
@click.option('--model-filename', default='best.pt')
def cli(model_filename):

    st.set_page_config(page_title="Fall Detector with streamlit",
                   page_icon='camera_icon.png'
                #    layout='wide'
                   )
    st.header('Fall Detector - Model Explanation', divider='rainbow')

    # Display the title image
    print(model_filename)
    model = YOLO(model_filename)

    st.subheader(':camera: Live camera')

    with st.expander('Open to select a confidence threshold'):
        conf = st.slider(label='Confidence threshold',
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        )
        st.write(f'Selected: {conf}')


    play_webcam(conf=conf, model=model)

    col1, col2, col3 = st.columns((1, 5, 1))
    # col1, col2 = st.columns((3, 6))

    col2.write('<span style="color:red;background-color:#183b80;padding: 10px;font-size: 24px;">falling : red</span><span style="color:orange;background-color:#183b80;padding: 10px;font-size: 24px;">sitting : orange</span><span style="color:pink;background-color:#183b80;padding: 10px;font-size: 24px;">standing : pink</span>',
              unsafe_allow_html=True)

if __name__ == '__main__':
    cli()
