import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer
import numpy as np
import av

def play_webcam(conf, model):
    """
    Plays a webcam stream on cloud. Detects Objects in real-time using the YOLO object detection model.

    Returns:
        None

    Raises:
        None
    """

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")

        orig_h, orig_w = image.shape[0:2]

        print(f"Current Frame Resolution: Width = {orig_w}, Height = {orig_h}")

        width = 720  # Set the desired width for processing

        # cv2.resize used in a forked thread may cause memory leaks
        processed_image = np.asarray(Image.fromarray(image).resize((width, int(width * orig_h / orig_w))))

        if model is not None:
            # Perform object detection using YOLO model
            res = model.predict(processed_image, conf=conf)

            # Plot the detected objects on the video frame
            res_plotted = res[0].plot()


        return av.VideoFrame.from_ndarray(res_plotted, format="bgr24")


    webrtc_streamer(
        key="example",
        video_frame_callback = video_frame_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )
