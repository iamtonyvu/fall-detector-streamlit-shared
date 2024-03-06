import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer
import numpy as np
import av
import math
import cv2

def play_webcam(conf, model):
    """
    Plays a webcam stream on cloud. Detects Objects in real-time using the YOLO object detection model.

    Returns:
        None

    Raises:
        None
    """

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        try:
            print("video_frame_callback")

            classNames = ["Fall", "Standing", "Sit"]
            classColor = [( 176, 58, 46 ), (46, 204, 113), (241, 196, 15)]
            image = frame.to_ndarray(format="bgr24")

            if model is not None:
                # Perform object detection using YOLO model
                pipeline_outputs = model(images=image)
                #print(pipeline_outputs)

                # Plot the detected objects on the video frame
                for r in pipeline_outputs:
                    boxes = r.boxes
                    scores = r.scores
                    labels = r.labels
                    for i in range(len(boxes)):
                        #print(boxes)
                        # bounding box
                        x1, y1, x2, y2 = int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3]) # convert to int values

                        # confidence
                        confidence = math.ceil((scores[i]*100))/100
                        #print("Confidence --->",confidence)

                        # class name
                        cls = int(float(labels[i]))
                        #print("Class name -->", classNames[cls])

                        # put box in cam
                        cv2.rectangle(image, (x1, y1), (x2, y2), classColor[cls], 3)


                        # object details
                        org = [x1, y1]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 1
                        thickness = 2

                        cv2.putText(image, f"{classNames[cls]} {confidence}", org, font, fontScale, classColor[cls], thickness)

            return av.VideoFrame.from_ndarray(image, format="bgr24")
        except Exception as e:
            print(e)


    webrtc_streamer(
        key="example",
        video_frame_callback = video_frame_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )
