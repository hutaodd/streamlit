

import mediapipe as mp
import streamlit as st
import time
from io import BytesIO
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
st.markdown("# 姿态检测")
st.sidebar.markdown("# 姿态检测")

# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose

# Setup the Pose function for images - independently for the images standalone processing.
pose_image = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Setup the Pose function for videos - for video processing.
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7,
                          min_tracking_confidence=0.7)

# Initialize mediapipe drawing class - to draw the landmarks points.
mp_drawing = mp.solutions.drawing_utils



def detectPose(image_pose, pose, draw=False, display=False):
    
    original_image = image_pose.copy()
    
    image_in_RGB = cv2.cvtColor(image_pose, cv2.COLOR_BGR2RGB)
    
    resultant = pose.process(image_in_RGB)

    if resultant.pose_landmarks and draw:    

        mp_drawing.draw_landmarks(image=original_image, landmark_list=resultant.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255),
                                                                               thickness=3, circle_radius=3),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(49,125,237),
                                                                               thickness=2, circle_radius=2))

    if display:
        
        
      st.image(original_image) 
    


uploaded_files = st.sidebar.file_uploader("上传姿态识别图片", accept_multiple_files=True)
for uploaded_file in uploaded_files:
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        # 将字节数据转化成字节流
        bytes_data = BytesIO(bytes_data)
        # Image.open()可以读字节流
        capture_img = Image.open(bytes_data)
        # cv2使用arry格式图片流，使用numpy转换
        img = np.array(capture_img)
        # 面部检测引擎
        #face_engine = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        # 调用
        #faces = face_engine.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)
        # 绘制画框
        # for (x, y, w, h) in faces:
        #  img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # For static images:

        output = img

        detectPose(output, pose_image, draw=True, display=True)
        # 显示图片
        



