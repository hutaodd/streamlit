import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

st.markdown("# 面部检测")
st.sidebar.markdown("# 面部检测")


uploaded_files = st.sidebar.file_uploader("上传面部检测图片", accept_multiple_files=True)
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
        face_engine = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        # 调用
        faces = face_engine.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)
        # 绘制画框
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # For static images:
        st.image(img)