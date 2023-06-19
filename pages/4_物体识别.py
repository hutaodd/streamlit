import streamlit as st
import cv2

import numpy as np

from io import BytesIO
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

st.markdown("# 物体识别")
st.sidebar.markdown("# 物体识别")


MARGIN = 55 # pixels
ROW_SIZE = 4  # pixels
FONT_SIZE = 4
FONT_THICKNESS = 4
TEXT_COLOR = (255, 241, 0)  # 
BOX_COLOR = (50, 113, 174)


uploaded_files = st.sidebar.file_uploader("上传物体识别图片", accept_multiple_files=True)
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

        BaseOptions = mp.tasks.BaseOptions
        ObjectDetector = mp.tasks.vision.ObjectDetector
        ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path='./efficientdet.tflite'),
            max_results=5,
            running_mode=VisionRunningMode.IMAGE)

        with ObjectDetector.create_from_options(options) as detector:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
            detection_result = detector.detect(mp_image)
            image_copy = np.copy(mp_image.numpy_view())
            
            for detection in detection_result.detections:
                # Draw bounding_box
                bbox = detection.bounding_box
                start_point = bbox.origin_x, bbox.origin_y
                end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
                cv2.rectangle(image_copy, start_point, end_point, BOX_COLOR, 3)
            
                # Draw label and score
                category = detection.categories[0]
                category_name = category.category_name
                probability = round(category.score, 2)
                result_text = category_name + ' (' + str(probability) + ')'
                text_location = (MARGIN + bbox.origin_x,
                                 MARGIN + ROW_SIZE + bbox.origin_y)
                cv2.putText(image_copy, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                            FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
        st.image(image_copy)