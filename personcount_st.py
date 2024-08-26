import streamlit as st
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import io
import torch

model = YOLO("best.pt")
img_proc = False

@st.cache_resource
def process_image(img, conf, iou):
   img = Image.open(img)
   results = model.predict(img, conf=conf, iou=iou)
   counts = len(results[0].boxes)
   pred_img = results[0].plot()
   pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
   return pred_img, counts

st.title("Simple YoloV8 Person Counter")

uploaded_img = st.sidebar.file_uploader("Upload an image\n", type=['png', 'jpg', 'jpeg'])
conf = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
iou = st.sidebar.slider("IoU Threshold",  0.0, 1.0, 0.7, 0.05)

if uploaded_img is not None:
   img_type = uploaded_img.type
   print(img_type)
   img_proc = True
   
if st.sidebar.button("Detect!", type="primary") and img_proc:
   pred_img, counts = process_image(uploaded_img, conf, iou)
   st.image(pred_img)
   st.text(f'Counts: {counts} person(s)')

   pred_img = Image.fromarray(pred_img)
   img_byte_array = io.BytesIO()
   pred_img.save(img_byte_array, format='PNG')
   img_byte_array = img_byte_array.getvalue()

   if st.download_button(label="Download Image", data=img_byte_array, file_name=f'detected_image.{img_type.replace("image/","")}',
                        mime=img_type):
      st.success("Downloaded successfully!")