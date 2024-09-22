import streamlit as st
import cv2
import time
import torch
from PIL import Image as PILImage
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO
import numpy as np

# File uploader
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if uploaded_file.type in ["image/jpeg", "image/png"]:
        if st.button("Get"):
            # Initialize CLIP and YOLO models
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
            yolo_model = YOLO("yolov10n.pt").cuda()

            labels = ["fighting", "running", "walking", "sitting", "talking"]

            # Read uploaded file into OpenCV format
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # CLIP processing
            start_time_clip = time.time()

            image_pil = PILImage.fromarray(image_rgb)
            inputs = clip_processor(images=image_pil, return_tensors="pt", padding=True, truncation=True)
            text_inputs = clip_processor(text=labels, return_tensors="pt", padding=True, truncation=True)

            with torch.no_grad():
                image_features = clip_model.get_image_features(**inputs)
                text_features = clip_model.get_text_features(**text_inputs)

            similarity = torch.cosine_similarity(image_features, text_features)

            # Calculate similarity scores for each label
            similarity_scores = similarity.tolist()
            predicted_label_index = similarity.argmax().item()
            predicted_label = labels[predicted_label_index]
            predicted_score = similarity_scores[predicted_label_index]

            end_time_clip = time.time()
            clip_time = end_time_clip - start_time_clip

            start_time_yolo = time.time()

            results = yolo_model(image)

            for result in results:
                boxes = result.boxes.xyxy
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box[:4])
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            end_time_yolo = time.time()
            yolo_time = end_time_yolo - start_time_yolo
            st.write(f"Time taken for YOLO detection: {yolo_time:.4f} seconds")
            st.write(f"Time taken for CLIP classification: {clip_time:.4f} seconds")
            st.write(f"Total time taken: {yolo_time + clip_time:.4f} seconds")

            st.write(f"Predicted Label: **{predicted_label}**")
            st.write(f"CLIP Similarity Score: **{predicted_score:.4f}**")
            st.write(f"CLIP Confidence: **{predicted_score * 100:.2f}%**")

            cv2.putText(image, f"Frame classification: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 2, cv2.LINE_AA)

            st.image(image, caption=f"Result: {predicted_label}", use_column_width=True)



    else:
        st.write("Only images are allowed (jpg, jpeg, png).")
