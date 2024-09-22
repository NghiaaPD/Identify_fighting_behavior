import cv2
import time
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO
import matplotlib.pyplot as plt

clip_model = CLIPModel.from_pretrained("./model")
clip_processor = CLIPProcessor.from_pretrained("./model")

yolo_model = YOLO("yolov10n.pt")

labels = ["fighting", "running", "walking", "sitting", "talking"]

image_path = "./Vietnamese_oanh_bo_me/dibo.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

start_time_clip = time.time()

image_pil = Image.fromarray(image_rgb)
inputs = clip_processor(images=image_pil, return_tensors="pt", padding=True, truncation=True)
text_inputs = clip_processor(text=labels, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    image_features = clip_model.get_image_features(**inputs)
    text_features = clip_model.get_text_features(**text_inputs)

similarity = torch.cosine_similarity(image_features, text_features)
predicted_label_index = similarity.argmax().item()
predicted_label = labels[predicted_label_index]

end_time_clip = time.time()
clip_time = end_time_clip - start_time_clip

print(f"Time taken for CLIP prediction: {clip_time:.4f} seconds")

start_time_yolo = time.time()

results = yolo_model(image_path)

for result in results:
    boxes = result.boxes.xyxy
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

end_time_yolo = time.time()
yolo_time = end_time_yolo - start_time_yolo
print(f"Time taken for YOLO prediction: {yolo_time:.4f} seconds")

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title(f'Predicted Action: {predicted_label}')
plt.axis('off')
plt.show()
