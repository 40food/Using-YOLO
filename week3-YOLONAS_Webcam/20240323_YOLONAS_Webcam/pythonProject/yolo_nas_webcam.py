import cv2
import torch.cuda
from super_gradients.training import models
from super_gradients.common.object_names import Models

model=models.get(Models.YOLOX_N,pretrained_weights='coco')
model=model.to("cuda" if torch.cuda.is_available() else 'cpu')
model.predict_webcam()