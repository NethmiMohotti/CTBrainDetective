# -*- coding: utf-8 -*-
"""Model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wgoTscw3JaFGZZsKienXETYj_OJgB7CG
"""

!pip install ultralytics -q
!pip install pyyaml -q

!pip show ultralytics

import torch
torch.__version__

from ultralytics import YOLO
##training model
model = YOLO("yolov8m.pt")

!touch data.yaml

from google.colab import drive
drive.mount('/content/drive')

model.train(data ="/content/data.yaml" , epochs = 5)

"""Best performing model/ fine tune model"""

infer = YOLO ("/content/runs/detect/train4/weights/best.pt")

infer.predict("/content/drive/MyDrive/FYP/Development/CT_Brain_hemorrhage/train/images", save = True, save_txt = True)
