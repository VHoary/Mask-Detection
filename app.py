import os
from ultralytics import YOLO
from tkinter import Tk
from tkinter.filedialog import askopenfilename

root = Tk()
root.withdraw()
file_path = askopenfilename()
model = YOLO("C:/Users/fzopr/Documents/Code/Python/DIP Project/App/runs/detect/train/weights/best.pt")

results = model(source=file_path, show=True, conf=0.5, save=True)
print("ผลลัพธ์นั้นจะถูกเก็บไว้ใน โฟลเดอร์ที่ติดตั้ง/runs/detect/predict")