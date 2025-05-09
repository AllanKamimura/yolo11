from runtime.yolo import YOLOv8TFLite
import cv2

model_path = "./yolo11n-seg_full_integer_quant.tflite"

iou = 0.45
confidence = 0.25
metadata = "./metadata.yaml"
img = "./bus.jpg"

detector = YOLOv8TFLite(model_path, confidence, iou, metadata)
result = detector.detect(img)

print(result[0].boxes)
cv2.imwrite("output.jpg", result[0].plot())
