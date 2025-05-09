from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-seg.pt")

# Export to tflite int8
model.export(
    task = "segment",
    data="coco128-seg.yaml",
    format = "tflite",
    int8 = True,
    nms = False,
    device = "cpu",
    imgsz = 640,
    batch = 1,
)
