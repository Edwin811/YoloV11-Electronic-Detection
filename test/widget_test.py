from ultralytics import YOLO
model = YOLO('best.pt')
results = model('some_test_image.jpg', imgsz=416)
results.show()
