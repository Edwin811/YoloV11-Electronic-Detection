import tensorflow as tf
import numpy as np
import cv2

interpreter = tf.lite.Interpreter(model_path="best_float32.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Baca gambar dan resize ke 416x416
img = cv2.imread("test.jpeg")
img_resized = cv2.resize(img, (416, 416))
input_data = np.expand_dims(img_resized / 255.0, axis=0).astype(np.float32)

# Jalankan inferensi
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Ambil hasil
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data.shape)  # (1, 7, 3549) misalnya

# (opsional) Cetak confidence
for i in range(output_data.shape[2]):
    conf = output_data[0][0][i]
    if conf > 0.5:
        print(f"Deteksi {i} → confidence={conf}")
