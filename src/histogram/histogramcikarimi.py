import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# 📂 Klasör yolu
folder_path = r'C:\Users\AYSENUR SIMSEK\Desktop\YZ-VOST-LP\hucre_goruntuleri\contrast\processedCanny'

# 🔹 Klasördeki tüm görüntüleri al
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

# 📊 R, G, B histogramlarını sıfırla
total_hist_r = np.zeros((256, 1), dtype=np.float32)
total_hist_g = np.zeros((256, 1), dtype=np.float32)
total_hist_b = np.zeros((256, 1), dtype=np.float32)

for image_file in image_files:
    # 🖼️ Görüntüyü oku (Renkli)
    image_path = os.path.join(folder_path, image_file)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Hata: {image_file} yüklenemedi!")
        continue

    # 📊 Her kanal için histogram hesapla
    hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])  # Red
    hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])  # Green
    hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])  # Blue

    # 📈 Histogramları topla
    total_hist_r += hist_r
    total_hist_g += hist_g
    total_hist_b += hist_b

# 🔹 Histogram Grafiğini Çiz
plt.figure(figsize=(8, 5))
plt.plot(total_hist_r, color='red', label='Red')
plt.plot(total_hist_g, color='green', label='Green')
plt.plot(total_hist_b, color='blue', label='Blue')
plt.title("Tüm Görüntülerin Birleşik RGB Histogramı")
plt.xlabel("Piksel Yoğunluğu (0-255)")
plt.ylabel("Toplam Piksel Sayısı")
plt.xlim([0, 256])
plt.legend()
plt.grid(True)

# 📊 Histogramı göster
plt.show()
