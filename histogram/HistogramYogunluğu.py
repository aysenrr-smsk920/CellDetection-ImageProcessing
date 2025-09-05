import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntüyü oku
image_path = "C:/Users/AYSENUR SIMSEK/Desktop/YZ-VOST-LP/hucre_goruntuleri/contrast/1.jpeg"  # Kendi yolunuzu yazın
image = cv2.imread(image_path)

# BGR'den RGB'ye çevir (OpenCV BGR formatında okur, matplotlib RGB bekler)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Renk kanallarını ayır
r, g, b = cv2.split(image_rgb)

# Histogramları hesapla
r_hist = cv2.calcHist([r], [0], None, [256], [0, 256])
g_hist = cv2.calcHist([g], [0], None, [256], [0, 256])
b_hist = cv2.calcHist([b], [0], None, [256], [0, 256])

# Histogramları çiz
plt.figure(figsize=(10, 6))
plt.title("Renk Kanallarının Histogramları")
plt.xlabel("Değer")
plt.ylabel("Yoğunluk")
plt.xlim(0, 256)

plt.plot(r_hist, color='red', label='Kırmızı Kanal')
plt.plot(g_hist, color='green', label='Yeşil Kanal')
plt.plot(b_hist, color='blue', label='Mavi Kanal')

plt.legend()
plt.show()
