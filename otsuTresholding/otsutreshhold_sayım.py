import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntüyü oku
image = cv2.imread('C:/Users/AYSENUR SIMSEK/Desktop/YZ-VOST-LP/hucre_goruntuleri/contrast/2.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Kırmızı, yeşil ve mavi kanalını ayır
red_channel = image[:, :, 0]
green_channel = image[:, :, 1]
blue_channel = image[:, :, 2]

# Ölü hücreleri kırmızı kanaldan tespit et (örneğin, kırmızı kanalın değerlerinin yüksek olduğu yerler)
_, red_threshold = cv2.threshold(red_channel, 100, 255, cv2.THRESH_BINARY)

# Canlı hücreleri yeşil ve mavi kanaldan tespit et (örneğin, yeşil ve mavi kanalların yüksek olduğu yerler)
_, green_threshold = cv2.threshold(green_channel, 100, 255, cv2.THRESH_BINARY)
_, blue_threshold = cv2.threshold(blue_channel, 100, 255, cv2.THRESH_BINARY)

# Kırmızı ve yeşil/mavi tespit sonuçlarının birleşimi
alive_cells = cv2.bitwise_and(green_threshold, blue_threshold)  # Canlı hücrelerin birleşimi
dead_cells = red_threshold  # Ölü hücreler kırmızı kanalında

# Kontur tespiti ile hücre sayma
def count_cells(threshold_image):
    contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)

# Ölü ve canlı hücrelerin sayısını al
dead_count = count_cells(dead_cells)
alive_count = count_cells(alive_cells)

#print(f"Ölü hücre sayısı: {dead_count}")
#print(f"Canlı hücre sayısı: {alive_count}")

# Görüntüleri göster
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title('Orijinal Görüntü')
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Ölü Hücreler (Kırmızı Kanal)')
plt.imshow(dead_cells, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Canlı Hücreler (Yeşil & Mavi Kanal)')
plt.imshow(alive_cells, cmap='gray')
plt.axis('off')

plt.show()