import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntüyü oku
image = cv2.imread('C:/Users/AYSENUR SIMSEK/Desktop/YZ-VOST-LP/hucre_goruntuleri/contrast/2.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Görüntüyü 2D diziye dönüştür
pixel_values = image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# K-Means kriterleri ve küme sayısı
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 4  # Küme sayısı
_, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Merkezleri uint8'e dönüştür
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]
segmented_image = segmented_data.reshape(image.shape)

# Görüntüyü göster
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Orijinal Görüntü')
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f'K-Means Segmentasyonu (k={k})')
plt.imshow(segmented_image)
plt.axis('off')

plt.show()
