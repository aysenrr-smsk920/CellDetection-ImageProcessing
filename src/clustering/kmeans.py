import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Görüntüyü oku ve RGB'ye çevir
image = cv2.imread('C:/Users/AYSENUR SIMSEK/Desktop/YZ-VOST-LP/hucre_goruntuleri/contrast/2.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 2. Görüntüyü 2D piksel dizisine dönüştür
pixel_values = image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# 3. K-Means parametreleri
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 4
_, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)
labels = labels.flatten()

# Küme merkezlerini analiz et
print("Küme Merkezleri (RGB):")
for i, c in enumerate(centers):
    print(f"Küme {i}: {c}")

# Küme merkezlerini daha net belirlemek için parlaklık yerine genel renk dağılımını kullan
def identify_clusters(centers):
    cluster_info = {}

    # Renk dağılımına göre sıralama (arka plan genelde en karanlık olmalı)
    brightness = [np.sum(c) for c in centers]
    bg_index = np.argmin(brightness)  # En az parlaklığa sahip olan arka plan olmalı

    others = [i for i in range(len(centers)) if i != bg_index]

    # En yüksek yeşil kanal -> canlı hücre
    green_index = max(others, key=lambda i: centers[i][1])
    others.remove(green_index)

    # Geriye kalan ölü hücre (yüksek kırmızı ve yeşil)
    if others:
        dead_index = others[0]
    else:
        dead_index = None

    cluster_info['background'] = bg_index
    cluster_info['alive'] = green_index
    cluster_info['dead'] = dead_index
    return cluster_info


cluster_map = identify_clusters(centers)

# 5. Renkli segmentasyon görüntüsü oluştur
custom_colors = {
    'alive': [0, 255, 0],  # Canlı hücre: yeşil
    'dead': [255, 140, 0],  # Ölü hücre: turuncu
    'background': [0, 0, 0]  # Arka plan: siyah
}

colored_result = np.zeros_like(pixel_values)
for i, label in enumerate(labels):
    if label == cluster_map['alive']:
        colored_result[i] = custom_colors['alive']
    elif cluster_map['dead'] is not None and label == cluster_map['dead']:
        colored_result[i] = custom_colors['dead']
    else:
        colored_result[i] = custom_colors['background']

colored_image = colored_result.reshape(image.shape)

# 6. Hücre sayımı için maskeler
alive_mask = (labels == cluster_map['alive']).astype(np.uint8).reshape(image.shape[0], image.shape[1])
dead_mask = (labels == cluster_map['dead']).astype(np.uint8).reshape(image.shape[0], image.shape[1]) if cluster_map[
                                                                                                            'dead'] is not None else np.zeros_like(
    alive_mask)

# Gürültüyü azalt (morfolojik açma işlemi)
kernel = np.ones((3, 3), np.uint8)
alive_mask = cv2.morphologyEx(alive_mask, cv2.MORPH_OPEN, kernel)
dead_mask = cv2.morphologyEx(dead_mask, cv2.MORPH_OPEN, kernel)

# 7. Hücreleri say (connected components)
num_alive, _ = cv2.connectedComponents(alive_mask)
num_dead, _ = cv2.connectedComponents(dead_mask)

# Arka planı çıkar
alive_count = num_alive - 1  # Arka planı saymamak için -1
dead_count = num_dead - 1  # Arka planı saymamak için -1

print(f"\nCanlı hücre sayısı: {alive_count}")
print(f"Ölü hücre sayısı: {dead_count}")

# 8. Görselleri göster
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Orijinal Görüntü")
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("K-Means Renkli Segmentasyon")
plt.imshow(colored_image)
plt.axis('off')
plt.show()
