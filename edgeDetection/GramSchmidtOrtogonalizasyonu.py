import cv2
import numpy as np
import matplotlib.pyplot as plt

def gram_schmidt_orthonormalization(image):
    """
    Gram-Schmidt sürecini uygular ve görüntü bileşenlerini ayrıştırır.
    """
    # Görüntünün yüklendiğini kontrol et
    if image is None:
        raise ValueError("Görüntü dosyası yüklenemedi. Dosya yolunu kontrol edin.")

    # Görüntüyü float türüne çevir ve 0-1 aralığına getir
    image = image.astype(np.float64) / 255.0

    # Renk kanallarını ayır
    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # Gram-Schmidt ile ortonormal bileşenler oluştur
    U1 = R
    if np.sum(U1 * U1) == 0:
        raise ValueError("U1 vektörü sıfır, Gram-Schmidt işlemi başarısız.")

    U2 = G - (np.sum(G * U1) / np.sum(U1 * U1)) * U1
    if np.sum(U2 * U2) == 0:
        raise ValueError("U2 vektörü sıfır, Gram-Schmidt işlemi başarısız.")

    U3 = B - (np.sum(B * U1) / np.sum(U1 * U1)) * U1 - (np.sum(B * U2) / np.sum(U2 * U2)) * U2
    if np.sum(U3 * U3) == 0:
        raise ValueError("U3 vektörü sıfır, Gram-Schmidt işlemi başarısız.")

    # Ortonormal bileşenleri normalize et
    U1 /= np.linalg.norm(U1) if np.linalg.norm(U1) != 0 else 1
    U2 /= np.linalg.norm(U2) if np.linalg.norm(U2) != 0 else 1
    U3 /= np.linalg.norm(U3) if np.linalg.norm(U3) != 0 else 1

    # NaN ve Inf değerleri sıfıra çevir
    U1 = np.nan_to_num(U1)
    U2 = np.nan_to_num(U2)
    U3 = np.nan_to_num(U3)

    return np.stack([U1, U2, U3], axis=-1)  # Ayrıştırılmış bileşenleri birleştir

# Görüntüyü oku
image_path = r'C:\Users\AYSENUR SIMSEK\Desktop\YZ-VOST-LP\hucre_goruntuleri\1.jpeg'
image = cv2.imread(image_path)

# Dosyanın gerçekten var olup olmadığını kontrol et
if image is None:
    raise FileNotFoundError(f"Görüntü dosyası bulunamadı: {image_path}")

# OpenCV BGR formatında okuduğu için RGB'ye çevir
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Gram-Schmidt uygulama
processed_image = gram_schmidt_orthonormalization(image)

# Görüntüyü 0-255 aralığına getirerek uint8 formatına dönüştür
processed_image_uint8 = np.clip(processed_image * 255, 0, 255).astype(np.uint8)

# Gri tonlamaya çevir ve eşikleme uygula
gray_image = cv2.cvtColor(processed_image_uint8, cv2.COLOR_RGB2GRAY)
_, binary_mask = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)

# Sonuçları göster
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Orijinal Görüntü")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(processed_image_uint8)  # Dönüştürülmüş görüntü
plt.title("Gram-Schmidt Ayrıştırılmış")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(binary_mask, cmap="gray")
plt.title("Eşiklenmiş Görüntü")
plt.axis("off")

plt.show()
