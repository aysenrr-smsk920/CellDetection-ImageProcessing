import cv2
import numpy as np

# Görüntüyü yükle
image = cv2.imread(r'C:\Users\AYSENUR SIMSEK\Desktop\YZ-VOST-LP\hucre_goruntuleri\kontrast\5.jpg')

# Görüntüyü yeniden boyutlandır (isteğe bağlı)
image_resized = cv2.resize(image, (int(image.shape[1] * 0.3), int(image.shape[0] * 0.3)))

# Laplacian Kenar Tespiti
laplacian = cv2.Laplacian(image_resized, cv2.CV_64F)

# Laplacian sonucu mutlak değere dönüştürülür (negatif değerleri giderir)
laplacian_abs = np.uint8(np.absolute(laplacian))

# Sonuçları Göster
cv2.imshow('Original Image', image_resized)
cv2.imshow('Laplacian Edge Detection', laplacian_abs)

cv2.waitKey(0)
cv2.destroyAllWindows()
