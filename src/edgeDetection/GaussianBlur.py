import cv2
import numpy as np

# Görüntüyü yükle
image = cv2.imread(r'C:\Users\AYSENUR SIMSEK\Desktop\YZ-VOST-LP\hucre_goruntuleri\3.jpeg')

# Sabit boyutlarda yeniden boyutlandır (burada %50 küçültüyoruz)
image_resized = cv2.resize(image, (int(image.shape[1] *1), int(image.shape[0] * 1)))

# Gaussian Blur Uygula
gaussian_blur = cv2.GaussianBlur(image_resized, (5 ,5), 0)

# Median Blur Uygula
median_blur = cv2.medianBlur(image_resized, 5)

# Sonuçları Göster
cv2.imshow('Resized Image', image_resized)
cv2.imshow('Gaussian Blur', gaussian_blur)
cv2.imshow('Median Blur', median_blur)

cv2.waitKey(0)
cv2.destroyAllWindows()
