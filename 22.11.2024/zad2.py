
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('images/4.jpg') 
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width, _ = image_rgb.shape


gray_image = np.zeros((height, width), dtype=np.uint8)
gray_image2 = np.zeros((height, width), dtype=np.uint8)


for y in range(height):
    for x in range(width):
        r, g, b = image_rgb[y, x]  
        gray1 = (0.299*r +0.587*g + 0.114*b)  
        gray_image[y, x] = gray1
        gray2 = round(((r+g+b)/3))
        gray_image2[y, x] = gray2

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(gray_image, cmap='gray')
plt.title("Grayscale (0.299R + 0.587G + 0.114B)")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(gray_image2, cmap='gray')
plt.title("Grayscale (round((r+g+b)/3))")
plt.axis('off')


plt.show()