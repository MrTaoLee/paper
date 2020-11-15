import matplotlib.pyplot as plt
import cv2
import numpy as np


image = cv2.imread("timg.png", cv2.IMREAD_GRAYSCALE)
print(np.unique(image))
plt.imshow(image)
plt.show()