'''
随机裁剪指定大小的区域
'''
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
img1=Image.open("sunflower.jpg")
img2=transforms.RandomResizedCrop(224)(img1)
img3=transforms.RandomResizedCrop(224)(img1)
img4=transforms.RandomResizedCrop(224)(img1)
plt.subplot(2,2,1),plt.imshow(img1),plt.title("original")
plt.subplot(2,2,2),plt.imshow(img2),plt.title("crop1")
plt.subplot(2,2,3),plt.imshow(img3),plt.title("crop2")
plt.subplot(2,2,4),plt.imshow(img4),plt.title("crop3")
plt.show()