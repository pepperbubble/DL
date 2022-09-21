'''
ImageFolder假设所有文件夹按照类别分类，文件夹名称为类名
'''
import os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
image_path = os.path.join(data_root, "data_set", "flower_data")
dataset = datasets.ImageFolder(root=os.path.join(image_path, "flower_photos"))

# 类别转换字典
print(dataset.class_to_idx)

# 取单张图片
print(dataset[0][1])
plt.imshow(dataset[0][0])
plt.show()

# 加入transform
'''
随机裁剪是一种数据扩充方法，可以提高模型精度和稳定性
'''
transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
)
dataset = datasets.ImageFolder(root=os.path.join(image_path, "flower_photos"), transform=transform)
print(dataset[0][1])
print(dataset[0][0])
