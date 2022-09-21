import json

import matplotlib.pyplot as plt
import torch
from model import vgg
from PIL import Image
from torchvision import transforms


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    )
    img_path = "../function_practice/sunflower.jpg"
    img = Image.open(img_path)
    img = data_transform(img)  # tensor(3,224,224)
    img = torch.unsqueeze(img, dim=0)  # tensor(1,3,224,224),unsquzee拓展维度，在指定维度插入1
    model = vgg(mode_name='vgg16', class_num=5)
    weights_path = "AlexNet.pth"
    model.load_state_dict(torch.load(weights_path))
    with open('class_index.json') as f:
        class_indict = json.load(f)
    model.eval() #开启eval模式，关闭了dropout
    with torch.no_grad():  # ？？？
        output = model(img) #tensor(1,5)
        output = torch.squeeze(model(img)) #tensor(5) 压缩掉batch维度
        predict = torch.softmax(output, dim=0) #softmax转换为概率分布
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    print(print_res)
    for i in range(len(predict)):
        print("class: {:10}  prob:{:.3}".format(class_indict[str(i)], predict[i].numpy()))


if __name__ == '__main__':
    main()
