import os
import cv2
import torch
import numpy as np
from model.deeplabv3plus import DeeplabV3Plus
from model.unet import ResNetUNet
from config import Config
from utils.image_process import crop_resize_data
from utils.process_labels import decode_color_labels
import matplotlib.pyplot as plt

#os.environ["CUDA_VISIBLE_DEVICES"] = ""
#for dvi in range(torch.cuda.device_count()):
#    print(torch.cuda.get_device_name(dvi))

print("hello pytorch{}".format(torch.__version__))
print(torch.cuda.is_available())


device_id = 0
predict_net = 'deeplabv3p'
nets = {'deeplabv3p': DeeplabV3Plus, 'unet': ResNetUNet}


def load_model(model_path):

    lane_config = Config()
    net = nets[predict_net](lane_config)
    net.eval()
    if torch.cuda.is_available():
        net = net.cuda(device=device_id)
        map_location = 'cuda:%d' % device_id
    else:
        map_location = 'cpu'

    model_param = torch.load(model_path, map_location=map_location)['state_dict']
    model_param = {k.replace('module.', ''):v for k, v in model_param.items()}
    net.load_state_dict(model_param)
    return net


def img_transform(img):
    img = crop_resize_data(img)
    img = np.transpose(img, (2, 0, 1))              # H,W 3换成 3 H W
    img = img[np.newaxis, ...].astype(np.float32)   # 增加一个biachsize的维度
    img = torch.from_numpy(img.copy())
    if torch.cuda.is_available():
        img = img.cuda(device=device_id)
    return img


def get_color_mask(pred):
    pred = torch.softmax(pred, dim=1)
    pred_heatmap = torch.max(pred, dim=1)
    # 1,H,W,C
    pred = torch.argmax(pred, dim=1)
    pred = torch.squeeze(pred)
    pred = pred.detach().cpu().numpy()          # 此处对应0-7的训练label
    pred = decode_color_labels(pred)
    pred = np.transpose(pred, (1, 2, 0))
    return pred 


def main():
    model_dir = 'logs'
    test_dir = 'test_example'
    model_path = os.path.join(model_dir,  'finalNet.pth.tar')
    net = load_model(model_path)

    img_path = os.path.join(test_dir, 'test.jpg')
    img = cv2.imread(img_path)
    img = img_transform(img)                      # 训练的时候的图片格式和和预测的时候应该一样。

    pred = net(img)
    color_mask = get_color_mask(pred)              # 将训练的图片解吗成原始的分割类型
    cv2.imwrite(os.path.join(test_dir, 'color_mask.png'), color_mask)


    # image = cv2.imread( test_dir+"color_mask.png", cv2.IMREAD_GRAYSCALE)
    # print(np.unique(image))
    # plt.imshow(color_mask)
    # plt.show()



if __name__ == '__main__':
    main()

