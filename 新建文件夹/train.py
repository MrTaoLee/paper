from tqdm import tqdm
import torch
import os
import shutil
from utils.metric import compute_iou
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.image_process import LaneDataset, ImageAug, DeformAug
from utils.image_process import ScaleAug, CutOut, ToTensor
from utils.loss import MySoftmaxCrossEntropyLoss
from model.deeplabv3plus import DeeplabV3Plus
from config import Config


# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

device_list = [0]   #改了


def train_epoch(net, epoch, dataLoader, optimizer, trainF, config):
    net.train()
    total_mask_loss = 0.0
    dataprocess = tqdm(dataLoader)
    for batch_item in dataprocess:
        image, mask = batch_item['image'], batch_item['mask']
        if torch.cuda.is_available():
            image, mask = image.cuda(device=device_list[0]), mask.cuda(device=device_list[0])
        optimizer.zero_grad()
        image = image.type(torch.cuda.FloatTensor)  #改
        out = net(image)

        mask = mask.long()  #改
        mask_loss = MySoftmaxCrossEntropyLoss(nbclasses=config.NUM_CLASSES)(out, mask)
        total_mask_loss += mask_loss.item()
        mask_loss.backward()
        optimizer.step()
        dataprocess.set_description_str("epoch:{}".format(epoch))
        dataprocess.set_postfix_str("mask_loss:{:.4f}".format(mask_loss.item()))
    trainF.write("Epoch:{}, mask loss is {:.4f} \n".format(epoch, total_mask_loss / len(dataLoader)))
    trainF.flush()


def test(net, epoch, dataLoader, testF, config):
    net.eval()
    total_mask_loss = 0.0
    dataprocess = tqdm(dataLoader)
    result = {"TP": {i:0 for i in range(8)}, "TA":{i:0 for i in range(8)}}
    for batch_item in dataprocess:
        image, mask = batch_item['image'], batch_item['mask']
        if torch.cuda.is_available():
            image, mask = image.cuda(device=device_list[0]), mask.cuda(device=device_list[0])

        image = image.type(torch.cuda.FloatTensor)  # 改
        mask = mask.long()  # 改

        out = net(image)
        print(out.shape,out[0].shape,mask.shape)

        mask_loss = MySoftmaxCrossEntropyLoss(nbclasses=config.NUM_CLASSES)(out, mask)
        total_mask_loss += mask_loss.detach().item()

        pred = torch.argmax(F.softmax(out, dim=1), dim=1)


        result = compute_iou(pred, mask, result)
        dataprocess.set_description_str("epoch:{}".format(epoch))
        dataprocess.set_postfix_str("mask_loss:{:.4f}".format(mask_loss))
    testF.write("Epoch:{}".format(epoch))
    for i in range(8):
        result_string = "{}: {:.4f} \n".format( i, result["TP"][i]/result["TA"][i] )
        print(result_string)
        testF.write(result_string)
    testF.write("Epoch:{}, mask loss is {:.4f} \n".format(epoch, total_mask_loss / len(dataLoader)))
    testF.flush()


def adjust_lr(optimizer, epoch):
    if epoch == 0:
        lr = 1e-4
    elif epoch == 5:
        lr = 1e-4
    elif epoch == 20:
        lr = 1e-5
    elif epoch == 40:
        lr = 1e-5
    else:
        return
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    lane_config = Config()
    if os.path.exists(lane_config.SAVE_PATH):
        shutil.rmtree(lane_config.SAVE_PATH)            #shutil.rmtree() 表示递归删除文件夹下的所有子文件夹和子文件
    os.makedirs(lane_config.SAVE_PATH, exist_ok=True)   #递归创建目录
    trainF = open(os.path.join(lane_config.SAVE_PATH, "train.csv"), 'w')
    testF = open(os.path.join(lane_config.SAVE_PATH, "test.csv"), 'w')
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_dataset = LaneDataset("train.csv", transform=transforms.Compose([ImageAug(), DeformAug(),
                                                                              ScaleAug(), CutOut(32, 0.5), ToTensor()]))
    train_data_batch = DataLoader(train_dataset, batch_size=2*len(device_list), shuffle=True, drop_last=True, **kwargs)  #训练生成器

    val_dataset = LaneDataset("val.csv", transform=transforms.Compose([ToTensor()]))
    val_data_batch = DataLoader(val_dataset, batch_size=1*len(device_list), shuffle=False, drop_last=False, **kwargs)    #验证生成器

    net = DeeplabV3Plus(lane_config)
    if torch.cuda.is_available():
        net = net.cuda(device=device_list[0])
        net = torch.nn.DataParallel(net, device_ids=device_list)
    # optimizer = torch.optim.SGD(net.parameters(), lr=lane_config.BASE_LR,
    #                             momentum=0.9, weight_decay=lane_config.WEIGHT_DECAY)
    optimizer = torch.optim.Adam(net.parameters(), lr=lane_config.BASE_LR, weight_decay=lane_config.WEIGHT_DECAY)
    
    for epoch in range(lane_config.EPOCHS):
        # adjust_lr(optimizer, epoch)
        train_epoch(net, epoch, train_data_batch, optimizer, trainF, lane_config)
        print("epoch",epoch)

        test(net, epoch, val_data_batch, testF, lane_config)
        if epoch % 5 == 0:
            torch.save(net, os.path.join(os.getcwd(), lane_config.SAVE_PATH, "laneNet{}.pth".format(epoch)))
            torch.save({'state_dict': net.state_dict()},
                       os.path.join(os.getcwd(), lane_config.SAVE_PATH, "laneNet{}.pth.tar".format(epoch)))
    trainF.close()
    testF.close()
    torch.save(net, os.path.join(os.getcwd(), lane_config.SAVE_PATH, "finalNet.pth"))
    torch.save({'state_dict': net.state_dict()}, os.path.join(os.getcwd(), lane_config.SAVE_PATH, "finalNet.pth.tar"))  #加的


if __name__ == "__main__":
    main()
