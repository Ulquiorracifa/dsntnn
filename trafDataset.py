import torch.utils.data
import traf_data
from PIL import Image
import cv2

class MyDataset(torch.utils.data.Dataset):
    def __init__(self,train_path, transform=None, target_transform=None):
        super(MyDataset,self).__init__()
        data, _, _= traf_data.get_data2(train_path)
        imgs = []
        for c in data:
            # print(c)
            c1 = c['filepath']
            c2 = int(c['bboxes'][0]['class'])
            c3 = int((c['bboxes'][0]['x1']+c['bboxes'][0]['x2'])/2)
            c4 = int((c['bboxes'][0]['y1']+c['bboxes'][0]['y2'])/2)
            # imgs.append((c['filepath'], int(c['bboxes'][0]['class']), int((c['bboxes'][0]['x1']+c['bboxes'][0]['x2'])/2), int((c['bboxes'][0]['y1']+c['bboxes'][0]['y2'])/2)))
            imgs.append((c1,c2,c3,c4))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label, xmid, ymid= self.imgs[index]
        img = Image.open(fn)
        io = (xmid-26, ymid-26, xmid+26, ymid+26)
        img = img.crop(io)
        img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img) #是否进行transform
        return img, label-1

    def __len__(self): #这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)