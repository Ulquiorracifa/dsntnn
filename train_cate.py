import torch
import torchvision
import traf_data
import platform
import logging
import torch.optim as optim
import torch.nn as nn
import numpy as np
import CNNet
import cv2
import os
import trafDataset
import torchvision.transforms as transforms

if platform.system() =='Windows':
    train_path = "D:\Download\\traf"
else:
    train_path = "/home/asprohy/data/traffic"



# train_data, class_count, class_map = traf_data.get_data2(train_path)
model_PATH = 'traf_cate29.pt'


net = CNNet.Net().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-4)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
epoch_num = 10
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = trafDataset.MyDataset(train_path, transform= transform)
batch_size = 10

for epoch in range(epoch_num):
    for i, data in enumerate(torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        shuffle=True, num_workers=2), 0):
        # img = cv2.imread(os.path.join(train_path, c['filepath']))
        # center = [int((c['bboxes'][0]['ymin']+c['bboxes'][0]['ymax'])/2), int((c['bboxes'][0]['xmin']+c['bboxes'][0]['xmax'])/2)]#[h,w]
        # img = np.array(img)
        # img = img[center[0]-26: center[0]+26, center[1]-26: center[1]+26]
        inputs, labels = data
        labels = labels.cuda()
        inputs = inputs.cuda()

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print('[%d, %5d] loss: %.4f' %(epoch + 1, (i+1)*batch_size, loss.item()))

print('Finished Training')
torch.save(net, 'cate29.pkl')

