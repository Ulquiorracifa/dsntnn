from CRNet import CoordRegressionNetwork
from torch import optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import scipy.misc
import torch
import dsntnn
import traf_data
import cv2
import os
import numpy as np
import logging
import platform

image_size = [800, 400]
if platform.system() =='Windows':
    train_path = "D:\Download\\traf"
else:
    train_path = "/home/asprohy/data/traffic"
# raccoon_face = scipy.misc.imresize(scipy.misc.face()[200:400, 600:800, :], image_size)

# eye_x, eye_y = 24, 26 # label
#
# plt.imshow(raccoon_face)
# plt.scatter([eye_x], [eye_y], color='red', marker='X')
# plt.show()

logging.basicConfig(filename='fastrcnnTraf27.log',level=logging.DEBUG)
train_data,_,_ = traf_data.get_data2(train_path)
datatype = 'traf'
model_PATH = 'traf_dsntnn27.pt'

# data = train_data
# img_all = []
# label_all = []
# img_label = {}
# for c in data:
#     img = cv2.imread(os.path.join(train_path, c[0]))
#     h,w = img.shape[:2]
#     img = cv2.resize(img, image_size)
#     label_all.append([int(c[1]/w*image_size[0]), int(c[2]/h*image_size[1])])
#     img = np.array(img)
#     img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
#     img_label[c[0]]={}
#     img_label[c[0]]['filepath'] = os.path.join(train_path, c[0])

img_all = []
label_all = []
# count = 0
# if datatype == 'traf':
#     for c in train_data:
#         img = cv2.imread(os.path.join(train_path, c['filepath']))
#         h,w = img.shape[:2]
#         img = cv2.resize(img, image_size)
#         img = np.array(img)
#         img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
#         tmpc = c['bboxes'][0]
#
#         label_all.append([int((tmpc['x1']+tmpc['x2']) / w * image_size[0]), int((tmpc['y1']+tmpc['y2']) / h * image_size[1])])
#
#         img_all.append(img_tensor)
#         count += 1
# else:
#     print('dataLoading is error')


#
# raccoon_face_tensor = img_all.permute(2, 0, 1).float()
# input_tensor = raccoon_face_tensor.div(255).unsqueeze(0)
# input_var = Variable(input_tensor, requires_grad=False)
#
# eye_coords_tensor = torch.Tensor([label_all])
# target_tensor = (eye_coords_tensor * 2 + 1) / torch.Tensor(image_size) - 1
# target_var = Variable(target_tensor, requires_grad=False)
#
# print('Target: {:0.4f}, {:0.4f}'.format(*list(target_tensor.squeeze())))
#
#
# model = CoordRegressionNetwork(n_locations=1)
#
# coords, heatmaps = model(input_var)
#
# print('Initial prediction: {:0.4f}, {:0.4f}'.format(*list(coords.data[0, 0])))
# plt.imshow(heatmaps[0, 0].data.cpu().numpy())
# plt.show()
#
# optimizer = optim.RMSprop(model.parameters(), lr=2.5e-4)
# epoch_num = 200
# for c in epoch_num:
#     # for i in train_data:
#     # Forward pass
#     coords, heatmaps = model(input_var)
#
#     # Per-location euclidean losses
#     euc_losses = dsntnn.euclidean_losses(coords, target_var)
#     # Per-location regularization losses
#     reg_losses = dsntnn.js_reg_losses(heatmaps, target_var, sigma_t=1.0)
#     # Combine losses into an overall loss
#     loss = dsntnn.average_loss(euc_losses + reg_losses)
#
#     # Calculate gradients
#     optimizer.zero_grad()
#     loss.backward()
#
#     # Update model parameters with RMSprop
#     optimizer.step()
#
# #single train

img = cv2.imread(os.path.join(train_path, train_data[0]['filepath']))
print(os.path.join(train_path, train_data[0]['filepath']))
h, w = img.shape[:2]
print('h[],w[]', h, w)
print('filepath',train_data[0]['filepath'])
# print(img)
img = cv2.resize(img, (image_size[0],image_size[1]))
img = np.array(img)
tmpc = train_data[0]['bboxes'][0]
print('lab', tmpc)
label_all = [int((tmpc['x1'] + tmpc['x2'])/2 / w * image_size[0]),int((tmpc['y1'] + tmpc['y2'])/2 / h * image_size[1])]
print('lab', label_all)
raccoon_face_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
input_tensor = raccoon_face_tensor.div(255).unsqueeze(0)
input_var = input_tensor.cuda()

eye_coords_tensor = torch.Tensor([[label_all]])
target_tensor = (eye_coords_tensor * 2 + 1) / torch.Tensor(image_size) - 1
target_var = target_tensor.cuda()

model = CoordRegressionNetwork(n_locations=1).cuda()

coords, heatmaps = model(input_var)

print('Initial prediction: {:0.4f}, {:0.4f}'.format(*list(coords.data[0, 0])))
plt.imshow(heatmaps[0, 0].data.cpu().numpy())
plt.scatter([label_all[0]], [label_all[1]], color='red', marker='X')
plt.show()

optimizer = optim.RMSprop(model.parameters(), lr=2.5e-4)
epoch_num = 10

for i in range(epoch_num):
    count =1
    for c in train_data[:2000]:
        # Forward pass
        img = cv2.imread(os.path.join(train_path, c['filepath']))
        h, w = img.shape[:2]
        img = cv2.resize(img, (image_size[0],image_size[1]))
        img = np.array(img)
        tmpc = c['bboxes'][0]
        label_all = [int((tmpc['x1'] + tmpc['x2'])/2 / w * image_size[0]),int((tmpc['y1'] + tmpc['y2'])/2 / h * image_size[1])]

        raccoon_face_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        input_tensor = raccoon_face_tensor.div(255).unsqueeze(0)
        input_var = input_tensor.cuda()

        eye_coords_tensor = torch.Tensor([[label_all]])
        target_tensor = (eye_coords_tensor * 2 + 1) / torch.Tensor(image_size) - 1
        target_var = target_tensor.cuda()

        coords, heatmaps = model(input_var)

        # Per-location euclidean losses
        euc_losses = dsntnn.euclidean_losses(coords, target_var)
        # Per-location regularization losses
        reg_losses = dsntnn.js_reg_losses(heatmaps, target_var, sigma_t=1.0).cuda()
        # Combine losses into an overall loss
        loss = dsntnn.average_loss(euc_losses + reg_losses).cuda()

        # Calculate gradients
        optimizer.zero_grad()
        loss.backward()
        count +=1

        if count%200==0:
            print("process: "+str(count)+"  /2000   in epoch:   " +str(i)+str(target_var))
            print("loss: "+str(loss) +" coords: "+str(list(coords.data[0, 0])))
            logging.info("process: "+str(count)+"  /2000   in epoch:   " +str(i)+str(target_var))

        # Update model parameters with RMSprop
        optimizer.step()

    if (i+1)%2 ==0:
        x =model.eval()
        print(x)
        logging.info(x)
        torch.save(model, model_PATH)
        logging.info("save model in ",i)

# Predictions after training
print('Predicted coords: {:0.4f}, {:0.4f}'.format(*list(coords.data[0, 0])))
logging.info('Predicted coords: {:0.4f}, {:0.4f}'.format(*list(coords.data[0, 0])))
plt.imshow(heatmaps[0, 0].data.cpu().numpy())
plt.show()