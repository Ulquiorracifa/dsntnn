import torch
import os
import cv2
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np



image_size = [800, 450]

test_path = '/home/asprohy/data/traffic/test_data'
# test_path = '/home/asprohy/data/traffic/train_traf'
filename = '0a6d32079b4543579e27672efdee9e99.jpg'
# filename = '0a2a7e81204b42d3baddd83e187adf4b.jpg'


img = cv2.imread(os.path.join(test_path, filename))
h, w = img.shape[:2]
img = cv2.resize(img, (image_size[0],image_size[1]))

raccoon_face_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
input_tensor = raccoon_face_tensor.div(255).unsqueeze(0)
input_var = input_tensor.cuda()
print(input_var)

PATH ='traf_dsntnn29.pt'

model = torch.load(PATH).cuda()
model.eval()

coords, heatmaps = model(input_var)
print(coords)


print('Initial prediction: {:0.4f}, {:0.4f}'.format(*list(coords.data[0, 0])))
# plt.imshow(heatmaps[0, 0].detach().cpu().numpy())
plt.imshow(np.array(img))
x = (((coords.data[0, 0]).cpu()+1) *torch.Tensor(image_size)-1)/2
x = x.numpy()
print(x)
plt.scatter([x[0]],[x[1]], color='red', marker='X')
plt.show()