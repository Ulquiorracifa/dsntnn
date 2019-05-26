import torch
import os
import cv2
from torch.autograd import Variable
import matplotlib.pyplot as plt



image_size = [800, 400]

test_path = '/home/asprohy/data/traffic/test_data'
filename = '0a1e3e7e0c754ecf8645186fdab0cb50.jpg'


img = cv2.imread(os.path.join(test_path, filename))
h, w = img.shape[:2]

raccoon_face_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
input_tensor = raccoon_face_tensor.div(255).unsqueeze(0)
input_var = Variable(input_tensor, requires_grad=False)
print(input_var)

PATH ='traf_dsntnn.pt'

model = torch.load(PATH)

coords, heatmaps = model(input_var)
print(coords)


print('Initial prediction: {:0.4f}, {:0.4f}'.format(*list(coords.data[0, 0])))
plt.imshow(heatmaps[0, 0].detach().cpu().numpy())
x = (((coords.data[0, 0]).numpy()+1) *image_size-1)/2
print(x)
plt.scatter([x[0]],[x[1]], color='red', marker='X')
plt.show()