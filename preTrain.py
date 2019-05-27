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


image_size = [40, 40]
raccoon_face = scipy.misc.imresize(scipy.misc.face()[200:400, 600:800, :], image_size)
eye_x, eye_y = 24, 26

plt.imshow(raccoon_face)
plt.scatter([eye_x], [eye_y], color='red', marker='X')
plt.show()


raccoon_face_tensor = torch.from_numpy(raccoon_face).permute(2, 0, 1).float()
input_tensor = raccoon_face_tensor.div(255).unsqueeze(0)
input_var = input_tensor.cuda()

eye_coords_tensor = torch.Tensor([[[eye_x, eye_y]]])
target_tensor = (eye_coords_tensor * 2 + 1) / torch.Tensor(image_size) - 1
target_var = target_tensor.cuda()

print('Target: {:0.4f}, {:0.4f}'.format(*list(target_tensor.squeeze())))

model = CoordRegressionNetwork(n_locations=1).cuda()

coords, heatmaps = model(input_var)

print('Initial prediction: {:0.4f}, {:0.4f}'.format(*list(coords[0, 0])))
plt.imshow(heatmaps[0, 0].detach().cpu().numpy())
plt.show()

optimizer = optim.RMSprop(model.parameters(), lr=2.5e-4)

for i in range(400):
    # Forward pass
    coords, heatmaps = model(input_var)

    # Per-location euclidean losses
    euc_losses = dsntnn.euclidean_losses(coords, target_var)
    # Per-location regularization losses
    reg_losses = dsntnn.js_reg_losses(heatmaps, target_var, sigma_t=1.0)
    # Combine losses into an overall loss
    loss = dsntnn.average_loss(euc_losses + reg_losses)

    # Calculate gradients
    optimizer.zero_grad()
    loss.backward()

    # Update model parameters with RMSprop
    optimizer.step()

# Predictions after training
print('Predicted coords: {:0.4f}, {:0.4f}'.format(*list(coords[0, 0])))
plt.imshow(heatmaps[0, 0].detach().cpu().numpy())
plt.show()
