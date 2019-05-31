import torch
import csv
import os
import pandas as pd
import trafDataset
import torchvision.transforms as transforms

root = '/home/asprohy/pyWorkSpace/dsntnn'

net = torch.load("cate29.pkl")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
locafile = "tmp.csv"
datas = pd.read_csv(os.path.join(root,locafile)).values
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
testset = trafDataset.MtestDataset(os.path.join(root, locafile), transform= transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                                     shuffle=False, num_workers=2)

res = []
count = 0
with torch.no_grad():
    for data in testloader:
        images, fn, xmid, ymid = data
        images = images.to(device)
        outputs = net(images)
        xmid = xmid.numpy()
        ymid = ymid.numpy()
        _, predicted = torch.max(outputs.data, 1)
        tp = (fn[0], xmid[0], ymid[0], xmid[0]-26,ymid[0]-26,xmid[0]+26,ymid[0]+26,predicted.cpu().numpy()[0])
        res.append(tp)
        # total += labels.size(0)
        # correct += (predicted == labels).sum().item()
        count += 1
        # print("predicted    :"+str(tp))
        # if count ==5:
        #     break
        if count %100==0:
            print("countNum:    " + str(count) + "    and X = " + str(predicted.cpu().numpy()[0])+" tp: "+str(tp))

print("finish test")
with open(os.path.join(root,'write.csv'), 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    for list in res:
        print(list)
        csv_writer.writerow(list)

print("finish write")