import matplotlib.pyplot as plt
import torch

from  net import *
from GT import *
import sys

np.random.seed(7777)
device = torch.device('cuda:0')

'''
channel = [[3,[16,16,16,16]],[64,[32,32,32,32]],[128,[32,32,32,32]],[128,[16,16,16,16]]]
feature_cnn = feature(channel)
multi_cnn = multi_cnn(feature_cnn).to(device)
multi_cnn.apply(init_weights)
'''
#multi_cnn = torch.load('./res/multi_cnn.pkl',map_location=device)

'''
#NWPU数据集
img_path = 'C:/LR/Pycharm/Project/EDENet/NWPU/images/'
gt_path= 'C:/LR/Pycharm/Project/EDENet/NWPU/mats/'
'''

'''
img_path = '/home/julyxia/C3Data/ProcessedData/NWPU/train/img/'
gt_path = './NWPU/mats/'
'''

#SH数据集
img_path= 'C:/LR/Pycharm/Project/mcnn/train_data/img/'
gt_path= 'C:/LR/Pycharm/Project/mcnn/train_data/mat/'

'''
img_path = '/home/yyyin/project/multi/ShanghaiTech/train_data/img/'
gt_path= '/home/yyyin/project/multi/ShanghaiTech/train_data/mat/'
'''

path = [img_path,gt_path]

def train(net,lr,epoch,path):
    trainer = torch.optim.Adam(net.parameters(),lr=lr)
    loss_func = torch.nn.MSELoss(reduce=True, size_average=False)
    ls_lg = []

    #训练记录保存
    for ep in range(1,epoch+1):
        for times in range(700):
            #32位浮点型
            img,den_map = Data_Loader_SH(path)
            img = img.to(device)
            den_map = den_map.to(device)
            out = net(img)
            real = den_map.sum().item()/1000
            pre = out.sum().item()/1000
            loss = loss_func(out, den_map)
            trainer.zero_grad()
            loss.backward()
            trainer.step()
            #显示
            sys.stdout.write('\r epoch: %d/%d  times: %d/700   MSE loss: %.4f  pre: %d real: %d         ' %\
                             (ep, epoch, times + 1, loss.item(), pre, real))
            sys.stdout.flush()
            if times%10==0:
                ls_lg.append(loss.item())
    #网络以及损失值数据
    torch.save(net,'./res/multi_cnn_nwpu.pkl')
    ls_lg = np.array(ls_lg)
    np.save('./res/ls_lg_nwpu.npy',ls_lg)

#开始训练
#train(multi_cnn,1e-5,5,path)


net = torch.load('./res/multi_cnn_new_SH.pkl',map_location=device)
img,den_map = Data_Loader_SH(path)

out = net(img.to(device))

plt.subplot(1,3,1)
plt.imshow(img.squeeze().permute(1,2,0))

plt.subplot(1,3,2)
plt.title(int(den_map.sum()/1000))
plt.imshow(den_map.squeeze(),CM.jet)

plt.subplot(1,3,3)
plt.title(int(out.sum()/1000))
plt.imshow(out.squeeze().detach().cpu(),CM.jet)
'''
'''