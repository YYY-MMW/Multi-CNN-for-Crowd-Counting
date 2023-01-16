import torch

from  net import *
from Data import *
import sys

np.random.seed(7777)
device = torch.device('cuda:0')

channel = [[3,[16,16,16,16]],[64,[32,32,32,32]],[128,[32,32,32,32]],[128,[16,16,16,16]]]
feature_cnn = feature(channel)
multi_cnn = multi_cnn(feature_cnn).to(device)
multi_cnn.apply(init_weights)

train_img_A = 'C:/LR/Pycharm/Project/mcnn/images/ShanghaiTech/part_A/train_data/images/IMG_'
train_den_A = 'C:/LR/Pycharm/Project/mcnn/images/ShanghaiTech/part_A/train_data/density-map/DEN_IMG_'
train_img_B = 'C:/LR/Pycharm/Project/mcnn/images/ShanghaiTech/part_B/train_data/images/IMG_'
train_den_B = 'C:/LR/Pycharm/Project/mcnn/images/ShanghaiTech/part_B/train_data/density-map/DEN_IMG_'
path = [train_img_A,train_img_B,train_den_A,train_den_B]

'''
train_img_A = '/home/yyyin/project/multi/ShanghaiTech/part_A/train_data/images/IMG_'
train_den_A = '/home/yyyin/project/multi/ShanghaiTech/part_A/train_data/density-map/DEN_IMG_'
train_img_B = '/home/yyyin/project/multi/ShanghaiTech/part_B/train_data/images/IMG_'
train_den_B = '/home/yyyin/project/multi/ShanghaiTech/part_B/train_data/density-map/DEN_IMG_'
path = [train_img_A,train_img_B,train_den_A,train_den_B]
'''

def train(net,lr,epoch,path):
    trainer = torch.optim.Adam(net.parameters(),lr=lr)
    loss_func = torch.nn.MSELoss(reduce=True, size_average=False)
    ls_lg = []

    #训练记录保存
    for ep in range(1,epoch+1):
        list = [data_iter(300),data_iter(400)]
        count = [0,0]
        for times in range(300):
            judge = np.random.randint(1,8)//4
            now_impath = path[judge] + str(list[judge][count[judge]]) + '.jpg'
            now_denpath = path[judge+2]+str(list[judge][count[judge]])+'.npy'
            #32位浮点型
            img = ((torch.from_numpy(cv2.imread(now_impath))).unsqueeze(0).transpose(2, 3).transpose(1, 2)/255).to(device).float()
            den_map = (torch.from_numpy(np.load(now_denpath))).to(device).unsqueeze(0).unsqueeze(0).float()*1000
            count[judge]+=1
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
    torch.save(net,'./res/multi_cnn.pkl')
    ls_lg = np.array(ls_lg)
    np.save('./res/ls_lg.npy',ls_lg)
'''
#开始训练
train(multi_cnn,1e-5,50,path)

'''
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
im_path = 'C:/LR/Pycharm/Project/mcnn/images/ShanghaiTech/part_A/train_data/images/IMG_'
g_path = 'C:/LR/Pycharm/Project/mcnn/images/ShanghaiTech/part_A/train_data/ground-truth/GT_IMG_'
den_path = 'C:/LR/Pycharm/Project/mcnn/images/ShanghaiTech/part_A/train_data/density-map/DEN_IMG_'
n,img,den = Test_Load(im_path,g_path,den_path,52)
multi_cnn = torch.load('./res/multi_cnn.pkl',map_location=device)
out = multi_cnn(img.to(device)).detach().squeeze().cpu().numpy()/1000
ls = np.load('./res/ls_lg.npy')
DRAW(out,n,1)

