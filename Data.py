import numpy as np
import PIL.Image  as Image
import scipy.io as sio
import cv2
import torch
from tqdm import tqdm
import cv2


#加载序列
def data_iter(len):
    index = np.arange(len)+1
    np.random.shuffle(index)
    index = np.hstack((index,index))
    return list(index)

#高斯核
def gauss(kernel_size, sigma):
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2   #中心点偏移
    s = sigma ** 2  #相似度
    sum_val = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2*s))
            sum_val += kernel[i, j]
    kernel = kernel / sum_val
    return kernel

#定高斯核加载
def Loader(num,impath,g_path,R=15):
    now_impath = impath+str(num)+'.jpg'
    now_gpath = g_path+str(num)+'.mat'
    data = sio.loadmat(now_gpath)['image_info'] #字典
    n = data[0][0][0][0][1][0][0]  # 人数
    P = data[0][0][0][0][0]   # 坐标
    img = torch.from_numpy(cv2.imread(now_impath)).unsqueeze(0).transpose(2, 3).transpose(1, 2)/255
    H = int(np.ceil(img.shape[2]/16))*16
    W = int(np.ceil(img.shape[3]/16))*16
    tot = np.zeros([H + 2*(R+1), W + 2*(R+1)], dtype=np.float64)
    GAU = gauss(kernel_size=R, sigma=6)
    for j in range(n):
        x = int(P[j][1]) + (R - 1) // 2
        y = int(P[j][0]) + (R - 1) // 2
        tot[x - (R - 1) // 2:x + (R - 1) // 2 + 1, y - (R - 1) // 2:y + (R - 1) // 2 + 1] = \
            tot[x - (R - 1) // 2:x + (R - 1) // 2 + 1, y - (R - 1) // 2:y + (R - 1) // 2 + 1] + GAU
    return n,tot,img

def TOT_2(P,m,H,W,R):
    #P = torch.from_numpy(P).to(device)
    dis = np.ones([len(P),m])*float('inf')
    #dis = torch.from_numpy(dis).to(device)  #移至GPU
    for i in range(len(P)):
        for j in range(i+1,len(P)):
            tmp = np.sqrt(np.sum((P[i]-P[j])**2))
            if tmp<dis[i].max():
                dis[i,dis[i].argmax()] = tmp
            if tmp<dis[j].max():
                dis[j,dis[j].argmax()] = tmp
    tot = np.zeros([H + 2*(R + 1), W + 2*(R + 1)])
    for i in range(len(dis)):
        delta = dis[i].mean()
        GAU = gauss(R,0.65*delta)
        #GAU = torch.from_numpy(GAU).to(device)
        x = int(P[i][1]) + (R - 1) // 2
        y = int(P[i][0]) + (R - 1) // 2
        tot[x - (R - 1) // 2:x + (R - 1) // 2 + 1, y - (R - 1) // 2:y + (R - 1) // 2 + 1] = \
                tot[x - (R - 1) // 2:x + (R - 1) // 2 + 1, y - (R - 1) // 2:y + (R - 1) // 2 + 1] + GAU
    return tot

#变高斯核密度图
def Loader_2(num,impath,g_path,R=15):
    now_impath = impath+str(num)+'.jpg'
    now_gpath = g_path+str(num)+'.mat'
    data = sio.loadmat(now_gpath)['image_info'] #字典
    n = data[0][0][0][0][1][0][0]  # 人数
    P = data[0][0][0][0][0]   # 坐标
    img = torch.from_numpy(cv2.imread(now_impath)).unsqueeze(0).transpose(2, 3).transpose(1, 2) / 255
    H = int(np.ceil(img.shape[2]/16))*16
    W = int(np.ceil(img.shape[3]/16))*16
    tot = TOT_2(P,8,H,W,R)
    return n,tot,img

def TOT_3(P,m,H,W):
    #P = torch.from_numpy(P).to(device)
    dis = np.ones([len(P),m])*float('inf')
    #dis = torch.from_numpy(dis).to(device)  #移至GPU
    for i in range(len(P)):
        for j in range(i+1,len(P)):
            tmp = np.sqrt(np.sum((P[i]-P[j])**2))
            if tmp<dis[i].max():
                dis[i,dis[i].argmax()] = tmp
            if tmp<dis[j].max():
                dis[j,dis[j].argmax()] = tmp
    tot = np.zeros([H, W])
    for i in range(len(dis)):
        delta = dis[i].mean()
        R = int(dis[i].max())//2*2+1
        GAU = gauss(R,0.3*delta)
        #GAU = torch.from_numpy(GAU).to(device)
        x = int(P[i][1])
        y = int(P[i][0])
        x_st = max((x-(R-1)//2),0)
        x_ed = min((x+(R-1)//2+1),H)
        y_st = max((y-(R-1)//2),0)
        y_ed = min((y+(R-1)//2+1),W)
        GAU_T = GAU[R//2-(x-x_st):R//2+(x_ed-x),R//2-(y-y_st):R//2+(y_ed-y)]
        tot_T = tot[x_st:x_ed,y_st:y_ed]
        tot[x_st:x_ed,y_st:y_ed] = tot[x_st:x_ed,y_st:y_ed]+GAU[R//2-(x-x_st):R//2+(x_ed-x),R//2-(y-y_st):R//2+(y_ed-y)]
    return tot

#变高斯核密度图
def Loader_3(num,impath,g_path):
    now_impath = impath+str(num)+'.jpg'
    now_gpath = g_path+str(num)+'.mat'
    data = sio.loadmat(now_gpath)['image_info'] #字典
    n = data[0][0][0][0][1][0][0]  # 人数
    P = data[0][0][0][0][0]   # 坐标
    img = torch.from_numpy(cv2.imread(now_impath)).unsqueeze(0).transpose(2, 3).transpose(1, 2) / 255
    H = int(int(img.shape[2]/8))*8
    W = int(int(img.shape[3]/8))*8
    tot = TOT_3(P,6,H,W)
    return n,tot,img

#绘图函数
def DRAW(tot,person,type=0):
    # 密度图颜色
    mp = sio.loadmat('C:/LR/Pycharm/Project/mcnn/color/map.mat')
    mp = mp['c']
    mp = mp[::-1]  # 对mp取逆序

    N = tot.shape[0]
    M = tot.shape[1]
    max_den = tot.max()
    den_map = np.zeros([N,M,3],dtype=np.float64)
    for X in range(N):
        for Y in range(M):
            pixel = 255 * tot[X][Y] / max_den
            den_map[X][Y] = mp[int(pixel)] * 255  # den_map三维array，每个位置存储array
    #绘图
    if type==0:
        text ='GT Count:'+str(person)
    else:
        text = 'Est Count:' + str(round(tot.sum().item()))
    title = '       pre:'+str(round(tot.sum().item()))+'       real:'+str(person)
    cv2.namedWindow(title, 0)
    cv2.resizeWindow(title , M//2,N//2)
    cv2.putText(den_map, text, (0,N-10 ), cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 225, 255), 2)
    cv2.imshow(title, den_map/255)  # 按RGB形式展示，但den_map为BGR
    cv2.waitKey()
    cv2.destroyAllWindows()
    return None

#密度图生成函数
def Den_Gen():
    impath = 'C:/LR/Pycharm/Project/mcnn/images/ShanghaiTech/part_A/train_data/images/IMG_'
    g_path = 'C:/LR/Pycharm/Project/mcnn/images/ShanghaiTech/part_A/train_data/ground-truth/GT_IMG_'
    for i in tqdm(range(1, 301)):
        n, tot, img = Loader_3(i, impath, g_path, )
        np.save(
            'C:/LR/Pycharm/Project/mcnn/images/ShanghaiTech/part_A/train_data/density-map/DEN_IMG_' + str(i) + '.npy',
            tot)

    impath = 'C:/LR/Pycharm/Project/mcnn/images/ShanghaiTech/part_A/test_data/images/IMG_'
    g_path = 'C:/LR/Pycharm/Project/mcnn/images/ShanghaiTech/part_A/test_data/ground-truth/GT_IMG_'
    for i in tqdm(range(1, 183)):
        n, tot, img = Loader_3(i, impath, g_path)
        np.save(
            'C:/LR/Pycharm/Project/mcnn/images/ShanghaiTech/part_A/test_data/density-map/DEN_IMG_' + str(i) + '.npy',
            tot)

    impath = 'C:/LR/Pycharm/Project/mcnn/images/ShanghaiTech/part_B/train_data/images/IMG_'
    g_path = 'C:/LR/Pycharm/Project/mcnn/images/ShanghaiTech/part_B/train_data/ground-truth/GT_IMG_'
    for i in tqdm(range(1, 401)):
        n, tot, img = Loader_3(i, impath, g_path)
        np.save(
            'C:/LR/Pycharm/Project/mcnn/images/ShanghaiTech/part_B/train_data/density-map/DEN_IMG_' + str(i) + '.npy',
            tot)

    impath = 'C:/LR/Pycharm/Project/mcnn/images/ShanghaiTech/part_B/test_data/images/IMG_'
    g_path = 'C:/LR/Pycharm/Project/mcnn/images/ShanghaiTech/part_B/test_data/ground-truth/GT_IMG_'
    for i in tqdm(range(1, 317)):
        n, tot, img = Loader_3(i, impath, g_path)
        np.save(
            'C:/LR/Pycharm/Project/mcnn/images/ShanghaiTech/part_B/test_data/density-map/DEN_IMG_' + str(i) + '.npy',
            tot)

def Test_Load(im_path,g_path,den_path,n):
    now_impath = im_path+str(n)+'.jpg'
    now_gpath = g_path+str(n)+'.mat'
    now_denpath = den_path+str(n)+'.npy'
    data = sio.loadmat(now_gpath)['image_info']  # 字典
    n = data[0][0][0][0][1][0][0]  # 人数
    img = (torch.from_numpy(cv2.imread(now_impath)).unsqueeze(0).transpose(2, 3).transpose(1, 2) / 255).float()
    den_map = (torch.from_numpy(np.load(now_denpath))).float()
    return n,img,den_map

