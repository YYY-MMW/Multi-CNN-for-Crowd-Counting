import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm as CM     #人群密度色表
import cv2
import scipy.ndimage    #高斯模糊
import scipy.io as sio  #图像读取
from skimage import exposure    #亮度调整


def gaussian_filter_density(gt):
    # 初始化密度图
    density = np.zeros(gt.shape, dtype=np.float32)
    # 获取gt中不为0的元素的个数
    gt_count = np.count_nonzero(gt)
    # 如果gt全为0，就返回全0的密度图
    if gt_count == 0:
        return density
    sigma = 7
    density += scipy.ndimage.gaussian_filter(gt, sigma, mode='constant')    #高斯模糊
    return density


def density_map(img,gt):
    k = np.zeros((img.shape[0],img.shape[1]))
    for i in range(len(gt)):#生成头部点注释图
        if gt[i][0] < img.shape[1] and gt[i][1] < img.shape[0]:
            k[int(gt[i][1])][int(gt[i][0])] += 1
    k = gaussian_filter_density(k)
    return k

#NWPU数据加载
def Data_Loader_NWPU(path):
    num = np.random.randint(1,3110)
    img_nowpath = path[0] + str(num).zfill(4) + '.jpg'
    gt_nowpath = path[1] + str(num).zfill(4) + '.mat'
    #灰度图
    gray = np.random.random()
    if gray<=0.1:
        img = cv2.imread(img_nowpath,cv2.IMREAD_GRAYSCALE)  #灰度图
        img = np.stack((img,img,img),-1)/225
    else:
        img = cv2.imread(img_nowpath)[:, :, [2, 1, 0]]/225
    #翻转
    fil = np.random.randint(0, 2)  # 0.5概率翻转
    if fil <=0.5:
        img_fil = np.fliplr(img)  # 水平翻转
    #增亮
    gamma = np.random.random()
    if gamma<=0.3:
        img = exposure.adjust_gamma(img, np.random.randint(0,2)+0.5)
    gt = sio.loadmat(gt_nowpath)
    den_map = density_map(img, gt['annPoints'])
    #裁剪
    H = img.shape[0]
    W = img.shape[1]
    if H>768:
        H =768
        h_start = np.random.randint(0, img.shape[0] - H)
    else:
        H = H//8*8
        h_start = 0
    if W>768:
        W =768
        w_start = np.random.randint(0, img.shape[1] - W)
    else:
        W = W//8*8
        w_start = 0
    img = img[h_start:h_start+H,w_start:w_start+W]
    den_map = den_map[h_start:h_start+H,w_start:w_start+W]
    den_map = torch.from_numpy(den_map).float().unsqueeze(0).unsqueeze(0) * 1000
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    return img,den_map

#ShanghaiTech数据加载
def Data_Loader_SH(path):
    num = np.random.randint(1,700)
    img_nowpath = path[0] + str(num).zfill(4) + '.jpg'
    gt_nowpath = path[1] + str(num).zfill(4) + '.mat'
    #灰度图
    gray = np.random.random()
    if gray<=0.1:
        img = cv2.imread(img_nowpath,cv2.IMREAD_GRAYSCALE)  #灰度图
        img = np.stack((img,img,img),-1)/225
    else:
        img = cv2.imread(img_nowpath)[:, :, [2, 1, 0]]/225
    #翻转
    fil = np.random.randint(0, 2)  # 0.5概率翻转
    if fil <=0.5:
        img_fil = np.fliplr(img)  # 水平翻转
    #增亮
    gamma = np.random.random()
    if gamma<=0.3:
        img = exposure.adjust_gamma(img, np.random.randint(0,2)+0.5)
    gt = sio.loadmat(gt_nowpath)
    den_map = density_map(img, gt['image_info'][0][0][0][0][0])

    #裁剪
    H = img.shape[0]
    W = img.shape[1]
    if H>256:
        H =256
        h_start = np.random.randint(0, img.shape[0] - H)
    else:
        H = H//8*8
        h_start = 0
    if W>256:
        W =256
        w_start = np.random.randint(0, img.shape[1] - W)
    else:
        W = W//8*8
        w_start = 0
    img = img[h_start:h_start+H,w_start:w_start+W]
    den_map = den_map[h_start:h_start+H,w_start:w_start+W]

    den_map = torch.from_numpy(den_map).float().unsqueeze(0).unsqueeze(0) * 1000
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    return img,den_map


'''
img_path= 'C:/LR/Pycharm/Project/mcnn/train_data/img/'
gt_path= 'C:/LR/Pycharm/Project/mcnn/train_data/mat/'
path = [img_path,gt_path]
img,den = Data_Loader_SH(path)
#plt.imshow(img.squeeze().permute(1,2,0).numpy())
plt.imshow(den.squeeze().numpy(),CM.jet)
plt.imshow(den_map.squeeze().detach().cpu(),CM.jet)
'''
