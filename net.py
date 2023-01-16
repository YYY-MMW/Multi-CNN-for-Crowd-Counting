import torch
import torch.nn as nn
import torchvision

#注意力模块
class CBAM_block(nn.Module):
    def __init__(self,channel,reduction=4,spatial_ker=7):
        super().__init__()
        #channel attention
        self.max = nn.AdaptiveMaxPool2d(1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels=channel,
                      out_channels=channel//reduction,
                      kernel_size=1,
                      bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=channel//reduction,
                      out_channels=channel,
                      kernel_size=1,
                      bias=True)
        )
        #spatial attention
        self.cov = nn.Conv2d(2,1,kernel_size=spatial_ker,padding=spatial_ker//2)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        max = self.mlp(self.max(x))
        avg = self.mlp(self.avg(x))
        m_c = self.sigmoid(max+avg)
        x = x*m_c
        max_c = torch.max(x,dim=1,keepdim=True)
        avg_c = torch.mean(x,dim=1,keepdim=True)
        spatial_out = self.sigmoid(self.cov(torch.cat([max_c.values,avg_c],dim=1)))
        x = x*spatial_out
        return x

#vgg16 前10层
def vgg_features_10():
    net = []
    vgg = torchvision.models.vgg16(pretrained=True)
    for i in range(23):
        if i not in [4,9,16]:
            net.append(vgg.features[i])
    return nn.Sequential(*net)

#multi块
class multi_block(nn.Module):
    def __init__(self,inchanels,outchannels):
        super().__init__()
        self.cov1 = nn.Sequential(
            nn.Conv2d(
                in_channels=inchanels,
                out_channels=outchannels[0],
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(outchannels[0]),
            nn.ReLU()
        )
        self.cov2 = nn.Sequential(
            nn.Conv2d(
                in_channels=inchanels,
                out_channels=inchanels//2,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(inchanels//2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=inchanels//2,
                out_channels=outchannels[1],
                kernel_size = 3,
                stride = 1,
                padding = 1,
            ),
            nn.BatchNorm2d(outchannels[1]),
            nn.ReLU(),
        )
        self.cov3 = nn.Sequential(
            nn.Conv2d(
                in_channels=inchanels,
                out_channels=inchanels//2,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(inchanels//2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=inchanels//2,
                out_channels=outchannels[2],
                kernel_size = 5,
                stride = 1,
                padding = 2,
            ),
            nn.BatchNorm2d(outchannels[2]),
            nn.ReLU(),
        )
        self.cov4 = nn.Sequential(
            nn.Conv2d(
                in_channels=inchanels,
                out_channels=inchanels//2,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(inchanels // 2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=inchanels//2,
                out_channels=outchannels[3],
                kernel_size = 7,
                stride = 1,
                padding = 3,
            ),
            nn.BatchNorm2d(outchannels[3]),
            nn.ReLU(),
        )

    def forward(self,X):
        cov1 = self.cov1(X)
        cov2 = self.cov2(X)
        cov3 = self.cov3(X)
        cov4 = self.cov4(X)
        out = torch.cat([cov1,cov2,cov3,cov4],dim=1)
        return out

#第一块
class first_block(nn.Module):
    def __init__(self,inchanels,outchannels):
        super().__init__()
        self.cov1 = nn.Sequential(
            nn.Conv2d(
                in_channels=inchanels,
                out_channels=outchannels[0],
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(outchannels[0]),
            nn.ReLU()
        )
        self.cov2 = nn.Sequential(
            nn.Conv2d(
                in_channels=inchanels,
                out_channels=outchannels[1],
                kernel_size = 3,
                stride = 1,
                padding = 1,
            ),
            nn.BatchNorm2d(outchannels[1]),
            nn.ReLU(),
        )
        self.cov3 = nn.Sequential(
            nn.Conv2d(
                in_channels=inchanels,
                out_channels=outchannels[2],
                kernel_size = 5,
                stride = 1,
                padding = 2,
            ),
            nn.BatchNorm2d(outchannels[2]),
            nn.ReLU(),
        )
        self.cov4 = nn.Sequential(
            nn.Conv2d(
                in_channels=inchanels,
                out_channels=outchannels[3],
                kernel_size = 7,
                stride = 1,
                padding = 3,
            ),
            nn.BatchNorm2d(outchannels[3]),
            nn.ReLU(),
        )

    def forward(self,X):
        cov1 = self.cov1(X)
        cov2 = self.cov2(X)
        cov3 = self.cov3(X)
        cov4 = self.cov4(X)
        out = torch.cat([cov1,cov2,cov3,cov4],dim=1)
        return out

#特征提取器
class feature(nn.Module):
    def __init__(self, channel):
        super().__init__()
        #归一化
        self.bn = nn.BatchNorm2d(3)
        #构建特征提取器
        net = []
        for i in range(4):
            if i ==0:
                net.append(first_block(channel[i][0], channel[i][1]))
            else:
                net.append(multi_block(channel[i][0],channel[i][1]))
            if i != 3:
                net.append(nn.MaxPool2d(kernel_size=2, stride=2))
                net.append(CBAM_block(sum(channel[i][1])))
        self.features = nn.Sequential(*net)

    def forward(self,x):
        BN = self.bn(x)
        mask = self.features(x)
        return mask

#residual block
class res_block(nn.Module):
    def __init__(self,inchannel,outchannel,use_res):
        super().__init__()
        self.cov1 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel,out_channels=outchannel,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
        )
        self.cov2 = nn.Sequential(
            nn.Conv2d(in_channels=outchannel,out_channels=outchannel,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(outchannel),
        )
        self.cov3 = None
        if use_res:
            self.cov3 = nn.Sequential(
                nn.Conv2d(in_channels=inchannel,out_channels=outchannel,kernel_size=1,stride=1,padding=0),
            )
        self.relu = nn.ReLU()
    def forward(self,X):
        Y = self.cov1(X)
        Y = self.cov2(Y)
        if self.cov3:
            X = self.cov3(X)
        Y = self.relu(Y+X)
        return Y


#计数网络
class multi_cnn(nn.Module):
    def __init__(self,feature_cnn):
        super().__init__()
        self.feature = feature_cnn
        #构建解码器(BN)
        self.generate = nn.Sequential(
            #解码1
            res_block(inchannel=64,outchannel=64,use_res=False),
            res_block(inchannel=64, outchannel=64, use_res=False),
            res_block(inchannel=64, outchannel=64, use_res=False),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),

            #解码2
            res_block(inchannel=64, outchannel=32, use_res=True),
            res_block(inchannel=32, outchannel=32, use_res=False),
            res_block(inchannel=32, outchannel=32, use_res=False),
            res_block(inchannel=32, outchannel=32, use_res=False),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),

            #解码3
            res_block(inchannel=32, outchannel=32, use_res=False),
            res_block(inchannel=32, outchannel=32, use_res=False),
            res_block(inchannel=32, outchannel=32, use_res=False),
            res_block(inchannel=32, outchannel=32, use_res=False),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),

            #密度图生成
            nn.Conv2d(in_channels=32,out_channels=16,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
    def forward(self,x):
        mask = self.feature(x)
        den = self.generate(mask)
        return den

#Xavier初始化
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)



