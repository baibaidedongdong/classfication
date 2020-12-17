import torch
from torch import nn
import torch.nn.functional as F

#首先是基础残差快的代码,resnet就是由大残差快组成，每个大残差快又包含数个这种基础残差快
class Bottleneck(nn.Module):
    def __init__(self, in_c, out_c, stride = 1, downsampling = False, expansion = 4):#expansion对应一个基础残差块输出通道数扩大的倍数,downsampling是参数用来给跳级结构改变输出通道的
        super(Bottleneck, self).__init__()                                          #这里的stride主要是用来控制残差快中第一个残差快的输出步幅，用来降低特征图尺寸(每个大残差都是通过第一个残差快降低feature_map)

        self.downsampling = downsampling
        self.expansion = expansion

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),#解释下inplace参数的意思，为True表示执行relu,false，不执行relu
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c*self.expansion, kernel_size=1, stride =1, bias=False),
            nn.BatchNorm2d(out_c*self.expansion)
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c*self.expansion, kernel_size=1, stride = stride, bias=False),
                nn.BatchNorm2d(out_c*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bottleneck(x)
        out_1 = x

        if self.downsampling:
            out_1 = self.downsample(x)

        out += out_1
        out = self.relu(out)

        return out

#定义resnet刚开始的卷积块，kernel_size = 7,和残差快不一样，需要单独写出来
def Conv1(in_c = 3, out_c = 64, stride = 2):
    return(nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size= 7, stride= stride,  padding=3, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride= 2, padding= 1)
    ))

class ResNet(nn.Module):
    def __init__(self, blocks, number_classes, expensions = 4):
        super(ResNet, self).__init__()

        self.expensions = expensions

        self.conv1 = Conv1(3, 64, stride= 2)

        self.layer1 = self.make_layer(64, 64, blocks[0], stride=1)#注意这里是self.make_layer而不是make_layer,self在这里相当于调用了ResNet这个类，这就是self的作用;另外这里的stride为１，因为在上一层已经下采样了一个词
        self.layer2 = self.make_layer(256, 128, blocks[1], stride=2)
        self.layer3 = self.make_layer(512, 256, blocks[2], stride=2)
        self.layer4 = self.make_layer(1024, 512, blocks[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)#这里设为７是因为我要测试的tensor的长宽为224，经过５次下采样特征图就变为了7,z这里可以自己改使它更加的有灵活性
        self.liner = nn.Linear(2048, number_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')#看了下源码，pytorch1.6凯明初始化的激活函数已经默认为leaky_relu,这里为了和论文中一样，还是改为relu
            if isinstance(m, nn.BatchNorm2d):           #BatchNorm2d也有２个超参数，所以要进行初始化
                nn.init.constant_(m.weight, 1)#将BatchNorm2d参数初始化为１
                nn.init.constant_(m.bias, 0)#偏置初始化为０，为什么这样初始化俺也不知道，由于在定义卷积层时bias=False，所以卷基层没有初始化bias

    def make_layer(self, in_c, out_c, block, stride):
        layers = []

        layers.append(Bottleneck(in_c, out_c, stride= stride, downsampling= True))#这里要单独写，这里用来进行下采样，每个大的残差结构的第一个小的残差块和后面的几个残差块不一样

        for i in range(1, block):
            layers.append(Bottleneck(out_c * self.expensions, out_c))
        return nn.Sequential(*layers)#*加数组可以将数组变为一个一个的元素，所以这里用*layers来将数组变为一个一个的单元，用nn.Sequential链接起来

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.liner(x)

        return x

def ResNet50():
    return ResNet([3, 4, 6, 3], number_classes=1000)

model = ResNet50()
print(model)
X = torch.rand(1, 3, 224, 224)
for blk in model.children():
    X = blk(X)
    print('output shape: ', X.shape)

