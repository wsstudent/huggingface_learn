import os
import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from .configuration_MyResnet import MyResnetConfig

# 设置CUDA异常阻塞，用于调试CUDA相关问题
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# 定义残差块
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        # 第一个3x3卷积层
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        # 第二个3x3卷积层
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        # 可选的1x1卷积层，用于调整输入的通道数
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        # 批量归一化层
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        # 第一个卷积 -> 批量归一化 -> ReLU激活
        Y = F.relu(self.bn1(self.conv1(X)))
        # 第二个卷积 -> 批量归一化
        Y = self.bn2(self.conv2(Y))
        # 如果使用1x1卷积，调整输入的通道数
        if self.conv3:
            X = self.conv3(X)
        # 将输入与输出相加
        Y += X
        return F.relu(Y)  # 返回激活后的结果


# 组合多个残差块
def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    """
    :param first_block: 是否为第一个块，用于确定是否需要1x1卷积
    :param input_channels: 输入通道数
    :param num_channels: 残差块的输出通道数
    :param num_residuals: 残差块的数量
    :return: 组合后的多个残差块
    """
    blk = []
    for i in range(num_residuals):
        # 第一个残差块需要降维
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


# 定义残差网络
def net(in_channels, num_channels, num_residuals, num_classes):
    """
    :param in_channels: 输入图像的通道数
    :param num_channels: 第一个卷积层的输出通道数
    :param num_residuals: 每个阶段的残差块数量
    :param num_classes: 分类的数量
    :return: 构建的残差网络模型
    """
    # 首先是一个7x7卷积层，接着是批量归一化、ReLU激活和3x3最大池化
    b1 = nn.Sequential(nn.Conv2d(in_channels, num_channels, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    # 构建多个残差块
    b2 = nn.Sequential(*resnet_block(64, num_channels, num_residuals[0], first_block=True))
    b3 = nn.Sequential(*resnet_block(num_channels, num_channels * 2, num_residuals[1]))
    b4 = nn.Sequential(*resnet_block(num_channels * 2, num_channels * 4, num_residuals[2]))
    b5 = nn.Sequential(*resnet_block(num_channels * 4, num_channels * 8, num_residuals[3]))

    # 全局平均池化后，连接一个全连接层进行分类
    resnet = nn.Sequential(b1, b2, b3, b4, b5,
                           nn.AdaptiveAvgPool2d((1, 1)),
                           nn.Flatten(), nn.Linear(num_channels * 8, num_classes))
    return resnet


# 定义自定义的残差网络模型，用于图像分类
class MyResnetModelForImageClassification(PreTrainedModel):
    config_class = MyResnetConfig  # 指定配置类

    def __init__(self, config):
        super().__init__(config)
        # 根据配置初始化模型
        self.model = net(
            in_channels=config.in_channels,
            num_channels=config.num_channels,
            num_residuals=config.num_residuals,
            num_classes=config.num_classes
        )

    """
    你可以让模型返回任何你想要的内容，
    但是像这样返回一个字典，并在传递标签时包含loss，可以使你的模型能够在 Trainer 类中直接使用。
    只要你计划使用自己的训练循环或其他库进行训练，也可以使用其他输出格式。
    """

    def forward(self, tensor, labels=None):
        # 前向传播，计算模型输出
        logits = self.model(tensor)
        if labels is not None:
            # 计算损失
            loss = torch.nn.functional.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}  # 返回损失和输出
        return {"logits": logits}  # 仅返回输出
