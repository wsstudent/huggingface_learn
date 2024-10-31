from transformers import PretrainedConfig

"""
编写自定义配置时需要记住的三个重要事项如下：
必须继承自 PretrainedConfig，
PretrainedConfig 的 __init__ 方法必须接受任何 kwargs，
这些 kwargs 需要传递给超类的 __init__ 方法。
"""
class MyResnetConfig(PretrainedConfig):
    model_type = "resnet"

    def __init__(
            self,
            num_classes: int = 176,  # 分类数
            in_channels: int = 3,  # 输入通道数
            num_channels: int = 64,  # 第一个卷积的输出通道数
            num_residuals=None,  # 每个残差块组合里残差块的数量
            **kwargs,
    ):
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_channels = num_channels
        if num_residuals is None:
            num_residuals = [2, 2, 2, 2]
        self.num_residuals = num_residuals
        super().__init__(**kwargs)
