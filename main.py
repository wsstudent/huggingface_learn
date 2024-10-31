from MyResnet_model.configuration_MyResnet import MyResnetConfig
from MyResnet_model.modeling_MyResnet import MyResnetModelForImageClassification
from transformers import AutoConfig, AutoModel, AutoTokenizer

# 这行代码将在 custom-resnet 文件夹内保存一个名为 config.json 的文件。
MyResnet_config = MyResnetConfig(in_channels=3, num_channels=64, num_classes=176)
MyResnet_config.save_pretrained("custom-resnet")
# 然后，你可以使用 from_pretrained 方法重新加载配置：
resnet50d_config = MyResnetConfig.from_pretrained("custom-resnet")

myResnet = MyResnetModelForImageClassification(MyResnet_config)
print(myResnet)
