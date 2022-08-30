from efficientnet_pytorch import EfficientNet
import torch
model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=23)
img = torch.randn([3,3,224,224])
out = model(img)
print(out.shape)
