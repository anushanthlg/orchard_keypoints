"""ResNet-18 encoder + deconv head producing one heatmap per keypoint."""
import torch.nn as nn 
import torchvision 
import config as C 


class KPNet(nn.Module):  # keypoint detection neural network
    def __init__(self, pretrained=True, num_kp=len(C.KP_NAMES)):  #sets the number of keypoints
        super().__init__()  # Initializes the parent PyTorch model
        w = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None  # pretrained weights
        r = torchvision.models.resnet18(weights=w)  # ResNet-18 model
        self.encoder = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool,  # Adds the initial convolution, normalization, activation, and pooling layers
                                     r.layer1, r.layer2, r.layer3, r.layer4)  # Adds the four ResNet feature-extraction stages; output stride is 32
        layers, c = [], 512  # empty layer list and sets the initial channel count to 512
        for _ in range(3):  # Repeats the upsampling block three times to increase resolution from 8 to 64
            layers += [nn.ConvTranspose2d(c, 128, 4, 2, 1, bias=False),  # Doubles spatial resolution and produces 128 channels
                       nn.BatchNorm2d(128),  # Normalizes the 128 feature channels
                       nn.ReLU(True)]  #  in-place ReLU activation
            c = 128  # input channel count for the next upsampling block
        self.deconv = nn.Sequential(*layers)  # Combines the upsampling layers into one sequential module
        self.head = nn.Conv2d(128, num_kp, 1)  # Produces one 64 × 64 heatmap per keypoint

    def forward(self, x):  # Defines how an input image passes through the model
        return self.head(self.deconv(self.encoder(x)))  # Extracts features, upsamples them, and predicts keypoint heatmaps