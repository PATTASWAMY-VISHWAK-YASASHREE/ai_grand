import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

def create_faster_rcnn_model(num_classes=7):  # 6 classes + background
    # Load a pre-trained model for transfer learning
    backbone = torchvision.models.resnet50(pretrained=True)
    
    # Remove the last two layers (avg pool and fc)
    modules = list(backbone.children())[:-2]
    backbone = nn.Sequential(*modules)
    
    # FasterRCNN needs to know the number of output channels in the backbone
    backbone_out_channels = 2048
    
    # Create anchor generator
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    
    # Create ROI pooler
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )
    
    # Create FasterRCNN model
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=800,
        max_size=1333
    )
    
    return model

class DocumentLayoutModel(nn.Module):
    def __init__(self, num_classes=7):  # 6 classes + background
        super(DocumentLayoutModel, self).__init__()
        self.model = create_faster_rcnn_model(num_classes)
        
    def forward(self, images, targets=None):
        if self.training and targets is not None:
            return self.model(images, targets)
        else:
            return self.model(images)