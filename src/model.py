import torch.nn as nn
import torchvision.models as models

'''
Steps :
    1. Load pretrained ImageNet weights
    2. Freeze all pretrained layers
    3. Unfreeze deepest convolutional block (layer-4)
    4. Replace final fully connected layer with custom classifier

    '''
def get_model(num_classes=3):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze layer4
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    # Replace FC
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # Original fc = Linear(512 -> 1000) :: Ours fc = Linear(512 -> 3)
    
    return model

    '''
    RESnet structure
    conv1 -> layer1-> layer2 -> layer3 -> layer4 -> fc

    '''
