import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from torchvision import models
from sklearn.metrics import f1_score
import copy


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder(root='rest/', transform=train_transform)
print(dataset.classes)
out_classes = len(dataset.classes)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_subset, val_subset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, pin_memory=True)
pred_loadel = DataLoader(val_subset, batch_size=32, shuffle=False, pin_memory=True)

device = torch.device("cuda")


class Layer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Layer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x) 
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, out_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self.make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, out_classes)

    def make_layer(self, block, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


resnet = ResNet(Layer, [2, 2, 2, 2], out_classes=out_classes)

pretrained1 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
pretrained1.fc = nn.Linear(pretrained1.fc.in_features, out_classes)

pretrained2 = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
pretrained2.classifier[-1] = nn.Linear(pretrained2.classifier[-1].in_features, out_classes)

def train_epoch(model, loader, loss_f, optimizer):
    model.train()
    loss_epoch = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(x)
        loss = loss_f(outputs, y)
        loss.backward()
        optimizer.step()
        
        loss_epoch += loss.item()
    return loss_epoch / len(loader)

def f1_eval(model, loader):
    model.eval()
    X = []
    Y = []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            
            X.extend(preds.cpu().numpy())
            Y.extend(y.cpu().numpy())
            
    return f1_score(Y, X, average='macro')

def training(model, name, epochs=15):
    
    model.to(device)
    loss_f = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    print(f"Now train {name}")
    best_f1 = 0.0
    
    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, loss_f, optimizer)
        f1 = f1_eval(model, pred_loadel)
        scheduler.step()
        
        best_f1 = max(best_f1, f1)
            
        print(f"Epoch ={epoch+1}/{epochs}  loss ={loss:.6f}  f1 ={f1:.6f}")
        
    return best_f1

training(resnet, "ResNet18_Scratch", epochs=4)

# training(pretrained1, "ResNet50_Pretrained", epochs=3)

training(pretrained2, "MobileNetV3_Pretrained", epochs=2)

input("Ready to test")


def test_run(model):
    model.to(device)

    model.eval()
    X = []
    Y = []

    with torch.no_grad():
        for x, y in pred_loadel:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            
            X.extend(preds.cpu().numpy())
            Y.extend(y.cpu().numpy())
            
    return f1_score(Y, X, average='macro')




print("resnet f1 = ", test_run(resnet))
print("MobileNetV3 f1 = ", test_run(pretrained2))