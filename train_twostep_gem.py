import numpy as np
import pandas as pd
import torchvision
import torch.nn as nn
from tqdm import tqdm
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import torch
from torchvision import transforms
import os
import numpy as np
import cv2
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import f1_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import json 
from resnet import ResNet, Bottleneck
import torch.nn.functional as F

batch_size=16
sample = pd.read_csv('label.csv')

print(sample.head())


def _resnext(arch, block, layers, pretrained, progress, **kwargs):
        model = ResNet(block, layers, **kwargs)
        return model
    
def resnext101_32x16d_wsl(progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x16 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 16
    return _resnext('resnext101_32x16d', Bottleneck, [3, 4, 23, 3], True, progress, **kwargs)


from torch.nn.parameter import Parameter
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)       
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'        


class OdirDatasetTest(Dataset):
    def __init__(self, sample, transform):
        self.data = sample
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join('./train_data/', self.data.loc[idx, 'image'])
        label = self.data.loc[idx,'label']
        image = Image.open(img_name)
        image = self.transform(image)
       
        return {'image': image,'label':label}
		
		
		
class LabelSmoothingLoss(nn.Module):
	def __init__(self, classes, smoothing=0.0, dim=-1):
		super(LabelSmoothingLoss, self).__init__()
		self.confidence = 1.0 - smoothing
		self.smoothing = smoothing
		self.cls = classes
		self.dim = dim

	def forward(self, pred, target):
		pred = pred.log_softmax(dim=self.dim)
		with torch.no_grad():
			# true_dist = pred.data.clone()
			true_dist = torch.zeros_like(pred)
			true_dist.fill_(self.smoothing / (self.cls - 1))
			true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
		return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
		
# data_transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    # transforms.RandomHorizontalFlip(),
    # transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

data_transform = transforms.Compose([
     #transforms.Resize(224),
#transforms.RandomCrop(224),
#transforms.RandomRotation(15),
transforms.RandomResizedCrop(224),
transforms.RandomHorizontalFlip(p=0.5),
transforms.RandomVerticalFlip(p=0.5),
#transforms.Grayscale(num_output_channels=3),
transforms.ColorJitter(0.4, 0.4, 0.4),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
	#transforms.Normalize(mean=[0.47145638718882443, 0.5182074217491532, 0.5559518648087405],std=[0.2878283895184953, 0.2726498177701412, 0.27027624285255336])
	#transforms.Normalize([0.3986, 0.2548, 0.1389], [0.3013, 0.2124, 0.1443])
])


data_transform_test = transforms.Compose([
     transforms.Resize(224),
transforms.CenterCrop(224),
##transforms.RandomHorizontalFlip(p=0.5),
#transforms.Grayscale(num_output_channels=3),
#transforms.ColorJitter(),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
	#transforms.Normalize(mean=[0.47145638718882443, 0.5182074217491532, 0.5559518648087405],std=[0.2878283895184953, 0.2726498177701412, 0.27027624285255336])
	#transforms.Normalize([0.3986, 0.2548, 0.1389], [0.3013, 0.2124, 0.1443])
])

#transforms.Normalize([0.3987, 0.2549, 0.1390], [0.3014, 0.2126, 0.1444])
dataset = OdirDatasetTest(sample=sample,transform=data_transform)

#划分训练集和测试集

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=42)
for train_index, test_index in split.split(sample, sample["label"]):
    train_dataset = sample.loc[train_index].reset_index().drop('index',axis=1)
    test_dataset = sample.loc[test_index].reset_index().drop('index',axis=1)


# sample = shuffle(sample)
# sample.to_csv('shuffled_sample0816.csv',index=False)
# train_size = int(0.8 * len(sample))
# train_dataset=sample[:train_size].reset_index().drop('index',axis=1)
print(train_dataset.head())
print(train_dataset.shape,train_dataset.label.sum())
# test_dataset = sample[train_size:].reset_index().drop('index',axis=1)
print(test_dataset.head())
train_dataset = OdirDatasetTest(sample=train_dataset,transform=data_transform)
test_dataset = OdirDatasetTest(sample=test_dataset,transform=data_transform_test)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=6)

#train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

#获取模型
device = torch.device("cuda:0")
#model = torchvision.models.resnet18(pretrained=True)
#model = torchvision.models.densenet121(pretrained=True)

	
model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x16d_wsl')
#model = resnext101_32x16d_wsl()
#model.avgpool = GeM()
model.fc = nn.Linear(2048, 54)
#model.load_state_dict(torch.load('model_resnext_best.pth'))
#print(model)
#model.classifier = nn.Linear(1024, 40)
# from efficientnet_pytorch import EfficientNet
# model = EfficientNet.from_pretrained('efficientnet-b3') 
# model._fc = nn.Linear(1536,40)
#model.fc= nn.Linear(2048, 40)
#print(model)
ct=0
for child in model.children():
	ct += 1
	#print(ct,child)
	if ct < 5:
		for param in child.parameters():
			param.requires_grad = False
model = model.to(device)


#optimizer = optim.Adam(model.fc.parameters(), lr=0.005)  #,weight_decay=0.0001
optimizer = optim.SGD(model.fc.parameters(), lr=0.005, momentum=0.9,weight_decay=0.001)
#optimizer = optim.RMSprop(model._fc.parameters(), lr=0.01, momentum=0.9,eps=0.001, weight_decay=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10,gamma=0.5)
#scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 50)
criterion = nn.CrossEntropyLoss()
#criterion = LabelSmoothingLoss(54,0.1)
best_acc = 0
best_epoch=0
num_epochs =20
for epoch in range(num_epochs):
	print('Epoch {}/{}'.format(epoch, num_epochs - 1))
	print('-' * 10)
	
	model.train()
	running_loss = 0.0
	tk0 = tqdm(train_dataloader, total=int(len(train_dataloader)))
	counter = 0
	labellist=[]
	pred=[]
	correct = 0
	scheduler.step()
	for i, x_batch in enumerate(tk0):
		inputs = x_batch['image']
		labels = x_batch['label'].view(-1).long()
		#ones = torch.sparse.torch.eye(2)
		#labels = ones.index_select(0,labels)
		inputs = inputs.to(device, dtype=torch.float)
		labels = labels.to(device, dtype=torch.long)
		optimizer.zero_grad()
		#sm = torch.nn.Softmax(dim = 1)
		
		with torch.set_grad_enabled(True):
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			
		running_loss += loss.item() * inputs.size(0)
		counter += 1
		tk0.set_postfix(loss=(running_loss / (counter * train_dataloader.batch_size)))
		for flag in x_batch['label'].view(-1):
			labellist.append(flag.item())
		sm = torch.nn.Softmax(dim = 1)
		pred = sm(outputs).data.max(1, keepdim=True)[1]
		#print(labels.view_as(pred).cpu())
		correct += pred.eq(labels.view_as(pred)).cpu().sum().item()
		#print(correct)
		
	epoch_loss = running_loss / len(train_dataloader)
	print('Train acc:',correct/len(train_dataloader)/batch_size)
	print('Training Loss: {:.4f}'.format(epoch_loss))
	tk1 = tqdm(test_dataloader, total=int(len(test_dataloader)))
	labels=[]
	pred=[]
	correct = 0
	model.eval()
	for i, x_batch in enumerate(tk1):
		inputs = x_batch['image']
		labels = x_batch['label'].view(-1).to(device, dtype=torch.long)
		#print(labels)
		with torch.no_grad():
			sm = torch.nn.Softmax(dim = 1)
			pred = sm(model(inputs.to(device, dtype=torch.float))).data.max(1, keepdim=True)[1]
			correct += pred.eq(labels.view_as(pred)).cpu().sum().item()
	print('Test acc:',correct/len(test_dataloader)/batch_size)
	# if correct/len(test_dataloader)/batch_size>best_acc:
		# torch.save(model.state_dict(), "model_resnext_best.pth")
		# best_acc=correct/len(test_dataloader)/batch_size
		# best_epoch=epoch
# print(best_acc,best_epoch)
torch.save(model.state_dict(), "model_resnext_firsttune_gem"+str(correct/len(test_dataloader)/batch_size)+"_1128.pth")

epoch = 0
optimizer = optim.SGD(model.parameters(), lr=0.00005, momentum=0.9,weight_decay=0.001)
#optimizer = optim.RMSprop(model._fc.parameters(), lr=0.01, momentum=0.9,eps=0.001, weight_decay=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10,gamma=0.5)
#scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 80)
criterion = nn.CrossEntropyLoss()
#criterion = LabelSmoothingLoss(54,0.1)
num_epochs = 80
for epoch in range(num_epochs):
	print('Epoch {}/{}'.format(epoch, num_epochs - 1))
	print('-' * 10)
	
	model.train()
	running_loss = 0.0
	tk0 = tqdm(train_dataloader, total=int(len(train_dataloader)))
	counter = 0
	labellist=[]
	pred=[]
	correct = 0
	scheduler.step()
	for i, x_batch in enumerate(tk0):
		inputs = x_batch['image']
		labels = x_batch['label'].view(-1).long()
		#ones = torch.sparse.torch.eye(2)
		#labels = ones.index_select(0,labels)
		inputs = inputs.to(device, dtype=torch.float)
		labels = labels.to(device, dtype=torch.long)
		optimizer.zero_grad()
		#sm = torch.nn.Softmax(dim = 1)
		with torch.set_grad_enabled(True):
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			
		running_loss += loss.item() * inputs.size(0)
		counter += 1
		tk0.set_postfix(loss=(running_loss / (counter * train_dataloader.batch_size)))
		for flag in x_batch['label'].view(-1):
			labellist.append(flag.item())
		sm = torch.nn.Softmax(dim = 1)
		pred = sm(outputs).data.max(1, keepdim=True)[1]
		#print(labels.view_as(pred).cpu())
		correct += pred.eq(labels.view_as(pred)).cpu().sum().item()
		#print(correct)
		
	epoch_loss = running_loss / len(train_dataloader)
	print('Train acc:',correct/len(train_dataloader)/batch_size)
	print('Training Loss: {:.4f}'.format(epoch_loss))
	tk1 = tqdm(test_dataloader, total=int(len(test_dataloader)))
	labels=[]
	pred=[]
	correct = 0
	running_loss = 0.0
	model.eval()
	for i, x_batch in enumerate(tk1):
		inputs = x_batch['image']
		labels = x_batch['label'].view(-1).to(device, dtype=torch.long)
		#print(labels)
		with torch.no_grad():
			sm = torch.nn.Softmax(dim = 1)
			outputs = model(inputs.to(device, dtype=torch.float))
			pred = sm(outputs).data.max(1, keepdim=True)[1]
			correct += pred.eq(labels.view_as(pred)).cpu().sum().item()
			loss = criterion(outputs, labels)
		running_loss += loss.item() * inputs.size(0)
	epoch_loss = running_loss / len(train_dataloader)
	print('Test acc:',correct/len(test_dataloader)/batch_size)
	print('Test Loss: {:.4f}'.format(epoch_loss))
	if correct/len(test_dataloader)/batch_size>0.97:
		torch.save(model.state_dict(), "model_resnext_secondtune_gem"+str(correct/len(test_dataloader)/batch_size)[:7]+"loss"+str(epoch_loss)+"epoch"+str(epoch)+"_1128.pth")