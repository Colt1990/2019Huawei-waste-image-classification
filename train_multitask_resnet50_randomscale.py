import numpy as np
import pandas as pd
import torchvision
import torch.nn as nn
from tqdm import tqdm
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import torch
from torchvision import transforms
from torchvision.transforms import *
import os
import numpy as np
import cv2
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import f1_score,roc_auc_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import math
import torch.nn.functional as F
from senet import SENet,SEResNeXtBottleneck,pretrained_settings,initialize_pretrained_model
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
import albumentations as albu
from torch.autograd import Variable


batch_size=32
#torch.multiprocessing.set_sharing_strategy('file_system')
sample = pd.read_csv('Train_label.csv')
sample['big'] = sample['Code']

sample['middle'] = sample['Code'] 
sample = sample[~sample.Code.str.contains(';')]
sample['Code'] = sample.Code.map(lambda x:int(x)-1)
sample = sample[sample['Code']!=28]
sample=sample.reset_index().drop('index',axis=1)


#创建大分类标签
def creat_big(file):
   code = np.zeros(3)
   if ';' in file:
        return ';'
   else:
            file = int(file)
            if file<8:
                code=0
            elif 8<=file and file<15:
                code=1
            else:
                code=2
   return code
sample['big'] = sample['big'].map(lambda x:creat_big(x))

#创建中分类标签
def creat_middle(file):
   code = np.zeros(10)
   if ';' in file:
        return ';'
   else:
            file = int(file)
            if file<6:
                code=0
            elif 6<=file and file<8:
                code=1
            elif 8<=file and file<12:
                code=2
            elif file==12:
                code=3
            elif file==13 or file==14:
                code=4
            elif file==15 or file==16:
                code=5
            elif 17<=file and file<20:
                code=6
            elif file==20 or file==21:
                code=7
            elif file==22 or file==23:
                code=8
            else:
                code=9
   return code
sample['middle'] = sample['middle'].map(lambda x:creat_middle(x))

#sample = sample[:50]
print(sample.head())

class OdirDatasetTest(Dataset):
    def __init__(self, sample, transform=None):
        self.data = sample
        self.transform = transform
        self.preprocess = get_preprocess()
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join('Train/', self.data.loc[idx, 'FileName'])
        label = self.data.loc[idx,'Code']
        middle = self.data.loc[idx,'middle']
        big = self.data.loc[idx,'big']
        image = cv2.imread(img_name)
        #image = Image.open(img_name)
        #if image.mode!='RGB':
        #    gray_rgb = transforms.Grayscale(num_output_channels=3)
        #    image = gray_rgb(image)
        if self.transform is not None:
                image = self.transform(image=image)
                #img = self.preprocess(image=image['image'])['image']# size size 3
                img = image['image']
        else:
                img = self.preprocess(image=image)['image']
        img = torch.from_numpy(img).permute(2,1,0).float()
        #print(img.shape)
        return {'image': img,'label':label,'middle':middle,'big':big}



def se_resnext101_32x4d(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNeXtBottleneck, [3, 4, 23, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnext101_32x4d'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model



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
		
class cloudnet(nn.Module):
        def __init__(self):
                super(cloudnet,self).__init__()
                self.basemodel= torchvision.models.resnet18(pretrained=False)
                #self.basemodel = se_resnext101_32x4d(pretrained=None)
                #self.basemodel.fc = nn.Linear(2048, 29)
                #self.basemodel.load_state_dict(torch.load('se_resnext101_32x4d-3b2fe3d8.pth'))
                #self.basemodel.load_state_dict(torch.load('../pretrainmodel/resnet50-19c8e357.pth'))
                self.basemodel.load_state_dict(torch.load('../pretrainmodel/resnet18-5c106cde.pth'))
                modules=list(self.basemodel.children())[:-2]
                self.basemodel=nn.Sequential(*modules)
                #print(self.basemodel)
                #self.fc_middle = nn.Linear(2048,10)
                #self.fc_big = nn.Linear(2048,3)
                self.avg = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(512,29)
        def forward(self,x):
                x = torch.flatten(self.avg(self.basemodel(x)), 1)#.view(batch_size,-1)
                #middle = self.fc_middle(x)
                #big = self.fc_big(x)
                #x = torch.cat([x,middle,big],1)
                x = self.fc(x)
                return x


class binet(nn.Module):
        def __init__(self):
                super(binet,self).__init__()
                #self.basemodel= torchvision.models.resnet50(pretrained=False)
                self.basemodel = se_resnext101_32x4d(pretrained=None)
                print(self.basemodel)
                #self.basemodel.last_linear = nn.Linear(2048, 29)
                #self.basemodel.load_state_dict(torch.load('model_sencond_seresnext101_32_4_0.54015_1005_.bin'))
                #self.basemodel.load_state_dict(torch.load('model_seresnext_aug_0.55093_lastlinear_1025_.bin'))
                self.basemodel.load_state_dict(torch.load('se_resnext101_32x4d-3b2fe3d8.pth'))
                modules=list(self.basemodel.children())[:-2]
                self.basemodel=nn.Sequential(*modules)
                self.fc = nn.Sequential(nn.Linear(2048**2,29))
        def forward(self,x):
                x = self.basemodel(x)
                b = x.size(0)
                x = x.view(b,2048,7**2)
                x = (torch.bmm(x,torch.transpose(x,1,2))/7**2).view(b,-1)
                x = nn.functional.normalize(torch.sign(x)*torch.sqrt(torch.abs(x)+1e-10))
                x = self.fc(x)
                return(x)






#######################loss#############################
def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    new_whale_indexs = (labels == 29*2).nonzero()
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
    for i in new_whale_indexs:
        is_pos[i, :] = 0
        is_pos[:, i] = 0
        is_pos[i, i] = 1

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        (dist_mat * is_pos.float()).contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    temp = dist_mat * is_neg.float()
    temp[temp == 0] = 10e5
    dist_an, relative_n_inds = torch.min(
        (temp).contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an

def shortest_dist(dist_mat):
  """Parallel version.
  Args:
    dist_mat: pytorch Variable, available shape:
      1) [m, n]
      2) [m, n, N], N is batch size
      3) [m, n, *], * can be arbitrary additional dimensions
  Returns:
    dist: three cases corresponding to `dist_mat`:
      1) scalar
      2) pytorch Variable, with shape [N]
      3) pytorch Variable, with shape [*]
  """
  m, n = dist_mat.size()[:2]
  # Just offering some reference for accessing intermediate distance.
  dist = [[0 for _ in range(n)] for _ in range(m)]
  for i in range(m):
    for j in range(n):
      if (i == 0) and (j == 0):
        dist[i][j] = dist_mat[i, j]
      elif (i == 0) and (j > 0):
        dist[i][j] = dist[i][j - 1] + dist_mat[i, j]
      elif (i > 0) and (j == 0):
        dist[i][j] = dist[i - 1][j] + dist_mat[i, j]
      else:
        dist[i][j] = torch.min(dist[i - 1][j], dist[i][j - 1]) + dist_mat[i, j]
  dist = dist[-1][-1]
  return dist

def local_dist(x, y):
  """
  Args:
    x: pytorch Variable, with shape [M, m, d]
    y: pytorch Variable, with shape [N, n, d]
  Returns:
    dist: pytorch Variable, with shape [M, N]
  """
  M, m, d = x.size()
  N, n, d = y.size()
  x = x.contiguous().view(M * m, d)
  y = y.contiguous().view(N * n, d)
  # shape [M * m, N * n]
  dist_mat = euclidean_dist(x, y)
  dist_mat = (torch.exp(dist_mat) - 1.) / (torch.exp(dist_mat) + 1.)
  # shape [M * m, N * n] -> [M, m, N, n] -> [m, n, M, N]
  dist_mat = dist_mat.contiguous().view(M, m, N, n).permute(1, 3, 0, 2)
  # shape [M, N]
  dist_mat = shortest_dist(dist_mat)
  return dist_mat
# class TripletLoss(object):
#     """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
#     Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
#     Loss for Person Re-Identification'."""
#
#     def __init__(self, margin=None):
#         self.margin = margin
#         if margin is not None:
#             self.ranking_loss = nn.MarginRankingLoss(margin=margin)
#         else:
#             self.ranking_loss = nn.SoftMarginLoss()
#
#     def __call__(self, feat, labels, normalize_feature=False):
#         # indexs = (labels != 5004).nonzero().view(-1)
#         # global_feat = global_feat[indexs].contiguous()
#         # labels = labels[indexs].contiguous()
#         if normalize_feature:
#             feat = normalize(feat, axis=-1)
#         if len(feat.size()) == 3:
#             dist_mat = local_dist(feat, feat)
#         else:
#             dist_mat = euclidean_dist(feat, feat)
#         dist_ap, dist_an = hard_example_mining(
#             dist_mat, labels)
#         y = dist_an.new().resize_as_(dist_an).fill_(1)
#         if self.margin is not None:
#             loss = self.ranking_loss(dist_an, dist_ap, y)
#         else:
#             loss = self.ranking_loss(dist_an - dist_ap, y)
#         return loss, dist_ap, dist_an
class TripletLoss(object):
  """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
  Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
  Loss for Person Re-Identification'."""
  def __init__(self, margin=None):
    self.margin = margin
    if margin is not None:
      self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    else:
      self.ranking_loss = nn.SoftMarginLoss()

  def __call__(self, dist_ap, dist_an):
    """
    Args:
      dist_ap: pytorch Variable, distance between anchor and positive sample,
        shape [N]
      dist_an: pytorch Variable, distance between anchor and negative sample,
        shape [N]
    Returns:
      loss: pytorch Variable, with shape [1]
    """
    y = Variable(dist_an.data.new().resize_as_(dist_an.data).fill_(1))
    if self.margin is not None:
      loss = self.ranking_loss(dist_an, dist_ap, y)
    else:
      loss = self.ranking_loss(dist_an - dist_ap, y)
    return loss



def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def batch_euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [N, m, d]
      y: pytorch Variable, with shape [N, n, d]
    Returns:
      dist: pytorch Variable, with shape [N, m, n]
    """
    assert len(x.size()) == 3
    assert len(y.size()) == 3
    assert x.size(0) == y.size(0)
    assert x.size(-1) == y.size(-1)

    N, m, d = x.size()
    N, n, d = y.size()

    # shape [N, m, n]
    xx = torch.pow(x, 2).sum(-1, keepdim=True).expand(N, m, n)
    yy = torch.pow(y, 2).sum(-1, keepdim=True).expand(N, n, m).permute(0, 2, 1)
    dist = xx + yy
    dist.baddbmm_(1, -2, x, y.permute(0, 2, 1))
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist




def batch_local_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [N, m, d]
      y: pytorch Variable, with shape [N, n, d]
    Returns:
      dist: pytorch Variable, with shape [N]
    """
    assert len(x.size()) == 3
    assert len(y.size()) == 3
    assert x.size(0) == y.size(0)
    assert x.size(-1) == y.size(-1)

    # shape [N, m, n]
    dist_mat = batch_euclidean_dist(x, y)
    dist_mat = (torch.exp(dist_mat) - 1.) / (torch.exp(dist_mat) + 1.)
    # shape [N]
    dist = shortest_dist(dist_mat.permute(1, 2, 0))
    return dist



def global_loss(tri_loss, global_feat, labels, normalize_feature=False):
    """
    Args:
      tri_loss: a `TripletLoss` object
      global_feat: pytorch Variable, shape [N, C]
      labels: pytorch LongTensor, with shape [N]
      normalize_feature: whether to normalize feature to unit length along the
        Channel dimension
    Returns:
      loss: pytorch Variable, with shape [1]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
      =============
      For Debugging
      =============
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      ===================
      For Mutual Learning
      ===================
      dist_mat: pytorch Variable, pairwise euclidean distance; shape [N, N]
    """
    if normalize_feature:
        global_feat = normalize(global_feat, axis=-1)
    # shape [N, N]
    dist_mat = euclidean_dist(global_feat, global_feat)
    dist_ap, dist_an = hard_example_mining(
        dist_mat, labels, return_inds=False)
    loss = tri_loss(dist_ap, dist_an)
    return loss, dist_ap, dist_an, dist_mat


def local_loss(
        tri_loss,
        local_feat,
        labels=None,
        p_inds=None,
        n_inds=None,
        normalize_feature=False):
    """
    Args:
      tri_loss: a `TripletLoss` object
      local_feat: pytorch Variable, shape [N, H, c] (NOTE THE SHAPE!)
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
      labels: pytorch LongTensor, with shape [N]
      normalize_feature: whether to normalize feature to unit length along the
        Channel dimension
    If hard samples are specified by `p_inds` and `n_inds`, then `labels` is not
    used. Otherwise, local distance finds its own hard samples independent of
    global distance.
    Returns:
      loss: pytorch Variable,with shape [1]
      =============
      For Debugging
      =============
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      ===================
      For Mutual Learning
      ===================
      dist_mat: pytorch Variable, pairwise local distance; shape [N, N]
    """
    if normalize_feature:
        local_feat = normalize(local_feat, axis=-1)
    if p_inds is None or n_inds is None:
        dist_mat = local_dist(local_feat, local_feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels, return_inds=False)
        loss = tri_loss(dist_ap, dist_an)
        return loss, dist_ap, dist_an, dist_mat
    else:
        dist_ap = batch_local_dist(local_feat, local_feat[p_inds])
        dist_an = batch_local_dist(local_feat, local_feat[n_inds])
        loss = tri_loss(dist_ap, dist_an)
        return loss, dist_ap, dist_an
        
        
        
def sigmoid_loss(results, labels, topk=10):
    if len(results.shape) == 1:
        results = results.view(1, -1)
    batch_size, class_num = results.shape
    #labels = labels.view(-1, 1)
    #one_hot_target = torch.zeros(batch_size, class_num).cuda().scatter_(1, labels, 1)
    one_hot_target = labels
    #lovasz_loss = lovasz_hinge(results, one_hot_target)
    error = torch.abs(one_hot_target - torch.sigmoid(results))
    error = error.topk(topk, 1, True, True)[0].contiguous()
    target_error = torch.zeros_like(error).float().cuda()
    error_loss = nn.BCELoss(reduce=True)(error, target_error)
    return error_loss
   
    
    
def f1_loss(predict, target):
    predict = torch.sigmoid(predict)
    #sm = torch.nn.Softmax(dim = 1)
    #predict = sm(predict)
    batch_size, class_num = predict.shape
    #target= target.view(-1, 1)
    #target = torch.zeros(batch_size, class_num).cuda().scatter_(1, target, 1)
    #print(target.shape)
    predict = torch.clamp(predict * (1-target), min=0.01) + predict * target
    tp = predict * target
    tp = tp.sum(dim=0)
    precision = tp / (predict.sum(dim=0) + 1e-8)
    recall = tp / (target.sum(dim=0) + 1e-8)
    f1 = 2 * (precision * recall / (precision + recall + 1e-8))
    return 1 - f1.mean()
###################loss##############################################







def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output       



class RandomCropIfNeeded():
    def __init__(self, height, width, always_apply=False, p=1.0):
        #super(self).__init__(always_apply, p)
        self.height = height
        self.width = width

    def __call__(self, img, h_start=0, w_start=0, **params):
        h, w= img.size
        return functional.resized_crop(img,h_start, w_start,min(self.height, h), min(self.width, w),(224,224))

# data_transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    # transforms.RandomHorizontalFlip(),
    # transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])


def get_training_augmentation():
    #just for train
    train_transform = [
                #albu.ToGray(p=0.2),
                albu.GaussNoise(p=0.2),
                #albu.NoOp(p=1),
                #albu.GaussianBlur(p=0.2),
                albu.Rotate(limit=(-20, 20), p=0.5),
                albu.RandomResizedCrop(224, 224, scale=(0.5, 1.0), ratio=(0.8, 1.2), p=1.0),
        
        albu.HorizontalFlip(p=0.5),
        albu.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]
    return albu.Compose(train_transform)

def get_preprocess(size=224):
    res=[albu.SmallestMaxSize(size),
        #albu.Resize(size, size),
        albu.CenterCrop(size, size),
        albu.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        ),
    ]
    return albu.Compose(res)



data_transform = transforms.Compose([
 #RandomCropIfNeeded(448,448),
 #transforms.Resize(224),
#transforms.CenterCrop(224),
transforms.RandomResizedCrop(224),
#transforms.RandomCrop(224),
transforms.RandomHorizontalFlip(p=0.5), 
#transforms.RandomVerticalFlip(p=0.5),
#transforms.RandomHorizontalFlip(p=0.5),
#transforms.Grayscale(num_output_channels=3),
#transforms.ColorJitter(0.3, 0.3, 0.3),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
	#transforms.Normalize([0.3986, 0.2548, 0.1389], [0.3013, 0.2124, 0.1443])
])

data_transform_test = transforms.Compose([
     # RandomCropIfNeeded(448,448),
 transforms.Resize(224),
transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
	#transforms.Normalize([0.3986, 0.2548, 0.1389], [0.3013, 0.2124, 0.1443])
])

dataset = OdirDatasetTest(sample=sample,transform=data_transform)

#划分训练集和测试集
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=42)
for train_index, test_index in split.split(sample, sample["Code"]):
    train_dataset = sample.loc[train_index].reset_index().drop('index',axis=1)
    test_dataset = sample.loc[test_index].reset_index().drop('index',axis=1)

#sample = shuffle(sample)
#train_size = int(0.8 * len(sample))
#train_dataset=sample[:train_size].reset_index().drop('index',axis=1)
print(train_dataset.head())
#print(train_dataset.shape,train_dataset.label.sum())
#test_dataset = sample[train_size:].reset_index().drop('index',axis=1)
print(test_dataset.head())
train_transform = get_training_augmentation()
train_dataset = OdirDatasetTest(sample=train_dataset,transform=train_transform)
test_dataset = OdirDatasetTest(sample=test_dataset,transform=None)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

#train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

#获取模型
device = torch.device("cuda:0")
#model = torchvision.models.resnet18(pretrained=False)
#model.load_state_dict(torch.load('H:/resnet18-5c106cde.pth'))
#model.fc = nn.Linear(512, 29)
#model = binet()
model = cloudnet()
#model = se_resnext101_32x4d(pretrained=None)

#model.load_state_dict(torch.load('se_resnext101_32x4d-3b2fe3d8.pth'))
#model.last_linear = nn.Linear(2048, 29)

#model.fc = metricnet()
#model.load_state_dict(torch.load('H:/huawei2019/model.bin'))
# from efficientnet_pytorch import EfficientNet
# model = EfficientNet.from_pretrained('efficientnet-b2')
# model._fc = nn.Linear(1408, 40)
#model.unfreeze()
model = model.to(device)
#tune = list(model.layer4.parameters())+list(model.avgpool.parameters())+list(model.fc.parameters())
#optimizer = optim.Adam(model.fc.parameters(), lr=0.005)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9,weight_decay=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10,gamma=0.9)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=0, last_epoch=-1)
criterion_test = nn.CrossEntropyLoss()
#criterion = nn.BCELoss()
criterion = LabelSmoothingLoss(29,0.1)
num_epochs = 100
for epoch in range(num_epochs):
	print('Epoch {}/{}'.format(epoch, num_epochs - 1))
	print('-' * 10)
	scheduler.step()
	model.train()
	running_loss = 0.0
	tk0 = tqdm(train_dataloader, total=int(len(train_dataloader)))
	counter = 0
	labellist=[]
	pred=[]
	correct = 0
	
	for _, x_batch in enumerate(tk0):
		inputs = x_batch['image']
		labels = x_batch['label'] 
		big_labels = x_batch['big'] 
		middle_labels = x_batch['middle'] 
		#for flag in x_batch['label']:
		#	labellist.append(flag.detach().numpy())
		#print(labellist)        
		#ones = torch.sparse.torch.eye(2)
		#labels = ones.index_select(0,labels)
		inputs = inputs.to(device, dtype=torch.float)
		labels = labels.to(device, dtype=torch.long)
		big_labels= big_labels.to(device, dtype=torch.long)
		middle_labels= middle_labels.to(device, dtype=torch.long)
		optimizer.zero_grad()
		sm = torch.nn.Softmax(dim = 1)
		sig = nn.Sigmoid()
		with torch.set_grad_enabled(True):
			outputs = model(inputs)
			#loss = f1_loss(outputs, labels)
			#print(outputs.shape)
			loss =criterion(outputs, labels)#+criterion(outputs[:,29:32], big_labels)+criterion(outputs[:,32:], middle_labels)
			#print(loss)
			
			loss.backward()
			optimizer.step()
		running_loss += loss.item() * inputs.size(0)
		counter += 1
		#outputs = sig(outputs)
		#tk0.set_postfix(loss=(running_loss / (counter * train_dataloader.batch_size)))
		
		#print(pred)
		#sm = torch.nn.Softmax(dim = 1)
		tk0.set_postfix(loss=(running_loss / (counter * train_dataloader.batch_size)))
		pred = sm(outputs).data.max(1, keepdim=True)[1]
		#print(labels.view_as(pred).cpu())
		correct += pred.eq(labels.view_as(pred)).cpu().sum().item()
		#tk0.close()
        #print(correct)
	#print(np.array(labellist).shape)
	#print(np.array(pred).shape)
	epoch_loss = running_loss / len(train_dataloader)
	print('Train acc:',correct/len(train_dataset))
	print('Training Loss: {:.4f}'.format(epoch_loss))
	tk1 = tqdm(test_dataloader, total=int(len(test_dataloader)))
	labellist=[]
	preds=[]
	correct = 0
	running_loss = 0.0
	model.eval()
	for i, x_batch in enumerate(tk1):
		inputs = x_batch['image']
		labels = x_batch['label'].to(device, dtype=torch.long)
		for flag in x_batch['label']:
			labellist.append(flag.detach().numpy())

		sig = nn.Sigmoid()
		with torch.no_grad():
			outputs = model(inputs.to(device, dtype=torch.float))
			
		loss =criterion_test(outputs, labels)
		running_loss += loss.item() * inputs.size(0)
		pred = sm(outputs).data.max(1, keepdim=True)[1]
		correct += pred.eq(labels.view_as(pred)).cpu().sum().item()
		for i in pred.cpu():
			preds.append(i.item())
	epoch_loss = running_loss / len(train_dataloader)
	print('Test f1_score:',f1_score(labellist,preds,average='macro'))
	print('acc:',correct/len(test_dataset))
	print('Test Loss: {:.4f}'.format(epoch_loss))
	print(classification_report(labellist,preds))
	if correct/len(test_dataset)>0.55:
		torch.save(model.state_dict(), "model_res101_aug_strong_"+str(correct/len(test_dataset))[:7]+"_1025_.bin")

		
