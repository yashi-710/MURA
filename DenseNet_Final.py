#!/usr/bin/env python
# coding: utf-8

# In[3]:


import time
import copy
import torch
from torchnet import meter
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import os
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import pil_loader


data_cat = ['train', 'valid'] # data categories

def train_model(model, criterion, optimizer, dataloaders, scheduler, 
                dataset_sizes, num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    costs = {x:[] for x in data_cat} # for storing costs per epoch
    accs = {x:[] for x in data_cat} # for storing accuracies per epoch
    print('Train batches:', len(dataloaders['train']))
    print('Valid batches:', len(dataloaders['valid']), '\n')
    for epoch in range(num_epochs):
        confusion_matrix = {x: meter.ConfusionMeter(2, normalized=True) 
                            for x in data_cat}
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in data_cat:
            model.train(phase=='train')
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for i, data in enumerate(dataloaders[phase]):
                # get the inputs
                print(i, end='\r')
                inputs = data['images'][0]
                labels = data['label'].type(torch.FloatTensor)
                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                outputs = model(inputs)
                outputs = torch.mean(outputs)
                loss = criterion(outputs, labels, phase)
                running_loss += loss.data[0]
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                preds = (outputs.data > 0.5).type(torch.cuda.FloatTensor)
                running_corrects += torch.sum(preds == labels.data)
                #confusion_matrix[phase].add(preds, labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            costs[phase].append(epoch_loss)
            accs[phase].append(epoch_acc)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            print('Confusion Meter:\n', confusion_matrix[phase].value())
            # deep copy the model
            if phase == 'valid':
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
        time_elapsed = time.time() - since
        print('Time elapsed: {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: {:4f}'.format(best_acc))
    plot_training(costs, accs)
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def get_metrics(model, criterion, dataloaders, dataset_sizes, phase='valid'):
    '''
    Loops over phase (train or valid) set to determine acc, loss and 
    confusion meter of the model.
    '''
    confusion_matrix = meter.ConfusionMeter(2, normalized=True)
    running_loss = 0.0
    running_corrects = 0
    for i, data in enumerate(dataloaders[phase]):
        print(i, end='\r')
        labels = data['label'].type(torch.FloatTensor)
        inputs = data['images'][0]
        # wrap them in Variable
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        # forward
        outputs = model(inputs)
        outputs = torch.mean(outputs)
        loss = criterion(outputs, labels, phase)
        # statistics
        running_loss += loss.data[0] * inputs.size(0)
        preds = (outputs.data > 0.5).type(torch.cuda.FloatTensor)
        running_corrects += torch.sum(preds == labels.data)
        confusion_matrix.add(preds, labels.data)

    loss = running_loss / dataset_sizes[phase]
    acc = running_corrects / dataset_sizes[phase]
    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, loss, acc))
    print('Confusion Meter:\n', confusion_matrix.value())


# In[3]:




def plot_training(costs, accs):
    '''
    Plots curve of Cost vs epochs and Accuracy vs epochs for 'train' and 'valid' sets during training
    '''
    train_acc = accs['train']
    valid_acc = accs['valid']
    train_cost = costs['train']
    valid_cost = costs['valid']
    epochs = range(len(train_acc))

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1,)
    plt.plot(epochs, train_acc)
    plt.plot(epochs, valid_acc)
    plt.legend(['train', 'valid'], loc='upper left')
    plt.title('Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_cost)
    plt.plot(epochs, valid_cost)
    plt.legend(['train', 'valid'], loc='upper left')
    plt.title('Cost')
    
    plt.show()

def n_p(x):
    '''convert numpy float to Variable tensor float'''    
    return Variable(torch.cuda.FloatTensor([x]), requires_grad=False)

def get_count(df, cat):
    '''
    Returns number of images in a study type dataframe which are of abnormal or normal
    Args:
    df -- dataframe
    cat -- category, "positive" for abnormal and "negative" for normal
    '''
    return df[df['Path'].str.contains(cat)]['Count'].sum()


if __name__=='main':
    pass


# In[4]:




data_cat = ['train', 'valid'] # data categories

def get_study_level_data(study_type):
    """
    Returns a dict, with keys 'train' and 'valid' and respective values as study level dataframes, 
    these dataframes contain three columns 'Path', 'Count', 'Label'
    Args:
        study_type (string): one of the seven study type folder names in 'train/valid/test' dataset 
    """
    study_data = {}
    study_label = {'positive': 1, 'negative': 0}
    for phase in data_cat:
        BASE_DIR = 'D:\\Major Project\\MURA-v1.1\\%s\\%s\\' % (phase, study_type)
        patients = list(os.walk(BASE_DIR))[0][1] # list of patient folder names
        study_data[phase] = pd.DataFrame(columns=['Path', 'Count', 'Label'])
        i = 0
        for patient in tqdm(patients): # for each patient folder
            for study in os.listdir(BASE_DIR + patient): # for each study in that patient folder
                label = study_label[study.split('_')[1]] # get label 0 or 1
                path = BASE_DIR + patient + '/' + study + '/' # path to this study
                study_data[phase].loc[i] = [path, len(os.listdir(path)), label] # add new row
                i+=1
    return study_data

class ImageDataset(Dataset):
    """training dataset."""

    def __init__(self, df, transform=None):
        """
        Args:
            df (pd.DataFrame): a pandas DataFrame with image path and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        study_path = self.df.iloc[idx, 0]
        count = self.df.iloc[idx, 1]
        images = []
        for i in range(count):
            image = pil_loader(study_path + 'image%s.png' % (i+1))
            images.append(self.transform(image))
        images = torch.stack(images)
        label = self.df.iloc[idx, 2]
        sample = {'images': images, 'label': label}
        return sample

def get_dataloaders(data, batch_size=4, study_level=False):
    '''
    Returns dataloader pipeline with data augmentation
    '''
    data_transforms = {
        'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ]),
        'valid': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {x: ImageDataset(data[x], transform=data_transforms[x]) for x in data_cat}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in data_cat}
    return dataloaders

if __name__=='main':
    pass


# In[5]:




__all__ = ['DenseNet', 'densenet169']


model_urls = {
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
}

def densenet169(pretrained=False, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                     **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['densenet169']), strict=False)
    return model

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        
        # Linear layer
        # self.classifier = nn.Linear(num_features, 1000)
        # self.fc = nn.Linear(1000, 1)
        
        self.fc = nn.Linear(num_features, 1)
        
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        # out = F.relu(self.classifier(out))
        out = F.sigmoid(self.fc(out))
        return out


# In[6]:


if __name__=="__main__":
   
    #from densenet import densenet169
    #from utils import plot_training, n_p, get_count
    #from train import train_model, get_metrics
    #from pipeline import get_study_level_data, get_dataloaders

    # #### load study level dict data
    study_data = get_study_level_data(study_type='XR_HUMERUS')

    # #### Create dataloaders pipeline
    data_cat = ['train', 'valid'] # data categories
    dataloaders = get_dataloaders(study_data, batch_size=1)
    dataset_sizes = {x: len(study_data[x]) for x in data_cat}

    # #### Build model
    # tai = total abnormal images, tni = total normal images
    tai = {x: get_count(study_data[x], 'positive') for x in data_cat}
    tni = {x: get_count(study_data[x], 'negative') for x in data_cat}
    Wt1 = {x: n_p(tni[x] / (tni[x] + tai[x])) for x in data_cat}
    Wt0 = {x: n_p(tai[x] / (tni[x] + tai[x])) for x in data_cat}

    print('tai:', tai)
    print('tni:', tni, '\n')
    print('Wt0 train:', Wt0['train'])
    print('Wt0 valid:', Wt0['valid'])
    print('Wt1 train:', Wt1['train'])
    print('Wt1 valid:', Wt1['valid'])

    class Loss(torch.nn.modules.Module):
        def __init__(self, Wt1, Wt0):
            super(Loss, self).__init__()
            self.Wt1 = Wt1
            self.Wt0 = Wt0
        
        def forward(self, inputs, targets, phase):
            loss = - (self.Wt1[phase] * targets * inputs.log() + self.Wt0[phase] * (1 - targets) * (1 - inputs).log())
            return loss

    model = densenet169(pretrained=True)
    model = model.cuda()

    criterion = Loss(Wt1, Wt0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, verbose=True)

    # #### Train model
    model = train_model(model, criterion, optimizer, dataloaders, scheduler, dataset_sizes, num_epochs=5)

    torch.save(model.state_dict(), 'models/model.pth')

    get_metrics(model, criterion, dataloaders, dataset_sizes)


# In[ ]:




