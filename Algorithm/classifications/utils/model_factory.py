from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, transforms, models as m1
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
import shutil
import torch.nn.functional as F
from Algorithm.classifications.pretrainedmodels import models as m2
import torch.onnx

def classification(args):

    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_name = args.modelname
    num_classes = args.classes
    feature_extract = args.freeze_layer
    return initialize_model(model_name, num_classes, feature_extract)

def set_parameter_requires_grad(model, freeze_layer):
    if freeze_layer:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = m1.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "resnet34":
        """ Resnet34
        """
        model_ft = m1.resnet34(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "resnet50":
        """ resnet50
        """
        model_ft = m1.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "resnet101":
        """ resnet101
        """
        model_ft = m1.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "resnet152":
        """ resnet152
        """
        model_ft = m1.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = m1.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "VGG11_bn":
        """ VGG11_bn
        """
        model_ft = m1.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = m1.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes

    elif model_name == "densenet121":
        """ densenet121
        """
        model_ft = m1.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)

    elif model_name == "densenet169":
        """ densenet169
        """
        model_ft = m1.densenet169(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)

    elif model_name == "densenet161":
        """ densenet161
        """
        model_ft = m1.densenet161(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)

    elif model_name == "densenet201":
        """ densenet201
        """
        model_ft = m1.densenet201(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)

    elif model_name == "inception_v3":
        """ inception_v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = m1.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "se_resnet50":
        """se_resnet50
        """
        pretrain = 'imagenet' if use_pretrained else ''
        model_ft = m2.se_resnet50(pretrained=pretrain)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.last_linear.in_features
        model_ft.last_linear = nn.Linear(num_ftrs, num_classes)

    elif model_name == "se_resnet101":
        """se_resnet101
        """
        pretrain = 'imagenet' if use_pretrained else ''
        model_ft = m2.se_resnet101(pretrained=pretrain)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.last_linear.in_features
        model_ft.last_linear = nn.Linear(num_ftrs, num_classes)

    elif model_name == "se_resnet152":
        """se_resnet152
        """
        pretrain = 'imagenet' if use_pretrained else ''
        model_ft = m2.se_resnet152(pretrained=pretrain)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.last_linear.in_features
        model_ft.last_linear = nn.Linear(num_ftrs, num_classes)

    elif model_name == "senet154":
        """senet154
        """
        pretrain = 'imagenet' if use_pretrained else ''
        model_ft = m2.senet154(pretrained=pretrain)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.last_linear.in_features
        model_ft.last_linear = nn.Linear(num_ftrs, num_classes)

    elif model_name == "se_resnext50_32x4d":
        """se_resnext50_32x4d
        """
        pretrain = 'imagenet' if use_pretrained else ''
        model_ft = m2.se_resnext50_32x4d(pretrained=pretrain)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.last_linear.in_features
        model_ft.last_linear = nn.Linear(num_ftrs, num_classes)

    elif model_name == "se_resnext101_32x4d":
        """se_resnext101_32x4d
        """
        pretrain = 'imagenet' if use_pretrained else ''
        model_ft = m2.se_resnext101_32x4d(pretrained=pretrain)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.last_linear.in_features
        model_ft.last_linear = nn.Linear(num_ftrs, num_classes)

    elif model_name == "resnext101_32x4d":
        """resnext101_32x4d
        """
        pretrain = 'imagenet' if use_pretrained else ''
        model_ft = m2.resnext101_32x4d(pretrained=pretrain)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.last_linear.in_features
        model_ft.last_linear = nn.Linear(num_ftrs, num_classes)

    elif model_name == "resnext101_64x4d":
        """resnext101_64x4d
        """
        pretrain = 'imagenet' if use_pretrained else ''
        model_ft = m2.resnext101_64x4d(pretrained=pretrain)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.last_linear.in_features
        model_ft.last_linear = nn.Linear(num_ftrs, num_classes)

    elif model_name == "dpn68":
        """dpn68
        """
        pretrain = 'imagenet' if use_pretrained else ''
        model_ft = m2.dpn68(pretrained=pretrain)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.last_linear.in_features
        model_ft.last_linear = nn.Linear(num_ftrs, num_classes)

    elif model_name == "xception":
        """xception  input size 299
        """
        pretrain = 'imagenet' if use_pretrained else ''
        model_ft = m2.xception(pretrained=pretrain)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.last_linear.in_features
        model_ft.last_linear = nn.Linear(num_ftrs, num_classes)

    elif model_name == "inceptionresnetv2":
        """inceptionresnetv2  input size 299
        """
        pretrain = 'imagenet' if use_pretrained else ''
        model_ft = m2.inceptionresnetv2(pretrained=pretrain)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.last_linear.in_features
        model_ft.last_linear = nn.Linear(num_ftrs, num_classes)

    elif model_name == "bninception":
        """bninception  input size 224
        """
        pretrain = 'imagenet' if use_pretrained else ''
        model_ft = m2.bninception(pretrained=pretrain)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.last_linear.in_features
        model_ft.last_linear = nn.Linear(num_ftrs, num_classes)

    elif model_name == "nasnetamobile":
        """nasnetamobile  input size 224
        """
        pretrain = 'imagenet' if use_pretrained else ''
        model_ft = m2.nasnetamobile(pretrained=pretrain)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.last_linear.in_features
        model_ft.last_linear = nn.Linear(num_ftrs, num_classes)

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft

def train_model(model, dataloaders, criterion, optimizer, save_path, device, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, os.path.join(save_path, "best_weight.pt"))
            if phase == 'val':
                val_acc_history.append(epoch_acc)

            if (epoch % 100 == 0):
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, os.path.join(save_path, "weight_"+str(epoch)+'.pt'))

        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def startTraining(model_name, num_classes, feature_extract,
                  data_transforms, data_dir,
                  batch_size, save_path, num_epochs, lr, weights=None, half=False):

    model_ft = initialize_model(model_name, num_classes, feature_extract)

    if weights:
        model_ft.load_state_dict(torch.load(weights))

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

    print('train class:{}  {}'.format(image_datasets['train'].class_to_idx, len(image_datasets['train'])))
    print('val class:{}  {}'.format(image_datasets['val'].class_to_idx, len(image_datasets['val'])))
    # Create training and validation dataloaders
    dataloaders_dict = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in
    ['train', 'val']}

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)

    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=lr, momentum=0.9)
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, save_path, device, num_epochs=num_epochs,
                                 is_inception=(model_name == "inception"))
