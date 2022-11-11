# ResNet_TrueFace
Resnet models used to fine tune and test trueface
## Models and weights
In this repository there are two resnet50 models.

 - The first model called resnet50pt is a pretrained model on imagenet and fine tuned on all trueface (i.e. incremental fine tuning, first on pre-social dataset, then post-social dataset). The corresponding weights file is `SharedDatasets1024BFDResnet50`.
 - The second model called resent50ft is a model inspired to https://github.com/grip-unina/GANimageDetection. Starting from their weights we fine tuned the resnet only on the post-social dataset. The corresponding weights file is `SharedDataset_60k_lr1e-5.pth`.
## Setup
All the parameteres must be set in the json file `settings`. There it is possible to select the model, the learning rate and the batch size to use during training and test.
