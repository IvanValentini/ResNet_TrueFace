# ----------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------
import os
# Import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import *
import torchvision.transforms as transforms
# Import networks
from resnet50pt import *
from resnet50ft import *
# Import utilities
import pandas as pd
import ImportDataset
import json
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import numpy as np


# ----------------------------------------------------------------------------
# SETTING HYPERPARAMETERS
# ----------------------------------------------------------------------------
# Load settings
f = open ("config.json", "r")
settings = json.loads(f.read())
print("Loaded settings")
# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------
# LOAD DATASET
# ----------------------------------------------------------------------------
# Import Dataset
transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #transforms.Resize([1024,1024])
    ])
data_set = ImportDataset.TuningDatabase(settings["DatasetPath"], transforms)

print("Imported dataset")
print(f"Loaded {len(data_set)} images")

# Generate train e test set
if settings["Train"]:
    train_len = int(len(data_set)*0.8)
    train_set, val_set = random_split(data_set, [train_len, len(data_set) - train_len])
    train_loader = DataLoader(dataset=train_set, batch_size=settings["BatchSize"], shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=settings["BatchSize"], shuffle=True)
else:
    print("Testing!")
    test_loader = DataLoader(dataset=data_set, batch_size=settings["BatchSize"], shuffle=True)


# ----------------------------------------------------------------------------
# INITIALIZE NEURAL NETWORK
# ----------------------------------------------------------------------------
# Initialize model
model = locals()[settings["Model"]](device=device, num_classes=settings["Classes"])
    
best_accuracy = 0
best_epoch = 0
current_epoch = 0
# Loss and optimizer
# criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=settings["LearningRate"])

# Check if checkpoints
if settings["LoadCheckpoint"]==True:
    print("Loaded checkpoints!")
    print(settings["LoadCheckpointPath"])
    checkpoint = torch.load(settings["LoadCheckpointPath"])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

# ----------------------------------------------------------------------------
# CHECK ACCURACY ON DATASET
# ----------------------------------------------------------------------------
def check_accuracy(loader, model, train):
    print("Checking accuracy!")
    if train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")
        
    zerosamples = 0
    zerocorrect = 0
    onesamples = 0
    onecorrect = 0
    num_correct = 0
    num_samples = 0

    y_true = torch.empty(0, dtype=torch.int).to(device=device)
    y_pred = torch.empty(0, dtype=torch.int).to(device=device)
    
    model.eval()
    
    with torch.no_grad():
        with tqdm(loader, unit="batch") as tbatch:
            for batch_idx, (x, y) in enumerate(tbatch):
                tbatch.set_description(f"Batch {batch_idx}")

                x = x.to(device=device)
                # Better way to do this?
                y = y.to(device=device).argmax(dim=1, keepdim=True).squeeze(dim=1).squeeze(dim=1).squeeze(dim=1)
                y_true = (torch.cat((y_true, y),dim=0))                
                scores = nn.Softmax(dim=1)(model(x))
                # scores = torch.sigmoid(model(x))
                _, predictions = scores.max(1)
                # predictions = torch.round(scores).int()
                predictions = predictions.squeeze()
                y_pred = torch.cat((y_pred, predictions.view(1)),dim=0)

                zerosamples += len(y[y==0])
                zerocorrect += (y[y==predictions]==0).sum().item()
                onesamples += len(y[y==1])
                onecorrect += (y[y==predictions]==1).sum().item()
                num_samples += predictions.view(1).size(0)
                num_correct += (predictions == y).sum()
                
                zeroaccuracy=0
                oneaccuracy=0
                if zerosamples != 0:
                    zeroaccuracy = float(zerocorrect)/float(zerosamples)*100
                if onesamples != 0:
                    oneaccuracy = float(onecorrect)/float(onesamples)*100
                accuracy = float(num_correct)/float(num_samples)*100
                tbatch.set_postfix(accuracy=accuracy, real=zeroaccuracy, fake=oneaccuracy)

        # Build confusion matrix
        cf_matrix = confusion_matrix(y_true.cpu().numpy(), y_pred.cpu().numpy())
        df_cm = pd.DataFrame((cf_matrix.T/np.sum(cf_matrix,axis=1)).T *100, index = [i for i in ['real','fake']],
                     columns = [i for i in ['real','fake']])
        print('Confusion_Matrix:\n {}'.format(df_cm))
        print(f'Got tot: {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f} \n')
    
    return accuracy


# ----------------------------------------------------------------------------
# TRAIN NETWORK
# ----------------------------------------------------------------------------
if settings["Train"]:
    for epoch in range(settings["Epochs"]):
        model.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            for batch_idx, (data, targets) in enumerate(tepoch):
                data = data.to(device=device)
                targets = targets.to(device=device)
                
                # forward
                scores = model(data)
                loss = criterion(scores, targets.to(dtype=torch.float))

                # batch insight
                predictions = scores.argmax(dim=1, keepdim=True).squeeze()
                trg = targets.argmax(dim=1, keepdim=True).squeeze()
                
                zerocorrect = (trg[trg==predictions]==0).sum().item()
                zerosamples = len(trg[trg==0])
                if zerosamples != 0:
                    zeroaccuracy = zerocorrect / zerosamples
                else:
                    zeroaccuracy = 1
                
                onecorrect = (trg[trg==predictions]==1).sum().item()
                onesamples = len(trg[trg==1])
                if onesamples != 0:
                    oneaccuracy = onecorrect / onesamples
                else:
                    oneaccuracy = 1
                
                correct = (predictions == trg).sum().item()
                accuracy = correct / settings["BatchSize"]
                
                # backward
                optimizer.zero_grad()
                loss.backward()
                
                # gradient descent or adam step
                optimizer.step()
                
                tepoch.set_postfix(loss=loss.item(), acc_tot=accuracy, acc_fake=oneaccuracy, acc_real=zeroaccuracy)

        current_acc = check_accuracy(val_loader, model, train=False)
        print(current_acc)
        
        # save checkpoint
        current_epoch += 1
        if ((current_epoch-best_epoch > 2) and (current_acc >= best_accuracy-2)) or (current_acc >= best_accuracy):
            best_accuracy = current_acc
            best_epoch = current_epoch

            print("{:.2f}, {:.2f}".format(current_acc,best_accuracy))
            print("{:.2f}, {:.2f}".format(current_epoch,best_epoch))

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, settings["SaveCheckpointPath"])


# ----------------------------------------------------------------------------
# CHECK ACCURACY INVOCATION
# ----------------------------------------------------------------------------
if not(settings["Train"]):
    check_accuracy(test_loader, model, train=False)
    