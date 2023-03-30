# ResNet_TrueFace
Resnet models used to fine tune and test trueface
## Models and weights
In this repository there are two resnet50 models.

 - The first model called resnet50pt is a pretrained model on imagenet and fine tuned on all trueface (i.e. incremental fine tuning, first on pre-social dataset, then post-social dataset). The corresponding weights file is `SharedDatasets1024BFDResnet50`.
 - The second model called resent50ft is a model inspired to https://github.com/grip-unina/GANimageDetection. Starting from their weights we fine tuned the resnet only on the post-social dataset. The corresponding weights file is `SharedDataset_60k_lr1e-5.pth`.
## Setup
All the parameteres must be set in the json file `settings`. There it is possible to select the model, the learning rate and the batch size to use during training and test.


ssh checkshared@10.196.36.18
scp -r images/ checkshared@10.196.36.18:/media/mmlab/Volume/ivanvalentini/images
det -u checkshared -m 10.196.36.18:8080 experiment create -f config.yaml .


Buongiorno, 
prima di lanciare il primo test vorrei chiedere conferma della configurazione. Questi sono i passaggi che ho svolto:

- Scaricato il progetto di esempio (https://drive.google.com/file/d/1O4Vm67ksRaSPorAQZbYFo-Sq604utSf4/view)
- Clonato al suo interno https://github.com/AntonioStefani/ResNet_TrueFace
- Installato determined sulla mia macchina
- Modificato startup-hook.sh in questo modo perché non mi risulta che siano installati di default torch e pandas (istruzioni per pytorch: https://pytorch.org/get-started/locally/):

```sh
pip install pandas
pip install torch torchvision torchaudio
```

- Creato una struttura delle cartelle in  di questo tipo nel cluster:
```
/media/mmlab/Volume/ForensicsDataset/VipCup/VipCup_Shared/bird/PostSocialDataset/Facebook
/media/mmlab/Volume/ForensicsDataset/VipCup/VipCup_Shared/bird/PostSocialDataset/Telegram
/media/mmlab/Volume/ForensicsDataset/VipCup/VipCup_Shared/bird/PostSocialDataset/Twitter
/media/mmlab/Volume/ForensicsDataset/VipCup/VipCup_Shared/bird/PreSocialDataset
```

- Modificato ResNet_TrueFace/config.json perché usi il modello indicato:
```json
{
    "Model": "resnet50ft", 
    "LoadCheckpointPath": "SharedDatasets1024BFDResnet50",
    "SaveCheckpointPath": "Checkpoints/SharedDataset_60k_lr1e-5.pth",
    "DatasetPath": "images/",
    "LoadCheckpoint": false,
    "Train": false,
    "ImgChannels": 3,
    "Classes": 2,
    "LearningRate": 0.00001,
    "BatchSize": 2,
    "Epochs": 10
}
```

- Modificato config.yaml perché abbia i path corretti:
 
```yaml
name: checkshared
entrypoint: python3 ResNet_TrueFace/main.py

resources:
   slots_per_trial: 1

bind_mounts:
  - host_path: /media/mmlab/Volume/ForensicsDataset/VipCup/VipCup_Shared/bird/PreSocialDataset
    container_path: /media/dataset
    read_only: true
  - host_path: /media/mmlab/Volume/ForensicsDataset/VipCup/VipCup_Shared/bird/PreSocialDataset/
    container_path: /media/checkpoint
    read_only: false

searcher:
  name: single
  metric: accuracy
  max_length: 1
  
max_restarts: 0
```

Ho anche caricato le 1000 immagini originali della categoria bird (in /media/mmlab/Volume/ForensicsDataset/VipCup/VipCup_Shared/bird/PreSocialDataset). E volevo fare il testing delle immagini.
Con i file così configurati mi è sufficiente far partire: 
```
det -u checkshared -m 10.196.36.18:8080 experiment create -f config.yaml .
```
per fare il testing delle immagini, o c'è un errore nella configurazione?


scp /mnt/c/Users/Ivan/Desktop/UniTN/MultimediaDataSecurity/truebees/TrueBees/original_images/fake/birds.zip checkshared@10.196.36.18:/media/mmlab/Volume/ForensicsDataset/VipCup/VipCup_Shared/bird/PreSocialDataset/
