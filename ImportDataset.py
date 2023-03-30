import os
import torch
from torchvision import datasets, transforms

from PIL import Image
import numpy as np

class LoaderDataset(datasets.DatasetFolder):
    def __init__(self, settings, num_classes=2):
        self.path = settings["DatasetPath"]
        self.num_classes = num_classes
        self.transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                            transforms.Resize([1024, 1024])
                        ])
        self.exclude = []
        self.samples = []
        
        if settings["Phase"] in ["train", "val"]:
            if settings["DatasetTest"] == "pre":
                dataset_path = os.path.join(self.path, "Train/TrueFace_PreSocial")
            elif settings["DatasetTest"] == "post":
                dataset_path = os.path.join(self.path, "Train/TrueFace_PostSocial")
                self.exclude = ['Whatsapp', "StyleGAN3", "09000", "10000", "11000", "12000", "13000", "Twitter", "Telegram"]
        elif settings["Phase"] == "test":
            if settings["DatasetTest"] == "pre":
                dataset_path = os.path.join(self.path, "Test/TrueFace_PreSocial")
            elif settings["DatasetTest"] == "post":
                dataset_path = os.path.join(self.path, "Test/TrueFace_PostSocial")
                self.exclude = ['StyleGAN1', "StyleGAN2", "14000", "Whatsapp"]
                # self.exclude = ['StyleGAN3', "Twitter", "Telegram", "Facebook", "13000"]

        for root_1, dirs_1, files_1 in os.walk(dataset_path, topdown=True):
            for entry in sorted(dirs_1):
                data_folder = os.path.join(root_1, entry)
                if entry == 'FFHQ' or entry == 'Real' or entry == '0_Real':
                    for root, dirs, files in os.walk(data_folder, topdown=True):
                        if all([i not in root for i in self.exclude]):
                            for file in sorted(files):
                                if (file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg")):
                                    item = os.path.join(root, file), torch.tensor([1.0,0.0])
                                    # item = os.path.join(root, file), torch.tensor([0.0])
                                    self.samples.append(item)
                elif (entry=='Fake' or entry=='1_Fake'):
                    for root, dirs, files in os.walk(data_folder, topdown=True):
                        if all([i not in root for i in self.exclude]):
                            for file in sorted(files):
                                if (file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg")):
                                    item = os.path.join(root, file), torch.tensor([0.0,1.0])
                                    # item = os.path.join(root, file), torch.tensor([1.0])
                                    self.samples.append(item)


    def __len__(self):
        return len(self.samples)

    def find_classes(self, directory):
        classes_mapping = {"real": 0, "generated": 1}
        return self.classes, classes_mapping

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.transform(Image.open(path))
        # print(sample.shape[0])
        if sample.shape[0] != 3:
            print("sbagliato {}".format(path))
        return sample, target
    
