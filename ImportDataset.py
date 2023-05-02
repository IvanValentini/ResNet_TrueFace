import torch
from torchvision import datasets, models, transforms
import os
from PIL import Image

class TuningDatabase(datasets.DatasetFolder):
    def __init__(self, path, transform):
        self.classes = ["real", "generated"]
        print(path)
        self.samples = []
        self.realdownsamplervariable = 0
        self.fakedownsamplervariable = 0
        self.transform = transform
        for root_1, dirs_1, files_1 in os.walk(path, topdown=True):
            for entry in dirs_1:
                data_folder = os.path.join(root_1, entry)
                if entry == 'FFHQ' or entry == 'Real' or entry == '0_Real':
                    for root, dirs, files in os.walk(data_folder, topdown=True):
                        for file in sorted(files):
                            if (file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"))  and (self.realdownsamplervariable % 1 == 0):
                                item = os.path.join(root, file), torch.tensor([[[1]],[[0]]])
                                # item = os.path.join(root, file), torch.tensor([[0]])
                                self.samples.append(item)
                                self.realdownsamplervariable += 1
                elif (((entry == 'StyleGAN' or entry == 'StyleGAN2') and os.path.basename(path) == 'forensicsDatasets') or entry=='Fake' or entry=='1_Fake'):
                    exclude = set(['code', 'tmp', 'dataStyleGAN2', 'StyleGAN3'])
                    # exclude = set(['StyleGAN', 'StyleGAN2'])
                    for root, dirs, files in os.walk(data_folder, topdown=True):
                        dirs[:] = [d for d in dirs if d not in exclude]
                        self.fakedownsamplervariable = 0
                        for file in sorted(files):
                            if (file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg")) and (self.fakedownsamplervariable % 1 == 0):
                                item = os.path.join(root, file), torch.tensor([[[0]],[[1]]])
                                # item = os.path.join(root, file), torch.tensor([[1]])
                                self.samples.append(item)
                                self.fakedownsamplervariable += 1
                # else:
                #     path = data_folder

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
    
