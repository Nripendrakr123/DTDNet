import os
from torchvision import transforms
from model import *
from dataloader import *
import torch
import cv2
import numpy as np
import shutil
from collections import defaultdict



data_path = "/home/nripendra/BBAVectors-Oriented-Object-Detection/crop_images"
transforms_test = transforms.Compose([transforms.Resize((128, 128)),transforms.ToTensor()])
test_data_set = TEST_ImageDataset(data_set_path=data_path, transforms=transforms_test)
test_loader = DataLoader(test_data_set, batch_size=1, shuffle=True)



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_classes = 8
class_names = ['Implant', 'ToothBud', 'Tooth', 'ImpTooth', 'RCT+CRW', 'I+CRW', 'Restoration', 'RCT']
print(class_names)
print(len(class_names))
custom_model = CustomConvNet(num_classes=num_classes).to(device)
PATH = "model.pth"
custom_model.load_state_dict(torch.load(PATH , map_location=device))


output_dir = "./image_output/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


class Model_infrance:
    def __init__(self , model_path = "/home/nripendra/Multiple_treatment&Diagnosis/Experiment/oriented_detection/TND_classification/weight/model.pth"):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        num_classes = 8
        self.class_names = ['Implant', 'ToothBud', 'Tooth', 'ImpTooth', 'RCT+CRW', 'I+CRW', 'Restoration', 'RCT']
        self.model_test = CustomConvNet(num_classes=num_classes).to(self.device)

        # visualkeras.layered_view(self.model_test).show()  # display using your system viewer
        # visualkeras.layered_view(self.model_test, to_file='output.png')  # write to disk
        # visualkeras.layered_view(self.model_test, to_file='output.png').show()

        self.model_PATH = model_path
        self.model_test.load_state_dict(torch.load(self.model_PATH, map_location=self.device))
        self.model_test.eval()
        self.transforms_test = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

    def get_output(self , path ):
        image  = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = transforms_test(image).unsqueeze(0)
        image = image.to(self.device)
        outputs = self.model_test(image)
        _, predicted = torch.max(outputs.data, 1)
        idx = predicted.data.cpu().numpy()[0]
        label_name = self.class_names[idx]
        '''
         outputs = custom_model(images)
        _, predicted = torch.max(outputs.data, 1)
        idx = predicted.data.cpu().numpy()[0]
        pil_name = class_names[idx]
        '''
        return label_name


def get_data(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = transforms_test(image).unsqueeze(0)
    return  image

custom_model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    cls_model = Model_infrance()
    d = defaultdict(int)
    for i ,item in enumerate(test_loader):
        images = item['image'].to(device)
        image_path = item['path'][0]
        #cv_image = get_data(image_path).to(device)
        # outputs = custom_model(images)
        # _, predicted = torch.max(outputs.data, 1)
        # idx = predicted.data.cpu().numpy()[0]
        # pil_name = class_names[idx]
        cv_name  = cls_model.get_output(image_path)
        d[cv_name] += 1
        path = os.path.join(output_dir , cv_name )
        if not os.path.exists(path):
            os.makedirs(path)
        save_path  = path + "/"

        shutil.copy(item['path'][0] , save_path)
    print(d)
