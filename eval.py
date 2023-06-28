#%%
import torch
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
from datasets import find_dataset_def
from loss import find_loss_def
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import scipy
import random, argparse
from model import find_model_def
from utils import *

parser = argparse.ArgumentParser(description='CVIP Lab Finger Vein Recognition as FVR')

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')

args = parser.parse_args()

print_args(args)

# 시드 값을 설정합니다.
seed_value = 777

random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
np.random.seed(seed_value)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
SiameseResNet = find_model_def('SiameseResNet')
model = SiameseResNet().to(device)

model.to(device)
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(args.loadckpt))


# Prepare test dataset
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

batch_size = 256
margin = 0.4

VeinDataset = find_dataset_def('VeinDataset')
test_dataset = VeinDataset(root_dir='/data/seungho/datasets/Finger_Recognition/FV_802_Small/test', transform=transform)
# test_dataset = VeinDataset(root_dir='../FingerVein3.5/data_test', transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model.eval()

def cal_label_predict(labels, predicts):
    labels = np.array(labels)
    predicts = np.array(predicts)
    TP = np.sum(np.logical_and(labels == 1, predicts == 1))
    TN = np.sum(np.logical_and(labels == 0, predicts == 0))
    FP = np.sum(np.logical_and(labels == 0, predicts == 1))
    FN = np.sum(np.logical_and(labels == 1, predicts == 0))
    FAR = FP / (FP + TN) * 100
    TAR = TP / (TP + FN) * 100
    print(f'TP : {TP}, TN : {TN}, FP : {FP}, FN : {FN}')
    return FAR, TAR

with torch.no_grad():
    test_distances = []
    test_targets = []
    for i, (img1, img2, label) in enumerate(test_loader):
        img1 = img1.to(device)
        img2 = img2.to(device)
        label = label.to(device)

        output1, output2 = model(img1, img2)
        dist = torch.sqrt(torch.sum(torch.pow(output1 - output2, 2), dim=1))

        test_distances.extend(dist.detach().cpu().numpy())
        test_targets.extend(label.detach().cpu().numpy())

    test_preds = np.array(test_distances) <= margin
    FAR, TAR = cal_label_predict(test_targets, test_preds)
    print(f"Total FAR: {FAR}, Total TAR: {TAR}, margin: {margin}")



fpr, tpr, thresholds = roc_curve(list(1-np.array(test_targets)), test_distances)
test_eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
# Compute TAR at 0.01% FAR
far_index = np.where(fpr <= 0.0001)[0]
if far_index.size > 0:  # Check if there is any threshold with FAR <= 0.01%
    far_index = far_index[-1]
    test_tar_at_far = tpr[far_index]
else:
    test_tar_at_far = None  # Set to None if there is no such threshold

print(f'EER: {round(test_eer, 2)}, TAR at 0.01% FAR: {round(test_tar_at_far, 2)}')


# Plot ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_score(list(1-np.array(test_targets)), test_distances))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig(f'./roc_{margin}.png',
            format='png', dpi=200)
