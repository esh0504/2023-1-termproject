import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
from model import find_model_def
from datasets import find_dataset_def
from loss import find_loss_def
from torchvision import transforms
import os, argparse
import random
import numpy as np
from datetime import datetime
from utils import *


parser = argparse.ArgumentParser(description='CVIP Lab Finger Vein Recognition as FVR')
parser.add_argument('--mode', default='train', help='train or test', choices=['train', 'test'])
parser.add_argument('--model', default='SiameseResNetv2', help='select model')
parser.add_argument('--device', default='cuda', help='select model')

parser.add_argument('--dataset', default='VeinDatasetv2', help='select dataset')
parser.add_argument('--trainpath', help='train datapath')
parser.add_argument('--testpath', help='test datapath')

parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='train batch size')

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')

parser.add_argument('--seed', type=int, default=777, metavar='S', help='random seed')

parser.add_argument('--margin', type=float, default=0.05, help='Classification Margin')
parser.add_argument('--loss', type=str, default="SepaTripletLoss")
parser.add_argument('--pos_margin', type=float, default=0.001)
parser.add_argument('--neg_margin', type=float, default=10.0)

parser.add_argument('--save_interval', type=int, default=5, metavar='S', help='save train epoch interval')
parser.add_argument('--exp_num', type=str, default='0', help='save checkpoints by exp_num')

args = parser.parse_args()

print_args(args)

# 시드 값을 설정합니다.
seed_value = 777

random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
np.random.seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

save_dir = os.path.join('./checkpoints', args.exp_num)

if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

VeinDataset = find_dataset_def(args.dataset)
train_dataset = VeinDataset(root_dir=args.trainpath, transform=transform)
# train_dataset = VeinDataset(root_dir='/data/seungho/datasets/Finger_Recognition/FV_802_Small/train', transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

SiameseResNet = find_model_def(args.model)
model = SiameseResNet().to(device)
model = torch.nn.DataParallel(model)

Loss = find_loss_def(args.loss)

if args.loss == 'TripletLoss':
    model_loss = Loss(torch.tensor(args.margin))
elif args.loss == 'SepaTripletLoss':
    model_loss = Loss(torch.tensor(args.pos_margin), torch.tensor(args.neg_margin))

optimizer = optim.Adam(model.parameters(), lr=args.lr)

best_loss = 10.0

for epoch in range(args.epochs):
    loss_lst = []
    pos_loss_lst = []
    neg_loss_lst = []
    for i, (anchor, neg, pos) in enumerate(train_loader):
        anchor = anchor.to(device)
        neg_img = neg.to(device)
        pos_img = pos.to(device)
        

        anchor_output, neg_output, pos_output = model(anchor, neg_img, pos_img)
        if args.loss == 'SepaTripletLoss':
            loss, pos_loss, neg_loss = model_loss(anchor_output, pos_output, neg_output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_lst.append(loss)
            pos_loss_lst.append(pos_loss)
            neg_loss_lst.append(neg_loss)
            print('Epoch {}/{}, Iter {}/{}, train loss = {:.6f}, Positive_Loss = {:.6f}, Negative_Loss = {:.6f}'.format(epoch, args.epochs, 
                                                                                                                               i, len(train_loader), loss, pos_loss, neg_loss))
        
        
        elif args.loss == 'TripletLoss':
            loss = model_loss(anchor_output, pos_output, neg_output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_lst.append(loss)
            print('Epoch {}/{}, Iter {}/{}, train loss = {:.6f}'.format(epoch, args.epochs, 
                                                                            i, len(train_loader), loss))
    
    loss_avg = sum(loss_lst)/len(loss_lst)
    best_loss = min(loss_avg, best_loss)
    
    if best_loss == loss_avg:
        torch.save(model.state_dict(), f'{save_dir}/best_model.pth')
        
    if args.loss == 'SepaTripletLoss':
        pos_loss_avg = sum(pos_loss_lst)/len(pos_loss_lst)
        neg_loss_avg = sum(neg_loss_lst)/len(neg_loss_lst)
        print(f'Epoch [{epoch+1}/{args.epochs}], Loss: {loss_avg}, Positive_loss : {pos_loss_avg}, Negative_loss : {neg_loss_avg}')
    elif args.loss == 'TripletLoss':
        print(f'Epoch [{epoch+1}/{args.epochs}], Loss: {loss_avg}')
    
    if epoch % args.save_interval == 0 :
        torch.save(model.state_dict(), f'{save_dir}/{epoch}.pth')

