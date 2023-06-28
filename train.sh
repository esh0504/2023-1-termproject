FV_Training="/data/seungho/datasets/Finger_Recognition/FV_802_Small/train"

python train.py --trainpath=$FV_Training --model='SiameseResNetv2' --dataset='VeinDatasetv2' --batch_size=256 --loss='TripletLoss' --margin=0.05 --pos_margin=0.001 --neg_margin=20.0 --exp_num=3