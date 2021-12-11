#yolov5
#https://github.com/ultralytics/yolov5
#posenet
#https://github.com/mks0601/3DMPPE_POSENET_RELEASE
#rootnet
#https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE

import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn
from tqdm import tqdm

sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
from model import get_pose_net
from model_rootnet import get_root_net
from classifier import classifier
from dataset import generate_patch_image
from utils_pose.pose_utils import process_bbox, pixel2cam
from vtrainer_utils import xyxy2xywh, select_biggest_box, joint_angle, get_joint_info_1, plot_grad_flow
from utils_pose.vis import vis_keypoints, vis_3d_multiple_skeleton, vis_3d_skeleton
from glob import glob 
from pathlib import Path
import copy
import math

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--image_dir', type=str, default='../dataset/')
    parser.add_argument('--visualization', action='store_true', default=False)
    parser.add_argument('--cls_weight', type=str, default = '')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)

    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
        gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args



# argument parsing
args = parse_args()
cfg.set_args(args.gpu_ids)
cudnn.benchmark = True

# Setting
# MuCo joint set
joint_num = 21
joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')
flip_pairs = ( (2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (17, 18), (19, 20) )
skeleton = ( (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20), (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18) )
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)]) # prepare input image


# Load pre-trained posenet
posenet_path = '../weights/posenet_weight.pth.tar'
assert osp.exists(posenet_path), 'Cannot find model at ' + posenet_path
print('Load checkpoint from {}'.format(posenet_path))
posenet = get_pose_net(cfg, False, joint_num)
posenet = DataParallel(posenet).cuda()
ckpt = torch.load(posenet_path)
posenet.load_state_dict(ckpt['network'])
posenet.eval()

# Load pre-trained rootnet 
rootnet_path = '../weights/rootnet_weight.pth.tar'
assert osp.exists(rootnet_path), 'Cannot find model at ' + rootnet_path
print('Load checkpoint from {}'.format(rootnet_path))
rootnet = get_root_net(cfg, False)
rootnet = DataParallel(rootnet).cuda()
ckpt = torch.load(rootnet_path)
rootnet.load_state_dict(ckpt['network'])
rootnet.eval()

#Load pre-trained detector: yolov5
detector = torch.hub.load('ultralytics/yolov5', 'yolov5l')
detector.cuda()
detector.classes = [0] # Only detect person 
detector.conf = 0.5 # Confidence threshold

# Load first classifier 
cls_net_1 = classifier(17)
cls_net_1.cuda()
optimizer_1 = Adam(cls_net_1.parameters(), lr=args.lr, weight_decay=1e-5)

# Load second classifier 
cls_net_2 = classifier(17)
cls_net_2.cuda()
optimizer_2 = Adam(cls_net_2.parameters(), lr=args.lr, weight_decay=1e-5)

if args.cls_weight != '':
    ckpt = torch.load(args.cls_weight)
    cls_net_1.load_state_dict(ckpt['cls_net_1'])
    cls_net_2.load_state_dict(ckpt['cls_net_2'])

# loss_functions
criterion_1 = nn.BCELoss()
criterion_2 = nn.BCELoss()


target_1 = [] #target_1: is plank or not   (1: good_images + bad_plank / 0: not_plank)
target_2 = [] #target_2: good plank or bad plank (1: good_images / 0: bad_plank + not_plank)
####### For mini batch, need to revised

# train_set
train_good_images = glob('../dataset/train/good/*.png')
train_bad_plank_images = glob('../dataset/train/bad/bad_plank/*.png')
train_not_plank_images = glob('../dataset/train/bad/bad_non_plank/*.png')




train_good_target_1 = [1. for _ in range(len(train_good_images) + len(train_bad_plank_images))]
train_bad_target_1 = [0. for _ in range(len(train_not_plank_images))]


train_good_target_2 = [1. for _ in range(len(train_good_images))]
train_bad_target_2 = [0. for _ in range(len(train_bad_plank_images) + len(train_not_plank_images))]


images = train_good_images + train_bad_plank_images + train_not_plank_images
targets_1 = train_good_target_1 + train_bad_target_1
targets_2 = train_good_target_2 + train_bad_target_2


weight_output_dir = '../weights/classfier/'
os.makedirs(weight_output_dir, exist_ok = True)

for epoch in tqdm(range(args.epochs)):
    
    batch_feature = []
    batch_joint = []
    batch_target_1 = []
    batch_target_2 = []

    image_length = len(images)
    #shuffle

    idx_list = np.random.permutation(image_length)
    cnt=0
    for i in idx_list:
        # Load one image
        image = images[i]
        target_1 = torch.tensor(targets_1[i])
        target_2 = torch.tensor(targets_2[i])

        original_img = cv2.imread(image)
        img = copy.deepcopy(original_img)
        
        img = img[..., ::-1]
        ############################
        #####  Detection part  #####
        ############################
        pred = detector(img)
        #print(pred.xyxy[0].shape)
        pred = pred.xyxy[0]
        
        if pred.shape[0] != 0:
            # If there are multiple person, then select biggest one 
            label = select_biggest_box(pred)
        else:
            #If no person detection, skip 
            print('No person detected')
            continue

        original_img_height, original_img_width = img.shape[:2]
        # prepare bbox # xmin, ymin, width, height
        label = xyxy2xywh(label)
        bbox_list = label[:,:4]
        
        # normalized camera intrinsics
        focal = [1500, 1500] # x-axis, y-axis
        princpt = [original_img_width/2, original_img_height/2] # x-axis, y-axis
        
        # for each cropped and resized human image, forward it to PoseNet
        output_pose_2d_list = []
        output_pose_3d_list = []
        
        bbox = process_bbox(bbox_list[0].cpu().numpy(), original_img_width, original_img_height)
        img, img2bb_trans = generate_patch_image(original_img, bbox, False, 1.0, 0.0, False) 
        
        img = transform(img).cuda()[None,:,:,:]
        
        #for rootnet
        k_value = np.array([math.sqrt(cfg.bbox_real[0]*cfg.bbox_real[1]*focal[0]*focal[1]/(bbox[2]*bbox[3]))]).astype(np.float32)
        k_value = torch.FloatTensor([k_value]).cuda()[None,:]


        ################################
        ##### Pose estimation part #####
        ################################

            # forward
        with torch.no_grad():    
            #feature_map: [1,2048,8,8]
            pose_3d, feature_map = posenet(img) # x,y: pixel, z: root-relative depth (mm)
            root_3d = rootnet(img, k_value)

        root_depth = root_3d[0,2].cpu().numpy()

        # inverse affine transform (restore the crop and resize)
        pose_3d = pose_3d[0].cpu().numpy()
        #256,256이미지에서의 위치로 옮겨주고 
        pose_3d[:,0] = pose_3d[:,0] / cfg.output_shape[1] * cfg.input_shape[1]
        pose_3d[:,1] = pose_3d[:,1] / cfg.output_shape[0] * cfg.input_shape[0]

        pose_3d_xy1 = np.concatenate((pose_3d[:,:2], np.ones_like(pose_3d[:,:1])),1)
        img2bb_trans_001 = np.concatenate((img2bb_trans, np.array([0,0,1]).reshape(1,3)))
        pose_3d[:,:2] = np.dot(np.linalg.inv(img2bb_trans_001), pose_3d_xy1.transpose(1,0)).transpose(1,0)[:,:2]

        joint_coor_2d = pose_3d[:,:2].copy()
        
        #root-relative discretized depth -> absolute continuous depth
        pose_3d[:,2] = (pose_3d[:,2] / cfg.depth_dim * 2 - 1) * (cfg.bbox_3d_shape[0]/2) + root_depth
        pose_3d = pixel2cam(pose_3d, focal, princpt)
        joint_coor_3d = pose_3d.copy()

        if args.visualization:
            # visualize 2d poses
            vis_img = original_img.copy()
        
            vis_kps = np.zeros((3,joint_num))
            vis_kps[0,:] = joint_coor_2d[:,0]
            vis_kps[1,:] = joint_coor_2d[:,1]
            vis_kps[2,:] = 1
            vis_img = vis_keypoints(vis_img, vis_kps, skeleton, joints_name)
        
            cv2.imwrite(args.image_dir + f'/results/output_{cnt}_2d.jpg', vis_img)
            
            # visualize 3d poses
            #vis_kps = np.array(output_pose_3d_list)
            vis_3d_skeleton(joint_coor_3d, np.ones_like(joint_coor_3d), skeleton, f'output_{cnt}_3d (x,y,z: camera-centered. mm.).jpg', args.image_dir+f'/3d_results/')
            #vis_3d_multiple_skeleton(vis_kps, np.ones_like(vis_kps), skeleton, f'output_{cnt}_3d (x,y,z: camera-centered. mm.).jpg', args.image_dir+f'/3d_results/')
            

        # accumulate outputs for batch training
        batch_feature.append(feature_map.clone().detach().cpu())
        batch_joint.append(torch.from_numpy(joint_coor_3d).unsqueeze(0))
        batch_target_1.append(target_1)
        batch_target_2.append(target_2)
        
        
        
        if len(batch_feature) == args.batch_size or i - 1 == image_length:
            
            batch_feature = torch.cat(batch_feature, dim=0).cuda()
            batch_joint = torch.cat(batch_joint, dim=0).cuda()
            batch_target_1 = torch.tensor(batch_target_1).cuda()
            batch_target_2 = torch.tensor(batch_target_2).cuda()
            # import pdb
            # pdb.set_trace()
            ###################################
            ####### classification part #######
            ###################################
            # Train
            cls_net_1.zero_grad()

            batch_joint_info = get_joint_info_1(batch_joint) # make feature from raw 3d joint coord
            pred = cls_net_1(batch_feature, batch_joint_info.cuda())

            loss_1 = criterion_1(pred.view(-1), batch_target_1)
            loss_1.backward()

            # plot_grad_flow(cls_net_1.named_parameters(),'cls_net_1')
            optimizer_1.step()
            

            batch_feature_2 = batch_feature[(batch_target_1 == 1)]
            batch_joint_info_2 = batch_joint_info[(batch_target_1 == 1)] 
            batch_target_2 = batch_target_1[(batch_target_1 == 1)]

            if (batch_target_1 == 1).sum() != 0:
                #############################
                ### second classification ###
                #############################
                cls_net_2.zero_grad()

                pred = cls_net_2(batch_feature_2, batch_joint_info_2.cuda())
                loss_2 = criterion_2(pred.view(-1), batch_target_2)
                loss_2.backward()
                # plot_grad_flow(cls_net_2.named_parameters(), 'cls_net_2')
                optimizer_2.step()

            print(f'loss_1: {loss_1.item()} // loss_2:{loss_2.item()}')

            # initialization
            batch_feature = []
            batch_joint = []
            batch_target_1 = []
            batch_target_2 = []

        cnt+=1
    
    torch.save({
            'model_1_state_dict': cls_net_1.state_dict(),
            'model_2_state_dict': cls_net_2.state_dict()
        }, weight_output_dir + f'epochs_{epoch}.pt')
    
    





    # joints_name = (
    #     'Head_top', 
    #     'Thorax', 
    #     'R_Shoulder', 
    #     'R_Elbow', 
    #     'R_Wrist', 
    #     'L_Shoulder', 
    #     'L_Elbow', 
    #     'L_Wrist', 
    #     'R_Hip', 
    #     'R_Knee', 
    #     'R_Ankle', 
    #     'L_Hip', 
    #     'L_Knee', 
    #     'L_Ankle', 
    #     'Pelvis', root!!
    #     'Spine', 
    #     'Head', 
    #     'R_Hand', 
    #     'L_Hand', 
    #     'R_Toe', 
    #     'L_Toe'
    #     )
