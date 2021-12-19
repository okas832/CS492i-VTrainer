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
from classifier import classifier, classifier_only_joint
from dataset import generate_patch_image
from utils_pose.pose_utils import process_bbox, pixel2cam
from vtrainer_utils import xyxy2xywh, select_biggest_box, joint_angle, get_joint_info_1, plot_grad_flow, fontscale, load_data
from utils_pose.vis import vis_keypoints, vis_3d_multiple_skeleton, vis_3d_skeleton
from glob import glob 
from pathlib import Path
import copy
import math

def evaluate(args, posenet, rootnet, detector, cls_net_1, cls_net_2, images, targets_1, targets_2):
    # Setting
    # MuCo joint set
    joint_num = 21
    joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')
    flip_pairs = ( (2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (17, 18), (19, 20) )
    skeleton = ( (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20), (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18) )
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)]) # prepare input image
        
    batch_feature = []
    batch_joint = []
    batch_target_1 = []
    batch_target_2 = []

    image_length = len(images)
    #shuffle
    idx_list = np.random.permutation(image_length)

    total_correct_1 = 0
    total_correct_2 = 0
    detected_plank_count = 0
    not_plank_correct_count = 0
    gt_plank_count = sum(targets_1)
    
    for i, image_idx in enumerate(idx_list):
        # Load one image
        image = images[image_idx]
        target_1 = torch.tensor(targets_1[image_idx])
        target_2 = torch.tensor(targets_2[image_idx])

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

        # accumulate outputs for batch training
        batch_feature.append(feature_map.clone().detach().cpu())
        batch_joint.append(torch.from_numpy(joint_coor_3d).unsqueeze(0))
        batch_target_1.append(target_1)
        batch_target_2.append(target_2)
        
        if len(batch_feature) == args.batch_size or i + 1 == image_length:
            
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

            batch_joint_info = get_joint_info_1(batch_joint) # make feature from raw 3d joint coord
            pred = cls_net_1(batch_feature, batch_joint_info.cuda())

            #loss_1 = criterion_1(pred.view(-1), batch_target_1)
            correct_1 = (pred.view(-1).round() == batch_target_1).sum()
            total_correct_1 += correct_1
            
            
            not_plank_correct_count += ((pred.view(-1).round() == batch_target_1) * (batch_target_1 == 0)).sum()
            # plot_grad_flow(cls_net_1.named_parameters(),'cls_net_1')
        
            batch_feature_2 = batch_feature[(pred.view(-1).round() == 1) * (pred.view(-1).round() ==batch_target_1)]
            batch_joint_info_2 = batch_joint_info[(pred.view(-1).round() == 1) * (pred.view(-1).round() == batch_target_1)] 
            batch_target_2 = batch_target_2[(pred.view(-1).round() == 1) * (pred.view(-1).round() ==  batch_target_1)]


            if (pred.view(-1).round() == 1).sum() != 0:
                detected_plank_count += (pred.view(-1).round() == 1).sum()
                #############################
                ### second classification ###
                #############################

                pred = cls_net_2(batch_feature_2, batch_joint_info_2.cuda())
                #loss_2 = criterion_2(pred.view(-1), batch_target_2)
                correct_2 = (pred.view(-1).round() == batch_target_2).sum()
                total_correct_2 += correct_2
                

            # print(f'loss_1: {loss_1.item()} // loss_2:{loss_2.item()}')

            # initialization
            batch_feature = []
            batch_joint = []
            batch_target_1 = []
            batch_target_2 = []

    print(f"Total Test image: {len(images)}")
    print(f'Total gt plank: {int(gt_plank_count)}')
    print(f'Total detected plank: {detected_plank_count}' )

    print(f'total_correct_2: {total_correct_2}')
    print(f'not_plank_correct_count: {not_plank_correct_count}')

    acc_1 = total_correct_1 / len(images) #(전체 이미지 중에서 plank 인지 아닌지)
    acc_2 = (not_plank_correct_count + total_correct_2) / len(images) # 전체이미지에서 좋은 plank인지 아닌지 
    # acc_2 = total_correct_2 / detected_plank_count #(plank로 detection 것 중에서 좋은plank 인지 아닌지)
    acc_3 = total_correct_2 / gt_plank_count # 실제 plank인 이미지 중에서 좋은 plank인지 아닌지
    return acc_1*100, acc_2*100, acc_3*100

    
def train(args, epoch, posenet, rootnet, detector, cls_net_1, cls_net_2, optimizer_1, optimizer_2, criterion_1, criterion_2, images, targets_1, targets_2,  weight_output_dir):
    # Setting
    # MuCo joint set
    joint_num = 21
    joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')
    flip_pairs = ( (2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (17, 18), (19, 20) )
    skeleton = ( (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20), (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18) )
    transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(p=0.5), transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)]) # prepare input image
        
    batch_feature = []
    batch_joint = []
    batch_target_1 = []
    batch_target_2 = []

    image_length = len(images)
    #shuffle
    idx_list = np.random.permutation(image_length)

    for i, image_idx in enumerate(idx_list):
        # Load one image
        image = images[image_idx]
        target_1 = torch.tensor(targets_1[image_idx])
        target_2 = torch.tensor(targets_2[image_idx])

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

        # accumulate outputs for batch training
        batch_feature.append(feature_map.clone().detach().cpu())
        batch_joint.append(torch.from_numpy(joint_coor_3d).unsqueeze(0))
        batch_target_1.append(target_1)
        batch_target_2.append(target_2)
        
        
        
        if len(batch_feature) == args.batch_size or i + 1 == image_length:
            
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
            batch_target_2 = batch_target_2[(batch_target_1 == 1)]

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

     
    
    torch.save({
            'model_1_state_dict': cls_net_1.state_dict(),
            'model_2_state_dict': cls_net_2.state_dict()
        }, weight_output_dir + f'epochs_{epoch}.pt')
    
if __name__ ==  '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--extra_tag', type=str, default='default/')
    parser.add_argument('--cls_weight', type=str, default = '')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--dataset_num', type=int, default=1)
    parser.add_argument('--posenet', type=str, default='../weights/posenet_weight.pth.tar')
    parser.add_argument('--rootnet', type=str, default='../weights/rootnet_weight.pth.tar')
    parser.add_argument('--eval', action='store_true')

    args = parser.parse_args()

    joint_num = 21
    joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')
    flip_pairs = ( (2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (17, 18), (19, 20) )
    skeleton = ( (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20), (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18) )
    transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(p=0.5), transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)]) # prepare input image

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
        gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    cfg.set_args(args.gpu_ids)
    cudnn.benchmark = True

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
    cls_net_1 = classifier_only_joint(17)
    cls_net_1.cuda()
    optimizer_1 = Adam(cls_net_1.parameters(), lr=args.lr, weight_decay=1e-5)

    # Load second classifier 
    cls_net_2 = classifier_only_joint(17)
    cls_net_2.cuda()
    optimizer_2 = Adam(cls_net_2.parameters(), lr=args.lr, weight_decay=1e-5)

    if args.cls_weight != '':
        ckpt = torch.load(args.cls_weight)
        print(args.cls_weight)
        cls_net_1.load_state_dict(ckpt['model_1_state_dict'])
        cls_net_2.load_state_dict(ckpt['model_2_state_dict'])
        epoch = ckpt['epoch']

    # loss_functions
    criterion_1 = nn.BCELoss()
    criterion_2 = nn.BCELoss()

    weight_output_dir = '../weights/classfier/' + args.extra_tag
    os.makedirs(weight_output_dir, exist_ok = True)

    train_images, train_targets_1, train_targets_2 = load_data(train=True, dataset_num=args.dataset_num)
    test_images, test_targets_1, test_targets_2 = load_data(train=False, dataset_num=args.dataset_num)
    
    # start training 
    
    best_acc_2 = 0
    best_epoch = 0
    if args.eval:
        with torch.no_grad():            
            cls_net_1.eval()
            cls_net_2.eval()
            acc_1, acc_2, acc_3 = evaluate(args, posenet, rootnet, detector, cls_net_1, cls_net_2, test_images, test_targets_1, test_targets_2)
            print(f"############# best model : {epoch} #############")
            print(f'acc_1(plank or not): {acc_1}%')
            print(f'acc_2(good plank or not(among the total images): {acc_2}%')
            print(f'acc_3(good plank or not(among the gt plank): {acc_3}%')


    else:
        for epoch in tqdm(range(args.epochs)):
            cls_net_1.train()
            cls_net_2.train()
            train(args, epoch, posenet, rootnet, detector, cls_net_1, cls_net_2, optimizer_1, optimizer_2, criterion_1, criterion_2, train_images, train_targets_1, train_targets_2, weight_output_dir)
            with torch.no_grad():
                
                cls_net_1.eval()
                cls_net_2.eval()
                    #acc_1, acc_2, acc_3 = evaluate_visualization(args, posenet, rootnet, detector, cls_net_1, cls_net_2, test_images, test_targets_1, test_targets_2, args.extra_tag)
                acc_1, acc_2, acc_3 = evaluate(args, posenet, rootnet, detector, cls_net_1, cls_net_2, test_images, test_targets_1, test_targets_2)
                print(f"############# {epoch}/{args.epochs} #############")
                print(f'acc_1(plank or not): {acc_1}%')
                print(f'acc_2(good plank or not(among the total images): {acc_2}%')
                print(f'acc_3(good plank or not(among the gt plank): {acc_3}%')

                if best_acc_2 < acc_2:
                    best_acc_2 = acc_2
                    best_epoch = epoch 
                    torch.save({
                        'model_1_state_dict': cls_net_1.state_dict(),
                        'model_2_state_dict': cls_net_2.state_dict(),
                        'epoch' : epoch
                    }, weight_output_dir + f'best_model.pt')

        with torch.no_grad():            
            cls_net_1.eval()
            cls_net_2.eval()
            ckpt = torch.load(weight_output_dir + f'best_model.pt')
            cls_net_1.load_state_dict(ckpt['model_1_state_dict'])
            cls_net_2.load_state_dict(ckpt['model_2_state_dict'])
            epoch = ckpt['epoch']

            acc_1, acc_2, acc_3 = evaluate(args, posenet, rootnet, detector, cls_net_1, cls_net_2, test_images, test_targets_1, test_targets_2)
            print(f"############# best model : {epoch} #############")
            print(f'acc_1(plank or not): {acc_1}%')
            print(f'acc_2(good plank or not(among the total images): {acc_2}%')
            print(f'acc_3(good plank or not(among the gt plank): {acc_3}%')
