import numpy as np
import torch
import copy
import math
import matplotlib.pyplot as plt
from glob import glob 

# convert box coordinate format 
# top-left point, bottom right point -> center x, center y, width, height
def xyxy2xywh(labels):
    #minx, miny, maxx, maxy
    width = labels[:,2] - labels[:,0] 
    height = labels[:,3] - labels[:,1]

    labels[:,2] = width
    labels[:,3] = height

    return labels

# Select biggest box 
def select_biggest_box(boxes):
    max_idx = 0
    max_area = 0
    if boxes.shape[0] == 1:
        return boxes
    for i in range(boxes.shape[0]):
        width = boxes[i,2] - boxes[i,0] 
        height = boxes[i,3] - boxes[i,1]
        area = width * height 

        if max_area < area:
            max_area = area
            max_idx = i
    return boxes[max_idx:max_idx+1,:]

# Th function that compute joint angle 
def joint_angle(joints, end_point1, end_point2, inter_point):
    v1 = joints[end_point1, :] - joints[inter_point,:]
    v2 = joints[end_point2,:] - joints[inter_point,:]

    angle = torch.arccos(torch.dot(v1,v2) / (torch.linalg.norm(v1)*torch.linalg.norm(v2))) # radian 

    return angle
# Choose visible body part. If the depth of R_shoulder smaller than L_shoulder, we define right parts as visible parts.
def left_right(joints):
    # joints[2][2]: R_shoulder의 z
    # joints[5][2]: L_shoulder의 z

    # True 왼쪽이 카메라쪽, False: 오른쪽이 카메라쪽 
    return joints[2,2] > joints[5,2]

# We compute specific joint angle for classifier.
def get_joint_info_1(joints): # (32, 21, 3)
    joint_info = torch.zeros((joints.shape[0],17)).cuda()

    for i in range(joints.shape[0]):
        j1 = joint_angle(joints[i],0,16,1)  # head_top, head, thorax
        j2 = joint_angle(joints[i],0,16,15) # head_top, head, spine
        j3 = joint_angle(joints[i],0,16,14) # head_top, head, pelvis

        j4 = joint_angle(joints[i],0,1,15) # head_top, thorax, spine
        j5 = joint_angle(joints[i],0,1,14) # head_top, thorax, pelvis

        j6 = joint_angle(joints[i],0,15,14) # head_top, spine, pelvis

        j7 = joint_angle(joints[i],16, 1, 15) # head, thorax, spine
        j8 = joint_angle(joints[i],16, 15, 14) # head, spine, pelvis

        j9 = joint_angle(joints[i],1,15,14) # thorax, spine, pelvis
        

        #choose left
        if left_right(joints[i]):
            j10 = joint_angle(joints[i],1,5,6) # thorax, L_shoulder, L_eldow
            j11 = joint_angle(joints[i],5,6,7) # L_shoulder, L_elbow, l_wrist
            j12 = joint_angle(joints[i],11,12,13) # l_hip,L_knee, L_ankle
            j13 = joint_angle(joints[i],12,13,20) # L_knee, L_ankle, L_Toe
            j14 = joint_angle(joints[i],13,14,15) # L_Ankle, pelvis, spine
            j15 = joint_angle(joints[i],13,15,1) # L_Ankle, spine, thorax
            j16 = joint_angle(joints[i],13,14,1) # L_Ankle, pelvis, thorax
            j17 = joint_angle(joints[i],15, 11, 13) # spine, L_hip, L_knee


        #choose right
        else:
            j10 = joint_angle(joints[i],1,2,3) #thorax, R_shoulder, R_elbow
            j11 = joint_angle(joints[i],2,3,4) #R_shoulder, R_elbow, R_wrist
            j12 = joint_angle(joints[i],8,9,10) # R_hip, R_knee, R_ankle
            j13 = joint_angle(joints[i],9,10,19) # R_knee, R_ankle, R_toe

            j14 = joint_angle(joints[i],10,14,15) # R_Ankle, pelvis, spine
            j15 = joint_angle(joints[i],10,15,1) # R_Ankle, spine, thorax
            j16 = joint_angle(joints[i],10,14,1) # R_Ankle, pelvis, thorax
            j17 = joint_angle(joints[i],15,8,9) # spine, L_hip, L_knee

        joint_info[i,:] = torch.tensor([j1,j2,j3,j4,j5,j6,j7,j8,j9,j10,j11,j12,j13,j14,j15,j16,j17])

    return joint_info


  # joints_name = (
    # 0     'Head_top', 
    # 1     'Thorax', 
    # 2    'R_Shoulder', 
    # 3    'R_Elbow', 
    # 4    'R_Wrist', 
    # 5    'L_Shoulder', 
    # 6    'L_Elbow', 
    # 7    'L_Wrist', 
    # 8    'R_Hip', 
    # 9    'R_Knee', 
    # 10    'R_Ankle', 
    # 11    'L_Hip', 
    # 12    'L_Knee', 
    # 13    'L_Ankle', 
    # 14    'Pelvis', root!!
    # 15    'Spine', 
    # 16    'Head', 
    # 17    'R_Hand', 
    # 18    'L_Hand', 
    # 19    'R_Toe', 
    # 20    'L_Toe'
    #     )

# for check grad_flow.
# This is just for check. 
# We get from stackoverflow
def plot_grad_flow(named_parameters, model_name):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig(f'./{model_name}_grad_flow.png')
    plt.close()
    

def plot_grad_flow_v2(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])


def fontscale(width):
    #1500일때 1
    return width/1500

# Load split dataset by its number 
def load_data(train, dataset_num):
    if train:
        mode = f'train_{dataset_num}'
    else:
        mode = f'test_{dataset_num}'

    target_1 = [] #target_1: is plank or not   (1: good_images + bad_plank / 0: not_plank)
    target_2 = [] #target_2: good plank or bad plank (1: good_images / 0: bad_plank + not_plank)
    ####### For mini batch, need to revised

    # dataset_set
    good_images = glob(f'../dataset/{mode}/good/*.png')
    bad_plank_images = glob(f'../dataset/{mode}/bad/bad_plank/*.png')
    not_plank_images = glob(f'../dataset/{mode}/bad/bad_not_plank/*.png')

    good_target_1 = [1. for _ in range(len(good_images) + len(bad_plank_images))]
    bad_target_1 = [0. for _ in range(len(not_plank_images))]

    good_target_2 = [1. for _ in range(len(good_images))]
    bad_target_2 = [0. for _ in range(len(bad_plank_images) + len(not_plank_images))]

    images = good_images + bad_plank_images + not_plank_images
    targets_1 = good_target_1 + bad_target_1
    targets_2 = good_target_2 + bad_target_2

    return images, targets_1, targets_2