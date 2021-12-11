import os 
import glob
import numpy as np
from PIL import Image
import argparse

# split list 
def split(data_list, ratio):
    data_len = len(data_list)
    max_idx = int(data_len * ratio)

    random = np.random.permutation(data_len)

    train_list = []
    test_list = []

    for i in range(data_len):
        sampled = random[i]
        if i < max_idx:
            train_list.append(data_list[sampled])
        else:
            test_list.append(data_list[sampled])

    return train_list, test_list 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--number', type=int, help='cross validation set')
    args = parser.parse_args()

    number = args.number


    data_dir = '../dataset/'
    train_dir = f'../dataset/train_{number}/'
    test_dir = f'../dataset/test_{number}/'
    os.makedirs(train_dir, exist_ok = True)
    os.makedirs(test_dir, exist_ok = True)

    ratio = 0.7
    ##################################################
    # good data 
    output_dir = '../dataset/'
    good_list = glob.glob('../dataset/good/*.png')
    good_list += glob.glob('../dataset/good/*.jpg')

    os.makedirs(f'../dataset/train_{number}/good/',exist_ok = True)
    os.makedirs(f'../dataset/test_{number}/good/', exist_ok = True)

    train_list, test_list = split(good_list, ratio)
    for i in range(len(train_list)):
        img = Image.open(train_list[i]).convert('RGB')
        img.save(f'../dataset/train_{number}/good/{i}.png')
    for i in range(len(test_list)):
        img = Image.open(test_list[i]).convert('RGB')
        img.save(f'../dataset/test_{number}/good/{i}.png')

    ##################################################
    #bad plank
    bad_plank = glob.glob('../dataset/bad/bad_plank/*.png')
    bad_plank += glob.glob('../dataset/bad/bad_plank/*.jpg')

    train_list, test_list = split(bad_plank, ratio)
    os.makedirs(f'../dataset/train_{number}/bad/bad_plank/', exist_ok=True)
    os.makedirs(f'../dataset/test_{number}/bad/bad_plank/', exist_ok=True)
    for i in range(len(train_list)):
        img = Image.open(train_list[i]).convert('RGB')
        img.save(f'../dataset/train_{number}/bad/bad_plank/{i}.png')
    for i in range(len(test_list)):
        img = Image.open(test_list[i]).convert('RGB')
        img.save(f'../dataset/test_{number}/bad/bad_plank/{i}.png')
    ##################################################
    #bad non plank
    bad_not_plank = glob.glob('../dataset/bad/bad_not_plank/*.png')
    bad_not_plank += glob.glob('../dataset/bad/bad_not_plank/*.jpg')
    bad_not_plank += glob.glob('../dataset/additional/*.jpg')
    os.makedirs(f'../dataset/train_{number}/bad/bad_not_plank/', exist_ok=True)
    os.makedirs(f'../dataset/test_{number}/bad/bad_not_plank/', exist_ok=True)

    train_list, test_list = split(bad_not_plank, ratio)
    for i in range(len(train_list)):
        img = Image.open(train_list[i]).convert('RGB')
        img.save(f'../dataset/train_{number}/bad/bad_not_plank/{i}.png')
    for i in range(len(test_list)):
        img = Image.open(test_list[i]).convert('RGB')
        img.save(f'../dataset/test_{number}/bad/bad_not_plank/{i}.png')
        




        


