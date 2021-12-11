#load only image from img_dir
import torch
from PIL import Image
import numpy as np
import skimage.transform
from torch.utils.data import Dataset
from glob import glob 

class image_dataset(Dataset):
    def __init__(self, img_dir, transform=None):
        
        self.img_list = []
        for ext in ('*.png', '*.jpg'):
            self.img_list.extend(glob(img_dir + ext))
        self.transform = transform
    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = Image.open(self.img_list[idx]).convert('RGB')

        
        if self.transform:
            sample = self.transform(sample)

        return sample


# return image, target, original image size
class image_class_dataset(Dataset):
    def __init__(self, annot_txt, transform=None):
        
        f = open(annot_txt, 'r')
        lines = f.readlines()
        self.img_list = []
        # prank / non_prank
        self.targets_1 = []
        # good_prank / bad_prank
        self.targets_2 = []
        for line in lines:
            image, target_1, target_2 = line.split()
            self.img_list.extend(image)
            self.targets_1.extend(int(target_1))
            self.targets_2.extend(int(target_2))

        self.transform = transform
    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = Image.open(self.img_list[idx]).convert('RGB')
        
        target_1 = np.array(int(self.targets_1[idx]))
        target_2 = np.array(int(self.targets_2[idx]))

        sample = {
            'image': image,
            'target_1': target_1,
            'target_2': target_2,
            'h_original': h_original,
            'w_original': w_original,
        }
        if self.transform:
            sample = self.transform(sample)

        return sample

# class vtainer_dataset(Dataset):
#     def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
#     self.img_labels = pd.read_csv(annotations_file)
#     self.img_dir = img_dir
#     self.transform = transform
#     self.target_transform = target_transform

#     def __len__(self):
#         return len(self.img_labels)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#         image = Image.open(self.img_list[idx]).convert('RGB')

#         label = self.img_labels.iloc[idx, 1]
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label

class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample, min_side=480, max_side=640):
        image = sample['image']
        height, width, channels = image.shape
        smallest_side = min(height, width)
        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side
        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(height, width)
        if largest_side * scale > max_side:
            scale = max_side / largest_side
        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(height*scale)), int(round((width*scale)))))
        height, width, cns = image.shape
        pad_h = 32 - height%32
        pad_w = 32 - width%32
        new_image = np.zeros((height + pad_h, width + pad_w, cns)).astype(np.float32)
        new_image[:height, :width, :] = image.astype(np.float32)
        
        # new_image = np.zeros((height + pad_h, width + pad_w, channels)).astype(np.float32)
        # new_image[:height, :width, :] = image.astype(np.float32)

        sample['image'] = torch.from_numpy(new_image)
        sample['scale'] = scale
        return sample
class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])
    def __call__(self, sample):
        image = sample['image']
        image = image/255.0
        return {'image':((image.astype(np.float32)-self.mean)/self.std)}

class RandomFlip(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample, flip_x=0.5):

        if np.random.rand() < flip_x:
            image = sample['image']
            image = image[:, ::-1, :]
            rows, cols, channels = image.shape           
            sample = {'image': image}

        return sample


def collater(batch):
    imgs = [x['image'] for x in batch]
    #annots = [x['targets'] for x in batch]
    scales = [x['scale'] for x in batch]
        
    heights = [int(s.shape[0]) for s in imgs]
    widths = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_height = np.array(heights).max()
    max_width = np.array(widths).max()
    
    padded_imgs = torch.zeros(batch_size, max_height, max_width, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    padded_imgs = padded_imgs.permute(0, 3, 1, 2) # batch_size, channels, height, width 

    #return {'image': padded_imgs, 'targets': torch.tensor(annots), 'scale': scales}
    return {'image': padded_imgs,'scale': scales}