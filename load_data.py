from torch.utils.data.dataset import Dataset
from torchvision import transforms
import os
import torch
import xlrd
import math
import nibabel as nib
import numpy as np
import pandas as pd
# from tensorlayer.prepro import apply_transform, transform_matrix_offset_center

def nii_loader(path):
    img = nib.load(str(path))
    data = img.get_data()
    # data = data - np.mean(data)
    # data = data / np.std(data)
    return data


def read_table(path):
    return(pd.read_excel(path).values) # default to first sheet

def read_excel_cols_data(path, col_num):
    workbook = xlrd.open_workbook(path)
    sheet1 = workbook.sheet_by_index(0)
    cols = sheet1.col_values(col_num)
    del cols[0]
    cols = np.asarray(cols)
    return cols

def white0(image, threshold=0):
    "standardize voxels with value > threshold"
    image = image.astype(np.float32)
    mask = (image > threshold).astype(int)

    image_h = image * mask
    image_l = image * (1 - mask)

    mean = np.sum(image_h) / np.sum(mask)
    std = np.sqrt(np.sum(np.abs(image_h - mean)**2) / np.sum(mask))

    if std > 0:
        ret = (image_h - mean) / std + image_l
    else:
        ret = image * 0.
    return ret

class move_augmentation(object):
    def __init__(self,max_pixel=10):
        self.max_pixel = max_pixel
    def __call__(self,images):
        move_direction = np.random.randint(0,7)
        a = (np.random.randint(0, self.max_pixel))
        # print('move_direction',move_direction)
        # print('move_distance',a)
        if move_direction==0:
            return images
        if move_direction==1: 
            return np.concatenate((images[:, a:,:], images[:, :a,:]), axis=1)
        if move_direction==2: 
            return np.concatenate((images[:, images.shape[1] - a:,:], images[:, :images.shape[1] - a,:]), 
            axis=1)
        if move_direction==3: 
            return np.concatenate((images[a:, :,:], images[:a, :, :]), axis=0)
        if move_direction==4: 
            return np.concatenate((images[images.shape[1] - a:, :,:], images[:images.shape[1] -a, :,:]), 
            axis=0)
        if move_direction==5: 
            return np.concatenate((images[:, :,a:], images[:, :, :a]), axis=2)
        if move_direction==6: 
            return np.concatenate((images[:, :, images.shape[1] - a:], images[:, :,:images.shape[1] -a]), 
            axis=2)
    def __repr__(self):
        return self.__class__.__name__ + '(max_pixel={0})'.format(self.max_pixel)

class flip_augmentation(object):
    def __init__(self,flip=True):
        self.flip = flip
    def __call__(self,images):
        flip_whether = np.random.randint(0,2)
        if flip_whether == 0:
            return images
        else:
            lr_flip_vol = np.fliplr(images)
            return lr_flip_vol
    def __repr__(self):
        return self.__class__.__name__


class rotation_augmentation(object):
    def __init__(self,rotation=True):
        self.rotation = rotation
    def __call__(self,images):
        rotation_wheather = np.random.randint(0,3)
        n = np.random.randint(18,181)
        if rotation_wheather == 0:
            return images
        elif rotation_wheather==1:
            rotate_theta = math.pi/n
            cos_gamma = np.cos(rotate_theta)
            sin_gamma = np.sin(rotate_theta)
            rot_affine = np.array([[cos_gamma, -sin_gamma, 0],
                                    [sin_gamma, cos_gamma, 0],
                                    [0, 0,1]])
            transform_matrix = transform_matrix_offset_center(rot_affine, images.shape[0], 
                                                                            images.shape[1])
            rotation_image = apply_transform(images, transform_matrix)
            return rotation_image
        else:
            rotate_theta = -math.pi/n
            cos_gamma = np.cos(rotate_theta)
            sin_gamma = np.sin(rotate_theta)
            rot_affine = np.array([[cos_gamma, -sin_gamma, 0],
                                    [sin_gamma, cos_gamma, 0],
                                    [0, 0,1]])
            transform_matrix = transform_matrix_offset_center(rot_affine, images.shape[0], 
                                                                            images.shape[1])
            rotation_image = apply_transform(images, transform_matrix)
            return rotation_image
        
    def __repr__(self):
        return self.__class__.__name__

class IMG_Folder(torch.utils.data.Dataset):
    def __init__(self,excel_path, data_path, loader=nii_loader,transforms=None):
        self.root = data_path
        self.sub_fns = sorted(os.listdir(self.root))
        self.table_refer = read_table(excel_path)
        self.loader = loader
        self.transform = transforms

    def __len__(self):
        return len(self.sub_fns)

    def __getitem__(self,index):
        sub_fn = self.sub_fns[index]
        for f in self.table_refer:
            
            sid = str(f[0])
            slabel = (int(f[1]))
            smale = f[2]
            if sid not in sub_fn:
                continue
            sub_path = os.path.join(self.root, sub_fn)
            img = self.loader(sub_path)
            img = white0(img)
            if self.transform is not None:
                img = self.transform(img)
            img = np.expand_dims(img, axis=0)
            img = np.ascontiguousarray(img, dtype= np.float32)
            img = torch.from_numpy(img).type(torch.FloatTensor)
            break
        return (img, sid, slabel, smale)


class my_DataSet(Dataset):
    def __init__(self, excel_path, data_path, loader=nii_loader):
        self.labels = read_excel_cols_data(excel_path, 1)
        self.id = read_excel_cols_data(excel_path, 0)
        img = os.listdir(data_path)
        self.img = [os.path.join(data_path, k) for k in img]

    def __getitem__(self, item):
        img_path = self.img[item]
        for i in range(self.id.shape[0]):
            if self.id[i] in img_path:
                label = self.labels[i]
                img_data = nib.load(img_path).get_fdata()
                img_data = img_data - np.mean(img_data)
                img_data = img_data / np.std(img_data)
                img_data = np.expand_dims(img_data, axis=0)
                data = torch.from_numpy(img_data)
                
                return data, label

    def __len__(self):
        return len(self.img)


if __name__ == '__main__':
    # np.random.seed(42)
    path = "/data/ziyang/age/lables/raw_age.xls"
    a = read_table(path)
    print(a)
    # transforms = transforms.Compose([flip_augmentation(),
    #                                  move_augmentation(),
    #                                  rotation_augmentation()]) 
    loader = IMG_Folder(
             data_path="/data/ziyang/age/data/stroke/nonlinear_brain"
            ,excel_path=path
            ,loader=nii_loader
            ,transforms=None)

    train_loader = torch.utils.data.DataLoader(loader,batch_size=20,num_workers=10,shuffle=True)
    for i, (img,id,label,male) in enumerate(train_loader):
        print(id)
        print('age',label)
        print('gender',male)
