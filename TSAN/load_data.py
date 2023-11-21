import os
import torch
import nibabel as nib
import numpy as np
import pandas as pd

def nii_loader(path):
    img = nib.load(str(path))
    data = img.get_fdata()
    return data

def read_table(path):
    return(pd.read_excel(path).values) # default to first sheet

def white0(image, threshold=0):
    "standardize voxels with value > threshold"
    image = image.astype(np.float32)
    mask = (image > threshold).astype(int)

    image_h = image * mask
    image_l = image * (1 - mask)

    mean = np.sum(image_h) / np.sum(mask)
    std = np.sqrt(np.sum(np.abs(image_h - mean)**2 * mask) / np.sum(mask))

    if std > 0:
        ret = (image_h - mean) / std + image_l
    else:
        ret = image * 0.
    return ret

class Integer_Multiple_Batch_Size(torch.utils.data.Dataset):
    
    def __init__(self, folder_dataset, batch_size=8):
        self.folder_dataset = folder_dataset
        self.batch_size = batch_size

        source_dataset_len = len(self.folder_dataset)
        num_need_to_complement = self.batch_size - (source_dataset_len % self.batch_size)
        
        idx_list = np.arange(0, source_dataset_len)
        complement_idx = idx_list[-num_need_to_complement:]
        self.complemented_idx = np.concatenate([idx_list, complement_idx], axis=0)
        self.complemented_size = self.complemented_idx.shape[0]
        print(self.complemented_idx.shape, self.complemented_size)
        
    def __len__(self):
        return self.complemented_size

    def __getitem__(self, index):
        return self.folder_dataset[self.complemented_idx[index]]
    
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
