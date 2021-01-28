import torch
import numpy as np 

def discriminate_age(age, range=5):
    dis = []
    for i in age:
        value = i // range
        x = i % range
        if x < range/2:
            discri_age = value * range
        else:
            discri_age = (value+1)*range
        dis.append(discri_age)
        dis_age = np.asarray(dis,dtype='float32')
        dis_age = np.expand_dims(dis_age,axis=1)
        dis_age = torch.from_numpy(dis_age)
    return dis_age