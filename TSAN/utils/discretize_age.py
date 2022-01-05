import torch
import numpy as np 

def discretize_age(age, range=5):
    '''
    [summary]

    Args:
        age (numpy array): [predicted brain age from first stage network, which is needed to discriminate.]
        range (int, optional): [discritized delta]. Defaults to 5.

    Returns:
        [numpy array]: [discritized predicted brain age]
    '''
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