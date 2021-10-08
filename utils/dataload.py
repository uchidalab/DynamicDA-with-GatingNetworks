import os, csv, torch
import glob
import numpy as np
from torchvision import datasets, transforms
from utils.augmentation import *

da_dict = {'noise':noise,'identity':identity,'jitter':jitter,'scaling':scaling,'rotation':rotation,'permutation':permutation,
          'magnitudeWarp':magnitudeWarp,'timeWarp':timeWarp,'windowSlice':windowSlice,'windowWarp':windowWarp,'toy1':toy1,'toy2':toy2}

class UCR(torch.utils.data.Dataset):
    def __init__(self, datapath, da1, da2, da3, da4, da5):
        self.datapath = datapath
        self.da1, self.da2, self.da3, self.da4, self.da5 = da1, da2, da3, da4, da5
        cnt = 0
        with open(self.datapath) as f:
            col_floats = []
            for cols in csv.reader(f, delimiter='\t'):
                col_float = [float(cols[i]) for i in range(len(cols))]
                col_floats.append(col_float)
                cnt+=1
        self.col_floats = col_floats
        self.datalen = cnt
    
    def __len__(self):
        return self.datalen
    
    def __getitem__(self, idx):
        datas = self.col_floats[idx]
        label = int(datas[0]-1) # must be 0 to num_of_class-1. So, in tsv, class must be in range 1 to num_of_class.
        data = [float(datas[i+1]/1.0) for i in range(len(datas[1:]))]
        
        # Data Augmentation
        if self.da1!='toy1' and self.da2!='toy1':
            da1_data = da_dict[self.da1](data)
            da2_data = da_dict[self.da2](data)
        else:
            da1_data = da_dict[self.da1](data, idx)
            da2_data = da_dict[self.da2](data, idx)
        
        
        if self.da3 == "NA": # 2 Encoder
            da3_data = data
            da4_data = data
            da5_data = data
        else: # 5 Encoder
            da3_data = da_dict[self.da3](data)
            da4_data = da_dict[self.da4](data)
            da5_data = da_dict[self.da5](data)
        
        return np.array(da1_data), np.array(da2_data), np.array(da3_data), np.array(da4_data), np.array(da5_data), np.array(label), np.array(idx)
        
def data_generator(data_name, batch_size, da1, da2, da3, da4, da5, dataset_path):
    test_dataset = UCR(datapath = dataset_path+'/{}/{}_TEST_Normalized.tsv'.format(data_name, data_name), da1=da1, da2=da2, da3=da3, da4=da4, da5=da5)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle = False, num_workers = 8, pin_memory = True)
    
    if 'validation' in dataset_path:
        train_dataset = UCR(datapath = dataset_path+'/{}/{}_TRAIN2_Normalized.tsv'.format(data_name, data_name), da1=da1, da2=da2, da3=da3, da4=da4, da5=da5)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle = True, num_workers = 8, pin_memory = True)
                
        validation_dataset = UCR(datapath = dataset_path+'/{}/{}_VALIDATION_Normalized.tsv'.format(data_name, data_name), da1=da1, da2=da2, da3=da3, da4=da4, da5=da5) 
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle = False, num_workers = 8, pin_memory = True)
    else:
        train_dataset = UCR(datapath = dataset_path+'/{}/{}_TRAIN_Normalized.tsv'.format(data_name, data_name), da1=da1, da2=da2, da3=da3, da4=da4, da5=da5)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle = True, num_workers = 8, pin_memory = True)
        
        validation_loader = None
    
    return train_loader, test_loader, validation_loader
    
if __name__ == "__main__":
    data_generator("Worms", 64)
    pass
