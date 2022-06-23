import h5py
#import helpers
import numpy as np
from pathlib import Path
import torch
from torch.utils import data
import cv2
import PIL
#import png
#from numpy import asarray
from numpy import savez_compressed
import os

class HDF5Dataset(data.Dataset):
    """Represents an abstract HDF5 dataset.
    
    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and 
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(self, file_path, recursive, load_data, data_cache_size=3, transform=None):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        #self.transform = transform
        self.transform = None
        # Search for all h5 files
        p = Path(file_path)
        assert(p.is_dir())
        if recursive:
            files = sorted(p.glob('**/*.h5'))
        else:
            files = sorted(p.glob('*.h5'))
        if len(files) < 1:
            raise RuntimeError('No hdf5 datasets found')

       
        #files = files[:20]
        #print(len(files)) 
        count = 1
        for h5dataset_fp in files:
            #print(count)
            self._add_data_infos(str(h5dataset_fp.resolve()), load_data)
            count += 1
            #if count == 5:
            #    break

    def __getitem__(self, index):
        #print(index, "get item is being called")
        # get data
        x, y = self.get_data("cbed_stack", index)
        #print("Before transform: ", np.shape(x))
        #x = self.log(x)
        #x = self.resize(x, (128, 128))
        x = self.xform(x)
        #print("After log and resize transform: ", np.shape(x))
        x = np.squeeze(x)
        #x = x.type(torch.DoubleTensor)
        #print(np.shape(x))
        if self.transform:
            x = self.transform(x)
            #print("After all transform: ", np.shape(x))
        else:
            x = torch.from_numpy(x)
        x = np.squeeze(x)
        # get label
        #y = self.get_label("cbed_stack", index)
        #y = torch.from_numpy(y)
        return (x, y)
    '''
    def convert(self, index):
       
        x, y = self.get_data("cbed_stack", index)
        
        x = self.xform(x)
        print("After log and resize transform: ", np.shape(x))
        flattened = np.concatenate(x, axis=1)
        png_img = png.from_array(flattened.astype(np.uint8), 'L')
        png_img.save("test-" + str(index) + ".png")
    ''' 
    
    def convert_to_npy(self, index, directory, offset):   
        x, y = self.get_data("cbed_stack", index)
        x = self.xform(x)
        os.makedirs(directory,exist_ok=True)
        np.savez(directory + "/sample-" + str(index + offset) +  ".npz", data=x, label=np.asarray([y]))        
        #data = np.load(directory + "/sample-" + str(index) +  ".npz")
        #print(data["data"])
        #print(data["label"])
    
    def reduce_to_npy(self, index, directory, offset):   
        x, y = self.get_data("cbed_stack", index)
        x = self.xform_red(x)
        np.savez(directory + "/sample-" + str(index + offset) +  ".npz", data=x, label=np.asarray([y]))        
     

    def __len__(self):
        #print(len(self.get_data_infos('cbed_stack')))
        return len(self.get_data_infos('cbed_stack'))
    
    def _add_data_infos(self, file_path, load_data):
        with h5py.File(file_path, 'r') as h5_file:
            # Walk through all groups, extracting datasets
            for gname, group in h5_file.items():
                #print("group:", gname, list(group.items()))
                label = int(group.attrs['space_group'])
                #print(label)
             
                for dname, ds in group.items():
                    #print(dname, ds)
                    # if data is not loaded its cache index is -1
                    idx = -1
                    if load_data:
                        # add data to the data cache
                        #print(ds.value)
                        idx = self._add_to_cache(ds[()], file_path, label)
                    
                    # type is derived from the name of the dataset; we expect the dataset
                    # name to have a name such as 'data' or 'label' to identify its type
                    # we also store the shape of the data in case we need it
                    self.data_info.append({'file_path': file_path, 'type': dname, 'shape': ds[()].shape, 'cache_idx': idx, 'label' : label})

    def _load_data(self, file_path):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        #print("Inside load data function:")
        with h5py.File(file_path, 'r') as h5_file:
            #print("file-items", h5_file.items())
            
            for gname, group in h5_file.items():
                #print("group:", gname, group)
                #print("group-items:", group.items())
                label = int(group.attrs['space_group'])
                for dname, ds in group.items():
                    # add data to the data cache and retrieve
                    # the cache index
                    #print("dname", dname)
                    #print("d-value:", ds.value)
                    idx = self._add_to_cache(ds[()], file_path)
                    #print("index:", idx)
                    # find the beginning index of the hdf5 file we are looking for
                    file_idx = next(i for i,v in enumerate(self.data_info) if v['file_path'] == file_path)

                    # the data info should have the same index since we loaded it in the same way
                    self.data_info[file_idx + idx]['cache_idx'] = idx
                    self.data_info[file_idx + idx]['label'] = label
                                 

        # remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.data_cache_size:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path)
            self.data_cache.pop(removal_keys[0])
            # remove invalid cache_idx
            self.data_info = [{'file_path': di['file_path'], 'type': di['type'], 'shape': di['shape'], 'cache_idx': -1, 'label': label} if di['file_path'] == removal_keys[0] else di for di in self.data_info]

    def _add_to_cache(self, data, file_path):
        """Adds data to the cache and returns its index. There is one cache
        list for every file_path, containing all datasets in that file.
        """
        if file_path not in self.data_cache:
            self.data_cache[file_path] = [data]
        else:
            self.data_cache[file_path].append(data)
        return len(self.data_cache[file_path]) - 1

    def get_data_infos(self, type):
        """Get data infos belonging to a certain type of data.
        """
        #print("data info:", self.data_info)
        data_info_type = [di for di in self.data_info if di['type'] == type]
        return data_info_type

    def get_data(self, type, i):
        """Call this function anytime you want to access a chunk of data from the
            dataset. This will make sure that the data is loaded in case it is
            not part of the data cache.
        """
        fp = self.get_data_infos(type)[i]['file_path']
        if fp not in self.data_cache:
            self._load_data(fp)
        
        # get new cache_idx assigned by _load_data_info
        cache_idx = self.get_data_infos(type)[i]['cache_idx']
        label = self.get_data_infos(type)[i]["label"]
        return self.data_cache[fp][cache_idx], label
  
    def log(self, img, param1=-17, param2=10):
        floor = param1
        scaled_img = np.maximum(floor, np.log(img)) - floor
        scaled_img *= param2 
        scaled_img = np.clip(scaled_img, 0, 255)
        return scaled_img

    def resize(self, img, size):
        resized_img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        return resized_img

    def xform(self, cbed_stack):
        xformed_cbed_stack = []
        for cbed in cbed_stack:
            lcbed = self.log(cbed)
            rcbed = self.resize(lcbed, (128, 128))
            xformed_cbed_stack.append(rcbed)
        return np.array(xformed_cbed_stack)
    
    def xform_red(self, cbed_stack):
        xformed_cbed_stack = []
        for cbed in cbed_stack:
            #lcbed = self.log(cbed)
            rcbed = self.resize(cbed, (128, 128))
            xformed_cbed_stack.append(rcbed)
        return np.array(xformed_cbed_stack)
        
        
       
