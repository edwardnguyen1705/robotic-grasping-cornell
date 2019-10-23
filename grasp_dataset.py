import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
#import imageio
from skimage import io
import os
import numpy as np
import statistics
import matplotlib.pyplot as plt
import pickle

class GraspDataset(Dataset):
    def __init__(self, name, image_set, dataset_path):
        """transforms.ToTensor(): Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
        Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
        [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
        """
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([transforms.ToTensor(), normalize]) 
        self.name = name
        self.image_set = image_set
        self.dataset_path = dataset_path
        self.cache_path = os.path.join(dataset_path, 'cache')
        self.classes = ('__background__', # always index 0
                         'bin_01', 'bin_02', 'bin_03', 'bin_04', 'bin_05',
                         'bin_06', 'bin_07', 'bin_08', 'bin_09', 'bin_10',
                         'bin_11', 'bin_12', 'bin_13', 'bin_14', 'bin_15',
                         'bin_16', 'bin_17', 'bin_18', 'bin_19')
        self.num_classes = len(self.classes)
        self.class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        
        self.image_ext = ['.png']
        self.img_width = 224
        self.img_height = 224
        self.txt_empty_list = [] # updated in func load_annotation(.)
        gt_rectdb = self.get_rectdb()
        self.gt_rectdb = gt_rectdb[0]
        self.image_indices = gt_rectdb[1]
        
    def __getitem__(self, index): # index: int i.e. 0, 1, 2
        img = io.imread(self.image_path_at(index))
        #print('img.shape: {0}'.format(img.shape))
        gt_rects_org = self.gt_rectdb[index]
        
        gt_classes = gt_rects_org['gt_classes']
        gt_rects = gt_rects_org['gt_rects']
        
        gt_classes = np.sort(gt_classes)
        #print('sorted gt_classes: {}'.format(gt_classes))
        # print('gt_classes.size: {}'.format(gt_classes.size))
        if (gt_classes.size == 1):
            gt_cls = gt_classes[0]
            gt_rect = gt_rects[0]
        else:
            unique, counts = np.unique(gt_classes, return_counts=True)
            #print('unique: {}, counts: {}'.format(unique, counts))
            # background class
            if (unique[0] == 0) and (np.around(counts[0]/counts.sum(),decimals=2) > np.around(1./counts.size, decimals=2)):
                gt_cls = gt_classes[0]
                gt_rect = gt_rects[0]
                #print('gt_cls: {}, gt_rect: {}'.format(gt_cls, gt_rect))
            else: 
                # Get median index                   
                #i = np.argsort(gt_classes)[gt_classes.size//2]
                # Get the cls having max frequency
                i = np.argmax(counts)
                gt_cls = unique[i]
                j = np.where(gt_classes == gt_cls)[0]
                gt_rect = gt_rects[j[0]]
                #print('i: {}, gt_cls: {}, gt_rect: {}'.format(i, gt_cls, gt_rect))

        
        '''
        num_rects = len(gt_classes)
        i = np.random.randint(num_rects)
        gt_cls = gt_classes[i]
        gt_rect = gt_rects[i]   
        '''
        
        #print('gt_cls: {}'.format(gt_cls))
        #print('gt_rect: {}'.format(gt_rect))
        gt_cls = torch.tensor(gt_cls)
        gt_rect = torch.tensor(gt_rect)
        gt_rect = [gt_cls, gt_rect]
        #img = torch.from_numpy(img)
        img = self.transform(img)
        
        return img, gt_rect
    
    def __len__(self):
        return len(self.gt_rectdb)
    
    def load_img_set_ind(self):
        '''
        return all image indices: pcd0101r_preprocessed_1, etc.
        '''
        image_set_file = os.path.join(self.dataset_path, 'ImageSets', 
                                        self.image_set + '.txt')
        with open(image_set_file) as f:
            image_indices = [x.strip() for x in f.readlines()]
        return image_indices
        
    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        index is pcd0101r_preprocessed_1 for example
        """
        for ext in self.image_ext:
            image_path = os.path.join(self.dataset_path, 'Images', index + ext)
            #print('image_path: {0}'.format(image_path))
            if os.path.exists(image_path):
                break
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path
        
    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self.image_indices[i])
    
    def load_annotation(self, index):
        '''
        Load cls, and rect in an image
        index is pcd0101r_preprocessed_1 for example
        '''
        
        filename = os.path.join(self.dataset_path, 'Annotations', index + '.txt')
        
        # if the file is empty
        if (os.stat(filename).st_size) == 0:
            self.txt_empty_list.append(index)
            print('Empty files: {}'.format(filename))
        else:
            print('Loading: {}'.format(filename))
            with open(filename) as f:
                data = f.readlines()
        
            num_objs = len(data)
            #print('Num of rects in image {0} is {1}'.format(index, len(data)))            
            
            gt_rects = np.zeros((num_objs, 4), dtype=np.uint8)
            gt_classes = np.zeros((num_objs), dtype=np.int32)
            
            # Load object rects into a data frame.
            for i, line in enumerate(data):
                # strip(): deletes white spaces from the begin and the end of line
                # split(): splits line into elements of a list by space separator
                obj = line.strip().split()
                if len(obj) != 5: # cls x1 y1 x2 y2
                    continue
                
                cls = int(obj[0])
                x1 = float(obj[1]) 
                y1 = float(obj[2]) 
                x2 = float(obj[3]) 
                y2 = float(obj[4])
                
                if ((x1 < 0) or (x1 > self.img_width) or (x2 < 0) or (x2 > self.img_width) or (y1 < 0) or (y1 > self.img_height) or (y2 < 0) or (y2 > self.img_height)):
                    continue
                
                gt_classes[i] = cls
                gt_rects[i, :] = [x1, y1, x2, y2]
            
            return {'gt_classes': gt_classes, 'gt_rects': gt_rects}
    
    def get_rectdb(self):
        """
        Return the database of ground-truth rects.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_rectdb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                rectdb = pickle.load(fid)
            print('{0} gt rectdb being loaded from {1}'.format(self.name, cache_file))
            return rectdb
        
        self.image_indices = self.load_img_set_ind()
        gt_rectdb = [self.load_annotation(index)
                    for index in self.image_indices]
                        
        # remove elements that have empty txt file
        for idx in self.txt_empty_list:
            self.image_indices.remove(idx)
        for i in range(gt_rectdb.count(None)):
            gt_rectdb.remove(None)
        
        
        gt_rectdb = [gt_rectdb, self.image_indices]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_rectdb, fid, pickle.HIGHEST_PROTOCOL)
        print('writing gt rectdb to {}'.format(cache_file))

        return gt_rectdb

if __name__ == '__main__':
    name = 'grasp'
    dataset_path = './dataset/grasp'
    image_set = 'train'
    inv_normalize = transforms.Normalize(
                            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                            std=[1/0.229, 1/0.224, 1/0.225])
    
    train_dataset = GraspDataset(name, image_set, dataset_path)
    print('len(train_dataset): {0}'.format(len(train_dataset)))
    #print('image_indices: {0}'.format(train_dataset.image_indices))
    #print('txt_empty_list: {0}'.format(train_dataset.txt_empty_list))
    print('len(gt_rectdb): {0}'.format(len(train_dataset.gt_rectdb)))
    #print('gt_rectdb: {0}'.format(train_dataset.gt_rectdb))
        

    img, gt_rect = train_dataset.__getitem__(10)
    img = inv_normalize(img)
    # CxHxW -> HxWxC
    img = np.transpose(img,(1,2,0))
    print('gt_cls: {0},\n gt_rect: {1}'.format(gt_rect[0], gt_rect[1]))
    plt.imshow(img)
    plt.show()

