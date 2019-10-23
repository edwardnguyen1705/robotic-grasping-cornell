import os
from available_gpus import get_available_gpus

gpus = get_available_gpus(mem_lim=1024)
if len(gpus):
    if len(gpus) == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus[0]
    else:
        gpu_ids_str = ",".join(gpus_available)
        print("gpus_str: {}".format(gpu_ids_str))
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str
        
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn.functional as F
from skimage import io
import numpy as np
import cv2
from shapely.geometry import Polygon

from grasp_dataset import GraspDataset
from network import GraspNet

ar = np.array
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(gpu_ids):
    num_gpus = len(gpu_ids)     
    model = GraspNet()
    state_dict = torch.load('./models/model.ckpt', map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    if num_gpus > 1:
        model = nn.DataParallel(model_torch).to(device)
    else: 
        _device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(_device)
    
    model.eval()
    return model

def Rotate2D(pts,cnt,ang=np.pi/4):
    '''pts = {} Rotates points(nx2) about center cnt(2) by angle ang(1) in radian'''
    return np.dot(pts-cnt,ar([[np.cos(ang),np.sin(ang)],[-np.sin(ang),np.cos(ang)]]))+cnt

def vis_detections(ax, im, score, dets):
    #im = im[:, :, (2, 1, 0)]
    #fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    bbox = dets
    score = score

    # plot rotated rectangles
    pts = ar([[bbox[0],bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]])
    cnt = ar([(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2])
    angle = score
    r_bbox = Rotate2D(pts, cnt, -np.pi/2-np.pi/20*(angle-1))
    pred_label_polygon = Polygon([(r_bbox[0,0],r_bbox[0,1]), (r_bbox[1,0], r_bbox[1,1]), (r_bbox[2,0], r_bbox[2,1]), (r_bbox[3,0], r_bbox[3,1])])
    pred_x, pred_y = pred_label_polygon.exterior.xy

    plt.plot(pred_x[0:2],pred_y[0:2], color='k', alpha = 0.7, linewidth=1, solid_capstyle='round', zorder=2)
    plt.plot(pred_x[1:3],pred_y[1:3], color='r', alpha = 0.7, linewidth=3, solid_capstyle='round', zorder=2)
    plt.plot(pred_x[2:4],pred_y[2:4], color='k', alpha = 0.7, linewidth=1, solid_capstyle='round', zorder=2)
    plt.plot(pred_x[3:5],pred_y[3:5], color='r', alpha = 0.7, linewidth=3, solid_capstyle='round', zorder=2)


model = load_model(gpus)
dataset_name = 'grasp'
dataset_path = './dataset/grasp'
image_set = 'test'
# Apply this transform for just only one image
inv_normalize = transforms.Normalize(
                            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                            std=[1/0.229, 1/0.224, 1/0.225])

batch_size = 1
dataset = GraspDataset(dataset_name, image_set, dataset_path)
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

with torch.no_grad():
    for i, (img, gt_rect) in enumerate(test_loader):
        
        img = img.to(device)
        #print('img.size(): {}'.format(img.size()))
        rect_pred, cls_score = model(img)   
        cls_score = cls_score.squeeze()
        rect_pred = rect_pred.squeeze()
        #print('cls_score.shape: {}'.format(cls_score.shape))
        cls_prob = F.softmax(cls_score,0)
        #print('cls_prob: {0}'.format(cls_prob))
        #print('rect_pred: {0}'.format(rect_pred))
        
        img = img.cpu()
        img = img[0,:,:,:]
        img = inv_normalize(img)
        img = img.numpy()
        # CxHxW -> HxWxC
        img = np.transpose(img,(1,2,0))
        
        cls_score = cls_score.cpu()
        cls_score = cls_score.detach().numpy()
        ind_max = np.argmax(cls_score)
        #print('ind_max: {}, cls_score[{}]: {}'.format(ind_max, ind_max, cls_score[ind_max]))
        
        rect_pred = rect_pred.cpu()
        rect_pred = rect_pred.detach().numpy()
        print('rect_pred: {0}'.format(rect_pred))
        
        p1 = (rect_pred[0], rect_pred[1])
        p2 = (rect_pred[0] + rect_pred[2], rect_pred[1])
        p3 = (rect_pred[0] + rect_pred[2], rect_pred[1] + rect_pred[3])
        p4 = (rect_pred[0], rect_pred[1] + rect_pred[3])

        # Create figure and axes
        fig,ax = plt.subplots(1)
        # Display the image
        ax.imshow(img)
        vis_detections(ax, img, ind_max, rect_pred)
        plt.draw()
        plt.show()
        
        '''
        cv2.line(img, p1, p2, (0, 0, 255), 2)
        cv2.line(img, p2, p3, (0, 0, 255), 2)
        cv2.line(img, p3, p4, (0, 0, 255), 2)
        cv2.line(img, p4, p1, (0, 0, 255), 2)
        cv2.imshow('bbox', img)
        cv2.waitKey(0)
        '''
        
        if i > 5:
            break










