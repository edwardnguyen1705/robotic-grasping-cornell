import sys
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

from grasp_dataset import GraspDataset
from network import GraspNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    dataset_name = 'grasp'
    dataset_path = './dataset/grasp'
    image_set = 'train'

    dataset = GraspDataset(dataset_name, image_set, dataset_path)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    loss_cls = 0
    loss_rect = 0

    model = GraspNet()
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    total_step = len(train_loader)
    for epoch in range(args.epochs):
        for i, (img, gt_rect) in enumerate(train_loader):
            #img, gt_rect  = train_iter.next()
            img = img.to(device)
            gt_cls = gt_rect[0]       # a batch of angle bin classes
            gt_cls = gt_cls.long()
            gt_cls = gt_cls.to(device)
            gt_rect = gt_rect[1]    # a batch of rect coordinates
            gt_rect = gt_rect.float()
            gt_rect = gt_rect.to(device)
            
            #print('img.requires_grad: {}'.format(img.requires_grad))
            
            rect_pred, cls_score = model(img)
            #print('rect_pred.requires_grad: {}'.format(rect_pred.requires_grad))
            
            cls_prob = F.softmax(cls_score, 1)
            #print('cls_prob.requires_grad: {}'.format(cls_prob.requires_grad))
            
            loss_cls = F.cross_entropy(cls_score, gt_cls)
            
            #print('loss_cls.requires_grad: {}'.format(loss_cls.requires_grad))
            
            bbox_inside_weights = gt_rect.new(gt_rect.size()).zero_()
            
            for b in range(gt_cls.numel()):
                #print('gt_cls[b]: {0}'.format(gt_cls[b]))
                if gt_cls[b] != 0:
                    bbox_inside_weights[b, :] = torch.tensor([1., 1., 1., 1.])
                
                
            #print('bbox_inside_weights: {0}'.format(bbox_inside_weights))
            #print('rect_pred.shape: {0}, gt_rect.shape: {1}'.format(rect_pred.shape, gt_rect.shape))
            #print('rect_pred: {0} \n gt_rect: {1}'.format(rect_pred, gt_rect))
            gt_rect = torch.mul(gt_rect, bbox_inside_weights)
            rect_pred = torch.mul(rect_pred, bbox_inside_weights)
            
            loss_rect = F.smooth_l1_loss(rect_pred, gt_rect, reduction='mean')
            loss = loss_cls + loss_rect
            print('epoch {}/{}, step: {}, loss_cls: {:.3f}, loss_rect: {:.3f}, loss: {:.3f}'.format(epoch + 1, args.epochs, i, loss_cls.item(), loss_rect.item(), loss.item()))
            
            # Backward and optimize
            # optimizer.zero_grad()
            model.zero_grad()
            loss.backward()
            optimizer.step()

    save_name = os.path.join(args.models, 'model_{}.ckpt'.format(epoch))
    torch.save({
      'epoch': epoch + 1,
      'model': model.state_dict(), # 'model' should be 'model_state_dict'
      'optimizer': optimizer.state_dict(),
      'loss': loss.item(),
    }, save_name)
    
    
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, help='number of epochs', default=1)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.0001)
    parser.add_argument('--batch-size', type=int, help='batch size', default=1)
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))




































    
    
    
    
    
    
    
