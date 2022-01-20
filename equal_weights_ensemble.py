import random
import torch, os, glob
from operator import itemgetter
import argparse, datetime
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchinfo import summary
import torch.optim as optim
import torch.nn.functional as F

from model.dda import *
from utils.dataload import data_generator
from utils.write_csv import *

parser = argparse.ArgumentParser(description='Sequence Modeling')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--iterations', type=int, default=10000,
                    help='iteration (default: 10000)')
parser.add_argument('--dataset', default='Crop',
                    help='UCR dataset (default: Crop)')
parser.add_argument('--da1', default='identity',
                    help='Data Augmentation 1 (default: identity)')
parser.add_argument('--da2', default='jitter',
                    help='Data Augmentation 2 (default: jitter)')
parser.add_argument('--da3', default='windowWarp',
                    help='Data Augmentation 3 (default: windowWarp)')
parser.add_argument('--da4', default='magnitudeWarp',
                    help='Data Augmentation 4 (default: magnitudeWarp)')
parser.add_argument('--da5', default='timeWarp',
                    help='Data Augmentation 5 (default: timeWarp)')
parser.add_argument('--gpu_id', type=int, default=0,
                    help='set gpu_id')
args = parser.parse_args()

dt = datetime.datetime.now()
now = "{:0=2}".format(dt.month) + "{:0=2}-".format(dt.day) + "{:0=2}".format(dt.hour) + "{:0=2}".format(dt.minute)

batch_size = args.batch_size
data_name = args.dataset
gpu_id = args.gpu_id

dataset_path = './dataset/UCRArchive_2018'
input_length, n_classes, NumOfTrain = get_length_numofclass(data_name)

z_size = 512 # f's size
input_channels = 1
seq_length = int(input_length / input_channels) 
epochs = np.ceil(args.iterations * (batch_size / NumOfTrain)).astype(int)
data_len_after_cnn = int((((((input_length-2)/2)-2)/2)-2)/2)

steps = 0
da1, da2, da3, da4, da5 = args.da1, args.da2, args.da3, args.da4, args.da5

print(args, 'epochs:{}'.format(epochs))
train_loader, test_loader = data_generator(data_name, batch_size, da1, da2, da3, da4, da5, dataset_path)
ts_encoder1, classifier1 = TSEncoder(data_len_after_cnn, z_size), Classifier(n_classes, z_size)
ts_encoder2, classifier2 = TSEncoder(data_len_after_cnn, z_size), Classifier(n_classes, z_size)
ts_encoder3, classifier3 = TSEncoder(data_len_after_cnn, z_size), Classifier(n_classes, z_size)
ts_encoder4, classifier4 = TSEncoder(data_len_after_cnn, z_size), Classifier(n_classes, z_size)
ts_encoder5, classifier5 = TSEncoder(data_len_after_cnn, z_size), Classifier(n_classes, z_size)

ts_encoder1.cuda(gpu_id)
ts_encoder2.cuda(gpu_id)
ts_encoder3.cuda(gpu_id)
ts_encoder4.cuda(gpu_id)
ts_encoder5.cuda(gpu_id)
classifier1.cuda(gpu_id)
classifier2.cuda(gpu_id)
classifier3.cuda(gpu_id)
classifier4.cuda(gpu_id)
classifier5.cuda(gpu_id)

MSE_loss, CE_loss = nn.MSELoss(), nn.CrossEntropyLoss()


def test_model(epoch):
    global now
    test_loss = 0.
    correct = 0
    base_path = './ensemble'
    
    ts_encoder1.load_state_dict(torch.load(base_path+'/{}_{}_{}/ts_encoder1.pth'.format(data_name, da1, epoch), map_location='cuda:0'))
    classifier1.load_state_dict(torch.load(base_path+'/{}_{}_{}/classifier.pth'.format(data_name, da1, epoch), map_location='cuda:0'))
    ts_encoder2.load_state_dict(torch.load(base_path+'/{}_{}_{}/ts_encoder1.pth'.format(data_name, da2, epoch), map_location='cuda:0'))
    classifier2.load_state_dict(torch.load(base_path+'/{}_{}_{}/classifier.pth'.format(data_name, da2, epoch), map_location='cuda:0'))
    ts_encoder3.load_state_dict(torch.load(base_path+'/{}_{}_{}/ts_encoder1.pth'.format(data_name, da3, epoch), map_location='cuda:0'))
    classifier3.load_state_dict(torch.load(base_path+'/{}_{}_{}/classifier.pth'.format(data_name, da3, epoch), map_location='cuda:0'))
    ts_encoder4.load_state_dict(torch.load(base_path+'/{}_{}_{}/ts_encoder1.pth'.format(data_name, da4, epoch), map_location='cuda:0'))
    classifier4.load_state_dict(torch.load(base_path+'/{}_{}_{}/classifier.pth'.format(data_name, da4, epoch), map_location='cuda:0'))
    ts_encoder5.load_state_dict(torch.load(base_path+'/{}_{}_{}/ts_encoder1.pth'.format(data_name, da5, epoch), map_location='cuda:0'))
    classifier5.load_state_dict(torch.load(base_path+'/{}_{}_{}/classifier.pth'.format(data_name, da5, epoch), map_location='cuda:0'))
    
    ts_encoder1.eval()
    classifier1.eval()
    ts_encoder2.eval()
    classifier2.eval()
    ts_encoder3.eval()
    classifier3.eval()
    ts_encoder4.eval()
    classifier4.eval()
    ts_encoder5.eval()
    classifier5.eval()

    with torch.no_grad():
        for da1_data, da2_data, da3_data, da4_data, da5_data, target, _ in test_loader:
            da1_data, da2_data, target = da1_data.cuda(gpu_id).to(dtype=torch.float), da2_data.cuda(gpu_id).to(dtype=torch.float), target.cuda(gpu_id)
            da3_data, da4_data, da5_data = da3_data.cuda(gpu_id).to(dtype=torch.float), da4_data.cuda(gpu_id).to(dtype=torch.float), da5_data.cuda(gpu_id).to(dtype=torch.float)

            da1_data, da2_data = da1_data.view(-1, input_channels, seq_length), da2_data.view(-1, input_channels, seq_length)
            da3_data, da4_data, da5_data = da3_data.view(-1, input_channels, seq_length), da4_data.view(-1, input_channels, seq_length), da5_data.view(-1, input_channels, seq_length)

            da1_data, da2_data, target = Variable(da1_data), Variable(da2_data), Variable(target)
            da3_data, da4_data, da5_data = Variable(da3_data), Variable(da4_data), Variable(da5_data)
            
            z1, z2, z3, z4, z5 = ts_encoder1(da1_data), ts_encoder2(da2_data), ts_encoder3(da3_data), ts_encoder4(da4_data), ts_encoder5(da5_data)
            y1, y2, y3, y4, y5 = classifier1(z1), classifier1(z2), classifier1(z3), classifier1(z4), classifier1(z5)
            
            y = y1+y2+y3+y4+y5
            
            target_list = target.to('cpu').detach().numpy() if not 'target_list' in locals() else np.concatenate([target_list, target.to('cpu').detach().numpy()])      
                        
            _, predict = torch.max(y.data, 1)
            test_correct = (predict == target).sum().item()

            loss = CE_loss(y, target)
            pred = y.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            
            pred_list = pred.to('cpu').detach().numpy() if not 'pred_list' in locals() else np.concatenate([pred_list, pred.to('cpu').detach().numpy()])
            test_loss += loss

        pred_list = np.array([item for l in pred_list for item in l ])
        test_loss /= len(test_loader.dataset)
        test_acc = correct / len(test_loader.dataset)
        print(' Test set: Average loss: {:.8f}, Accuracy: {:>4}/{:<4} ({:>3.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset), 100.*test_acc))
        

if __name__ == "__main__":
    test_model(epochs)
            
