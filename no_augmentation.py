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
parser.add_argument('--gpu_id', type=int, default=0,
                    help='set gpu_id for server')
parser.add_argument('--iterations', type=int, default=10000,
                    help='iteration (default: 10000)')
parser.add_argument('--lr', type=float, default=2e-3,
                    help='initial learning rate (default: 2e-3)')
parser.add_argument('--dataset', default='Crop',
                    help='UCR dataset (default: Crop)')
parser.add_argument('--da', default='identity',
                    help='Data Augmentation (default: identity)')
parser.add_argument('--seed', default=1111,
                    help='random seed')
args = parser.parse_args()

# fix random
seed = args.seed
if True: # set seed
    np.random.seed(seed)
    torch.manual_seed(seed) # fix the initial value of the network weight
    torch.cuda.manual_seed(seed) # for cuda
    torch.cuda.manual_seed_all(seed) # for multi-GPU
    torch.backends.cudnn.deterministic = True # choose the determintic algorithm

dt = datetime.datetime.now()
now = "{:0=2}".format(dt.month) + "{:0=2}-".format(dt.day) + "{:0=2}".format(dt.hour) + "{:0=2}".format(dt.minute)

batch_size = args.batch_size
data_name = args.dataset
gpu_id = args.gpu_id
limit_num = args.limit_num

dataset_path = './dataset/UCRArchive_2018'
input_length, n_classes, NumOfTrain = get_length_numofclass(data_name)

z_size = 512 # f's size
input_channels = 1
seq_length = int(input_length / input_channels) 
epochs = np.ceil(args.iterations * (batch_size / NumOfTrain)).astype(int)
data_len_after_cnn = int((((((input_length-2)/2)-2)/2)-2)/2)

steps = 0
da1, da2, da3, da4, da5 = args.da, 'identity', "NA", "NA", "NA" # da2 is not used.

print(args, 'epochs:{}'.format(epochs))
train_loader, test_loader, _ = data_generator(data_name, batch_size, da1, da2, da3, da4, da5, dataset_path)

ts_encoder1 = TSEncoder(data_len_after_cnn, z_size)
classifier = Classifier(n_classes, z_size)

ts_encoder1.cuda(gpu_id)
classifier.cuda(gpu_id)
    
lr = args.lr
optimizer = optim.Adam(
    [{'params': ts_encoder1.parameters()}, 
    {'params': classifier.parameters()},
    ], lr=lr, betas=(0.5, 0.999))

CE_loss = nn.CrossEntropyLoss()

def train(ep):
    global steps, now
    train_loss = 0.
    correct = 0
    ts_encoder1.train()
    classifier.train()
    
    for da1_data, _, _, _, _, target, _ in train_loader:
        da1_data, target = da1_data.cuda(gpu_id).to(dtype=torch.float), target.cuda(gpu_id)
        da1_data = da1_data.view(-1, input_channels, seq_length)
        da1_data, target = Variable(da1_data), Variable(target)
        
        z = ts_encoder1(da1_data)
        y = classifier(z)
        loss = CE_loss(y, target)
        
        optimizer.zero_grad()
        pred = y.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss.backward()
        optimizer.step()
        
        train_loss += loss
    train_loss /= len(train_loader.dataset)
    #print('Train set: Average loss: {:.8f}, Accuracy: {:>4}/{:<4} ({:>3.0f}%) Average Params: {}|{:.4f}, {}|{:.4f}'.format(
    #    train_loss, correct, len(train_loader.dataset), 100. * correct / len(train_loader.dataset), da1, params_mean, da2, 1-params_mean))
            
    return train_loss, correct/len(train_loader.dataset)

def test(epoch):
    global now
    test_loss = 0.
    correct = 0
    ts_encoder1.eval()
    classifier.eval()

    with torch.no_grad():
        for da1_data, _, _, _, _, target, _ in test_loader:
            da1_data, target = da1_data.cuda(gpu_id).to(dtype=torch.float), target.cuda(gpu_id)
            da1_data = da1_data.view(-1, input_channels, seq_length)
            da1_data, target = Variable(da1_data), Variable(target)
            
            z = ts_encoder1(da1_data)
            y = classifier(z)
            
            z_list = z if not 'z_list' in locals() else torch.cat([z_list, z])
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
        print(' Test set: Average loss: {:.8f}, Accuracy: {:>4}/{:<4} ({:>3.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
            
        random.seed(42) # must
        idx_list = np.arange(len(target_list))
        if len(target_list)>limit_num:
            idx_list = random.sample(list(idx_list), limit_num)
            draw_target = itemgetter(idx_list)(target_list)
            draw_pred = itemgetter(idx_list)(pred_list)
            draw_z = itemgetter(idx_list)(z_list[:])
        else:
            draw_target = target_list
            draw_pred = pred_list
            draw_z = z_list[:]

        return test_loss, correct / len(test_loader.dataset)

def test_model():
    base_path = './result/No-Aug_{}_{}_{}'.format(data_name, args.da, epochs)
    ts_encoder1.load_state_dict(torch.load(base_path+'/ts_encoder1.pth', map_location='cuda:0'))
    classifier.load_state_dict(torch.load(base_path+'/classifier.pth', map_location='cuda:0'))
    test(epochs) # pseudo epoch.
    exit(0)

if __name__ == "__main__":
    #test_model()
    best_loss, best_acc = 10e5, 0.
    for epoch in range(1, epochs+1):
        print('Epoch:{}/{}'.format(epoch, epochs))
        train_loss, train_acc = train(epoch)
        if epoch%25==0 or epoch==epochs or epoch==1:
            test_loss, test_acc = test(epoch)
        #else:
            #test_loss, test_acc = test(epoch)
        
            # save to tsv file
            detached_train_acc = train_acc.to('cpu').detach().numpy().tolist()
            detached_train_loss = train_loss.to('cpu').detach().numpy().tolist()
            detached_test_acc = test_acc.to('cpu').detach().numpy().tolist()
            detached_test_loss = test_loss.to('cpu').detach().numpy().tolist()
            update_csv_ts1(data_name, 'No-Aug', '{}/{}'.format(dataset_path, data_name), detached_train_acc, detached_train_loss, detached_test_acc, detached_test_loss, epoch, epochs, da1, now, otherparams=None)
            
        if epoch==epochs:
            model_save_path = './result/No-Aug_{}_{}_{}/'.format(data_name, args.da, epoch)
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            torch.save(ts_encoder1.state_dict(), model_save_path+'ts_encoder1.pth')
            torch.save(classifier.state_dict(), model_save_path+'classifier.pth')
            
