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
from utils.draw_graphs import draw_graph, draw_hist
from utils.write_csv import *
from utils.alphavis import alpha_pca
from utils.feature_pca import feature_pca

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
parser.add_argument('--consis_lambda', type=float, default=1.0,
                    help='weights for consistency loss')
parser.add_argument('--server_id', type=int, default=0,
                    help='when run on local, set it to 0.')
parser.add_argument('--limit_num', type=int, default=300,
                    help='max vizualized samples')
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
da1, da2, da3, da4, da5 = args.da1, args.da2, args.da3, args.da4, args.da5

print(args, 'epochs:{}'.format(epochs))
train_loader, test_loader, _ = data_generator(data_name, batch_size, da1, da2, da3, da4, da5, dataset_path)

ts_encoder1, ts_encoder2 = TSEncoder(data_len_after_cnn, z_size), TSEncoder(data_len_after_cnn, z_size)
ts_encoder3, ts_encoder4 = TSEncoder(data_len_after_cnn, z_size), TSEncoder(data_len_after_cnn, z_size)
ts_encoder5 = TSEncoder(data_len_after_cnn, z_size)
classifier = Classifier(n_classes, z_size*5) # for concat

ts_encoder1.cuda(gpu_id)
ts_encoder2.cuda(gpu_id)
ts_encoder3.cuda(gpu_id)
ts_encoder4.cuda(gpu_id)
ts_encoder5.cuda(gpu_id)
classifier.cuda(gpu_id)
    
lr = args.lr
optimizer = optim.Adam(
    [{'params': ts_encoder1.parameters()},
    {'params': ts_encoder2.parameters()},
    {'params': ts_encoder3.parameters()},
    {'params': ts_encoder4.parameters()},
    {'params': ts_encoder5.parameters()},
    {'params': classifier.parameters()},
    ], lr=lr, betas=(0.5, 0.999))

MSE_loss, CE_loss = nn.MSELoss(), nn.CrossEntropyLoss()

def train(ep):
    global steps, now
    train_loss = 0.
    correct = 0
    ts_encoder1.train()
    ts_encoder2.train()
    ts_encoder3.train()
    ts_encoder4.train()
    ts_encoder5.train()
    classifier.train()
    
    for da1_data, da2_data, da3_data, da4_data, da5_data, target, _ in train_loader:
        da1_data, da2_data, target = da1_data.cuda(gpu_id).to(dtype=torch.float), da2_data.cuda(gpu_id).to(dtype=torch.float), target.cuda(gpu_id)
        da3_data, da4_data, da5_data = da3_data.cuda(gpu_id).to(dtype=torch.float), da4_data.cuda(gpu_id).to(dtype=torch.float), da5_data.cuda(gpu_id).to(dtype=torch.float)
        
        da1_data, da2_data = da1_data.view(-1, input_channels, seq_length), da2_data.view(-1, input_channels, seq_length)
        da3_data, da4_data, da5_data = da3_data.view(-1, input_channels, seq_length), da4_data.view(-1, input_channels, seq_length), da5_data.view(-1, input_channels, seq_length)
        
        da1_data, da2_data, target = Variable(da1_data), Variable(da2_data), Variable(target)
        da3_data, da4_data, da5_data = Variable(da3_data), Variable(da4_data), Variable(da5_data)
        
        z1, z2, z3, z4, z5 = ts_encoder1(da1_data), ts_encoder2(da2_data), ts_encoder3(da3_data), ts_encoder4(da4_data), ts_encoder5(da5_data)
        z = torch.cat((z1,z2,z3,z4,z5), 1)
        y = classifier(z)
        
        loss = CE_loss(y, target)
        
        optimizer.zero_grad()
        pred = y.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss.backward()
        optimizer.step()
        
        train_loss += loss
    train_loss /= len(train_loader.dataset)
    #print('     Train set: Average loss: {:.8f}, Accuracy: {:>4}/{:<4} ({:>3.0f}%)'.format(
    #    train_loss, correct, len(train_loader.dataset), 100. * correct / len(train_loader.dataset)))
            
    return train_loss, correct/len(train_loader.dataset)

def test(epoch, feature_img_path=None):
    global now
    test_loss = 0.
    correct = 0
    ts_encoder1.eval()
    ts_encoder2.eval()
    ts_encoder3.eval()
    ts_encoder4.eval()
    ts_encoder5.eval()
    classifier.eval()

    with torch.no_grad():
        for da1_data, da2_data, da3_data, da4_data, da5_data, target, _ in test_loader:
            da1_data, da2_data, target = da1_data.cuda(gpu_id).to(dtype=torch.float), da2_data.cuda(gpu_id).to(dtype=torch.float), target.cuda(gpu_id)
            da3_data, da4_data, da5_data = da3_data.cuda(gpu_id).to(dtype=torch.float), da4_data.cuda(gpu_id).to(dtype=torch.float), da5_data.cuda(gpu_id).to(dtype=torch.float)
            da1_data = da1_data.view(-1, input_channels, seq_length)
            da2_data = da2_data.view(-1, input_channels, seq_length)
            da3_data = da3_data.view(-1, input_channels, seq_length)
            da4_data = da4_data.view(-1, input_channels, seq_length)
            da5_data = da5_data.view(-1, input_channels, seq_length)
            da1_data, da2_data, target = Variable(da1_data), Variable(da2_data), Variable(target)
            da3_data, da4_data, da5_data = Variable(da3_data), Variable(da4_data), Variable(da5_data)
            
            z1, z2, z3, z4, z5 = ts_encoder1(da1_data), ts_encoder2(da2_data), ts_encoder3(da3_data), ts_encoder4(da4_data), ts_encoder5(da5_data)
            
            z1_list = z1 if not 'z1_list' in locals() else torch.cat([z1_list, z1])
            z2_list = z2 if not 'z2_list' in locals() else torch.cat([z2_list, z2])
            z3_list = z3 if not 'z3_list' in locals() else torch.cat([z3_list, z3])
            z4_list = z4 if not 'z4_list' in locals() else torch.cat([z4_list, z4])
            z5_list = z5 if not 'z5_list' in locals() else torch.cat([z5_list, z5])
            
            target_list = target.to('cpu').detach().numpy() if not 'target_list' in locals() else np.concatenate([target_list, target.to('cpu').detach().numpy()])            

            z = torch.cat((z1,z2,z3,z4,z5), 1)
            y = classifier(z)
            
            _, predict = torch.max(y.data, 1)
            test_correct = (predict == target).sum().item()

            loss = CE_loss(y, target)
            
            pred = y.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            
            pred_list = pred.to('cpu').detach().numpy() if not 'pred_list' in locals() else np.concatenate([pred_list, pred.to('cpu').detach().numpy()])
            test_loss += loss

        pred_list = np.array([item for l in pred_list for item in l ])
        test_loss /= len(test_loader.dataset)
        print('      Test set: Average loss: {:.8f}, Accuracy: {:>4}/{:<4} ({:>3.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
        
        random.seed(42) # must
        idx_list = np.arange(len(target_list))
        if len(target_list)>limit_num:
            idx_list = random.sample(list(idx_list), limit_num)
            draw_target = itemgetter(idx_list)(target_list)
            draw_pred = itemgetter(idx_list)(pred_list)
            draw_z1 = itemgetter(idx_list)(z1_list[:])
            draw_z2 = itemgetter(idx_list)(z2_list[:])
            draw_z3 = itemgetter(idx_list)(z3_list[:])
            draw_z4 = itemgetter(idx_list)(z4_list[:])
            draw_z5 = itemgetter(idx_list)(z5_list[:])
        else:
            draw_target = target_list
            draw_pred = pred_list
            draw_z1 = z1_list[:]
            draw_z2 = z2_list[:]
            draw_z3 = z3_list[:]
            draw_z4 = z4_list[:]
            draw_z5 = z5_list[:]
        
        # only when using saved model
        #feature_pca("TSNE", data_name, da1, draw_z1, -1, da2, draw_z2, -1, da3, draw_z3, -1, da4, draw_z4, -1, da5, draw_z5, -1, epoch, 100*correct/len(test_loader.dataset), './', -1)
        
        torch.cuda.empty_cache()
        
        return test_loss, correct / len(test_loader.dataset), -1, -1, -1, -1, -1

def test_model():
    base_path = './result/Concat_{}_identity-jitter-windowWarp-magnitudeWarp-timeWarp_{}_{}'.format(data_name, args.consis_lambda, epochs)
    ts_encoder1.load_state_dict(torch.load(base_path+'/ts_encoder1.pth', map_location='cuda:0'))
    ts_encoder2.load_state_dict(torch.load(base_path+'/ts_encoder2.pth', map_location='cuda:0'))
    ts_encoder3.load_state_dict(torch.load(base_path+'/ts_encoder3.pth', map_location='cuda:0'))
    ts_encoder4.load_state_dict(torch.load(base_path+'/ts_encoder4.pth', map_location='cuda:0'))
    ts_encoder5.load_state_dict(torch.load(base_path+'/ts_encoder5.pth', map_location='cuda:0'))
    classifier.load_state_dict(torch.load(base_path+'/classifier.pth', map_location='cuda:0'))
    test(epochs, base_path) # pseudo epoch.
    exit(0)

if __name__ == "__main__":
    #test_model()
    best_loss, best_acc = 10e5, 0.
    for epoch in range(1, epochs+1):
        print('Epoch:{}/{}'.format(epoch, epochs))
        train_loss, train_acc = train(epoch)
        if epoch%25==0 or epoch==epochs or epoch==1:
            test_loss, test_acc, p1, p2, p3, p4, p5 = test(epoch)
        
            # save to tsv file
            detached_train_acc, detached_train_loss = train_acc.to('cpu').detach().numpy().tolist(), train_loss.to('cpu').detach().numpy().tolist()
            detached_test_acc, detached_test_loss = test_acc.to('cpu').detach().numpy().tolist(), test_loss.to('cpu').detach().numpy().tolist()
            detached_p1, detached_p2, detached_p3, detached_p4, detached_p5 = p1, p2, p3, p4, p5
            update_csv_ts5(data_name, 'Concat', '{}/{}'.format(dataset_path, data_name), detached_train_acc, detached_train_loss, detached_test_acc, detached_test_loss, epoch, epochs, da1, detached_p1, da2, detached_p2, da3, detached_p3, da4, detached_p4, da5, detached_p5, args.consis_lambda, now, otherparams=None)
            
            if epoch==epochs:
                model_save_path = './result/Concat_{}_{}-{}-{}-{}-{}_{}_{}/'.format(data_name, args.da1, args.da2, args.da3, args.da4, args.da5, args.consis_lambda, epoch)
                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)
                torch.save(ts_encoder1.state_dict(), model_save_path+'ts_encoder1.pth')
                torch.save(ts_encoder2.state_dict(), model_save_path+'ts_encoder2.pth')
                torch.save(ts_encoder3.state_dict(), model_save_path+'ts_encoder3.pth')
                torch.save(ts_encoder4.state_dict(), model_save_path+'ts_encoder4.pth')
                torch.save(ts_encoder5.state_dict(), model_save_path+'ts_encoder5.pth')
                torch.save(classifier.state_dict(), model_save_path+'classifier.pth')
            
