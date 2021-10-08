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
from utils.write_gspread import *
from utils.alphavis import alpha_number_line

parser = argparse.ArgumentParser(description='Sequence Modeling')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--gpu_id', type=int, default=0,
                    help='set gpu_id for server')
parser.add_argument('--iterations', type=int, default=10000,
                    help='iteration (default: 10000)')
parser.add_argument('--lr', type=float, default=2e-3,
                    help='initial learning rate (default: 2e-3)')
parser.add_argument('--dataset', default='Adiac',
                    help='UCR dataset (default: Adiac)')
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
parser.add_argument('--shareWeights', action='store_true',
                    help='share network weights')
parser.add_argument('--server_id', type=int, default=0,
                    help='when run on local, set it to 0.')
parser.add_argument('--limit_num', type=int, default=300,
                    help='max vizualized samples')
parser.add_argument('--sheet', default="Trial",
                    help='google spreadsheet name')
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

dataset_path = '/workdir/DataRepo{:0>2}/UCR/UCRArchive_2018'.format(args.server_id) if args.server_id!=0 else '/home/daisuke/Documents/CloudStation/lab/DataRepo/UCR/UCRArchive_2018'
input_length, n_classes, NumOfTrain = get_length_numofclass(data_name, dataset_path)
sheet_name = args.sheet

z_size = 512 # f's size
input_channels = 1
seq_length = int(input_length / input_channels) 
epochs = np.ceil(args.iterations * (batch_size / NumOfTrain)).astype(int)
print(batch_size, epochs)
data_len_after_cnn = int((((((input_length-2)/2)-2)/2)-2)/2)

steps = 0
da1, da2, da3, da4, da5 = args.da1, args.da2, "NA", "NA", "NA"

print(args, 'epochs:{}'.format(epochs))
train_loader, test_loader, _ = data_generator(data_name, batch_size, da1, da2, da3, da4, da5, dataset_path)

ts_encoder1, ts_encoder2 = TSEncoder(data_len_after_cnn, z_size), TSEncoder(data_len_after_cnn, z_size)
classifier = Classifier(n_classes, z_size*2) # for concat

ts_encoder1.cuda(gpu_id)
ts_encoder2.cuda(gpu_id)
classifier.cuda(gpu_id)
    
lr = args.lr
optimizer = optim.Adam(
    [{'params': ts_encoder1.parameters()}, 
    {'params': ts_encoder2.parameters()}, 
    {'params': classifier.parameters()},
    ], lr=lr, betas=(0.5, 0.999))

MSE_loss, CE_loss = nn.MSELoss(), nn.CrossEntropyLoss()

def train(ep):
    global steps, now
    train_loss = 0.
    correct = 0
    ts_encoder1.train()
    ts_encoder2.train()
    classifier.train()
    
    for da1_data, da2_data, _, _, _, target, _ in train_loader:
        da1_data, da2_data, target = da1_data.cuda(gpu_id).to(dtype=torch.float), da2_data.cuda(gpu_id).to(dtype=torch.float), target.cuda(gpu_id)
        da1_data = da1_data.view(-1, input_channels, seq_length)
        da2_data = da2_data.view(-1, input_channels, seq_length)
        da1_data, da2_data, target = Variable(da1_data), Variable(da2_data), Variable(target)
        
        z1, z2 = ts_encoder1(da1_data), ts_encoder2(da2_data)
        z = torch.cat((z1,z2), 1)
        y = classifier(z)
        loss = CE_loss(y, target)
        
        optimizer.zero_grad()
        pred = y.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss.backward()
        optimizer.step()
        
        train_loss += loss
    train_loss /= len(train_loader.dataset)
    #print('Train set: Average loss: {:.8f}, Accuracy: {:>4}/{:<4} ({:>3.0f}%)'.format(
    #    train_loss, correct, len(train_loader.dataset), 100. * correct / len(train_loader.dataset)))
            
    return train_loss, correct/len(train_loader.dataset)

def test(epoch, showParams=False, da1name=None, da2name=None):
    global now
    test_loss = 0.
    correct = 0
    ts_encoder1.eval()
    ts_encoder2.eval()
    classifier.eval()

    with torch.no_grad():
        for da1_data, da2_data, _, _, _, target, _ in test_loader:
            da1_data, da2_data, target = da1_data.cuda(gpu_id).to(dtype=torch.float), da2_data.cuda(gpu_id).to(dtype=torch.float), target.cuda(gpu_id)
            da1_data = da1_data.view(-1, input_channels, seq_length)
            da2_data = da2_data.view(-1, input_channels, seq_length)
            da1_data, da2_data, target = Variable(da1_data), Variable(da2_data), Variable(target)
            
            z1, z2 = ts_encoder1(da1_data), ts_encoder2(da2_data)
            z = torch.cat((z1,z2), 1)
            target_list = target.to('cpu').detach().numpy() if not 'target_list' in locals() else np.concatenate([target_list, target.to('cpu').detach().numpy()])

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
        print(' Epoch:{} Test set: Average loss: {:.8f}, Accuracy: {:>4}/{:<4} ({:>3.0f}%)'.format(
            epoch, test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

        return test_loss, correct / len(test_loader.dataset)


def test_model():
    #model_path = glob.glob('/home/daisuke/Downloads/last_test_accuracy/2en-shared-with-model/*{}*{}*{}*.pth'.format(data_name, da1, da2))
    model_path = glob.glob('/home/daisuke/Documents/CloudStation/lab/Codes/DynamicDA/data/*{}*{}*{}*.pth'.format(data_name, da1, da2))
    #da1name, da2name, = model_path[0].split("_")[2].split("-vs-")[0], model_path[0].split("_")[2].split("-vs-")[1]
    print(model_path)
    model.load_state_dict(torch.load(model_path[0], map_location='cuda:0'))
    test(1, da1name=da1, da2name=da2, report=False) # pseudo epoch
    exit(0)

if __name__ == "__main__":
    #test_model()
    best_loss, best_acc = 10e9, 0.
    for epoch in range(1, epochs+1):
        train_loss, train_acc = train(epoch)
        if epoch%10==0 or epoch==1:
            test_loss, test_acc = test(epoch, showParams=False, da1name=da1, da2name=da2)
        #else:
            #test_loss, test_acc = test(epoch, showParams=False, da1name=da1, da2name=da2)
        
            # save to tsv file
            detached_train_acc = train_acc.to('cpu').detach().numpy().tolist()
            detached_train_loss = train_loss.to('cpu').detach().numpy().tolist()
            detached_test_acc = test_acc.to('cpu').detach().numpy().tolist()
            detached_test_loss = test_loss.to('cpu').detach().numpy().tolist()
            update_gspread_ts2(sheet_name, data_name, 'Matsuo-concat', '{}/{}'.format(dataset_path, data_name), detached_train_acc, detached_train_loss, detached_test_acc, detached_test_loss, epoch, epochs, .5, da1, da2, now, otherparams=None)

        if epoch==epochs: # last epoch
            if not os.path.exists('./data/{}_{}_{}-vs-{}_{}/'.format(sheet_name, data_name, args.da1, args.da2, epoch)):
                os.makedirs('./data/{}_{}_{}-vs-{}_{}/'.format(sheet_name, data_name, args.da1, args.da2, epoch))
            torch.save(ts_encoder1.state_dict(), './data/{}_{}_{}-vs-{}_{}/ts_encoder1.pth'.format(sheet_name, data_name, args.da1, args.da2, epoch))
            torch.save(ts_encoder2.state_dict(), './data/{}_{}_{}-vs-{}_{}/ts_encoder2.pth'.format(sheet_name, data_name, args.da1, args.da2, epoch))
            torch.save(classifier.state_dict(), './data/{}_{}_{}-vs-{}_{}/classifier.pth'.format(sheet_name, data_name, args.da1, args.da2, epoch))
