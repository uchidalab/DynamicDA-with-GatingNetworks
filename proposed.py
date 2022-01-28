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
from utils.visualize import *

parser = argparse.ArgumentParser(description='Sequence Modeling')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
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
parser.add_argument('--gpu_id', type=int, default=0,
                    help='set gpu_id')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--frozen', action='store_true',
                    help='use a frozen model to test')
parser.add_argument('--limit_num', type=int, default=300,
                    help='max vizualized samples')
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
train_loader, test_loader = data_generator(data_name, batch_size, da1, da2, da3, da4, da5, dataset_path)

ts_encoder1, ts_encoder2 = TSEncoder(data_len_after_cnn, z_size), TSEncoder(data_len_after_cnn, z_size)
ts_encoder3, ts_encoder4 = TSEncoder(data_len_after_cnn, z_size), TSEncoder(data_len_after_cnn, z_size)
ts_encoder5 = TSEncoder(data_len_after_cnn, z_size)
gating5 = Gating5(data_len_after_cnn, z_size)
classifier = Classifier(n_classes, z_size)

ts_encoder1.cuda(gpu_id)
ts_encoder2.cuda(gpu_id)
ts_encoder3.cuda(gpu_id)
ts_encoder4.cuda(gpu_id)
ts_encoder5.cuda(gpu_id)
gating5.cuda(gpu_id)
classifier.cuda(gpu_id)
    
lr = args.lr
optimizer = optim.Adam(
    [{'params': ts_encoder1.parameters()},
    {'params': ts_encoder2.parameters()},
    {'params': ts_encoder3.parameters()},
    {'params': ts_encoder4.parameters()},
    {'params': ts_encoder5.parameters()},
    {'params': gating5.parameters()}, 
    {'params': classifier.parameters()},
    ], lr=lr, betas=(0.5, 0.999))

MSE_loss, CE_loss = nn.MSELoss(), nn.CrossEntropyLoss()

def train(ep):
    global steps, now
    train_loss = 0.
    params_mean1 = params_mean2 = params_mean3 = params_mean4 = params_mean5 = 0.
    correct = 0
    ts_encoder1.train()
    ts_encoder2.train()
    ts_encoder3.train()
    ts_encoder4.train()
    ts_encoder5.train()
    gating5.train()
    classifier.train()
    
    for da1_data, da2_data, da3_data, da4_data, da5_data, target, _ in train_loader:
        da1_data, da2_data, target = da1_data.cuda(gpu_id).to(dtype=torch.float), da2_data.cuda(gpu_id).to(dtype=torch.float), target.cuda(gpu_id)
        da3_data, da4_data, da5_data = da3_data.cuda(gpu_id).to(dtype=torch.float), da4_data.cuda(gpu_id).to(dtype=torch.float), da5_data.cuda(gpu_id).to(dtype=torch.float)
        
        da1_data, da2_data = da1_data.view(-1, input_channels, seq_length), da2_data.view(-1, input_channels, seq_length)
        da3_data, da4_data, da5_data = da3_data.view(-1, input_channels, seq_length), da4_data.view(-1, input_channels, seq_length), da5_data.view(-1, input_channels, seq_length)
        
        da1_data, da2_data, target = Variable(da1_data), Variable(da2_data), Variable(target)
        da3_data, da4_data, da5_data = Variable(da3_data), Variable(da4_data), Variable(da5_data)
        
        z1, z2, z3, z4, z5 = ts_encoder1(da1_data), ts_encoder2(da2_data), ts_encoder3(da3_data), ts_encoder4(da4_data), ts_encoder5(da5_data)
        alpha = gating5(torch.cat([da1_data, da2_data, da3_data, da4_data, da5_data], dim=1))
        alpha = alpha.view(-1, 5)
        alpha1, alpha2, alpha3, alpha4, alpha5 = alpha[:,0].view(-1,1), alpha[:,1].view(-1,1), alpha[:,2].view(-1,1), alpha[:,3].view(-1,1), alpha[:,4].view(-1,1)
        
        params_list1 = alpha1.to('cpu').detach().numpy() if not 'params_list1' in locals() else np.concatenate([params_list1, alpha1.to('cpu').detach().numpy()])
        params_list2 = alpha2.to('cpu').detach().numpy() if not 'params_list2' in locals() else np.concatenate([params_list2, alpha2.to('cpu').detach().numpy()])
        params_list3 = alpha3.to('cpu').detach().numpy() if not 'params_list3' in locals() else np.concatenate([params_list3, alpha3.to('cpu').detach().numpy()])
        params_list4 = alpha4.to('cpu').detach().numpy() if not 'params_list4' in locals() else np.concatenate([params_list4, alpha4.to('cpu').detach().numpy()])
        params_list5 = alpha5.to('cpu').detach().numpy() if not 'params_list5' in locals() else np.concatenate([params_list5, alpha5.to('cpu').detach().numpy()])
        params_mean1 += torch.sum(alpha1, 0)
        params_mean2 += torch.sum(alpha2, 0)
        params_mean3 += torch.sum(alpha3, 0)
        params_mean4 += torch.sum(alpha4, 0)
        params_mean5 += torch.sum(alpha5, 0)
        
        alpha1, alpha2, alpha3, alpha4, alpha5 = alpha1.expand(-1, z_size), alpha2.expand(-1, z_size), alpha3.expand(-1, z_size), alpha4.expand(-1, z_size), alpha5.expand(-1, z_size)
        z = alpha1*z1 + alpha2*z2 + alpha3*z3 + alpha4*z4 + alpha5*z5
        y = classifier(z)
        
        loss = CE_loss(y, target)
        z_mean = (z1+z2+z3+z4+z5)/5            
        consistency_loss = (MSE_loss(z_mean, z1)+MSE_loss(z_mean, z2)+MSE_loss(z_mean, z3)+MSE_loss(z_mean, z4)+MSE_loss(z_mean, z5))/5
        loss = loss + args.consis_lambda*consistency_loss
        
        optimizer.zero_grad()
        pred = y.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss.backward()
        optimizer.step()
        
        train_loss += loss
    
    train_loss /= len(train_loader.dataset)
    params_mean1 /= len(train_loader.dataset)
    params_mean2 /= len(train_loader.dataset)
    params_mean3 /= len(train_loader.dataset)
    params_mean4 /= len(train_loader.dataset)
    params_mean5 /= len(train_loader.dataset)
    #print('     Train set: Average loss: {:.8f}, Accuracy: {:>4}/{:<4} ({:>3.1f}%) Average Params: {}|{:.4f}, {}|{:.4f}, {}|{:.4f}, {}|{:.4f}, {}|{:.4f}'.format(
    #    train_loss, correct, len(train_loader.dataset), 100.*correct / len(train_loader.dataset), da1, params_mean1[0], da2, params_mean2[0], da3, params_mean3[0], da4, params_mean4[0], da5, params_mean5[0]))
            
    return train_loss, correct/len(train_loader.dataset)

def test(epoch, test_model=False):
    global now
    test_loss = 0.
    params_mean1 = params_mean2 = params_mean3 = params_mean4 = params_mean5 = 0.
    correct = 0
    ts_encoder1.eval()
    ts_encoder2.eval()
    ts_encoder3.eval()
    ts_encoder4.eval()
    ts_encoder5.eval()
    gating5.eval()
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
            
            alpha = gating5(torch.cat([da1_data, da2_data, da3_data, da4_data, da5_data], dim=1))
            alpha = alpha.view(-1, 5)
            alpha1, alpha2, alpha3, alpha4, alpha5 = alpha[:,0].view(-1,1), alpha[:,1].view(-1,1), alpha[:,2].view(-1,1), alpha[:,3].view(-1,1), alpha[:,4].view(-1,1)
            
            target_list = target.to('cpu').detach().numpy() if not 'target_list' in locals() else np.concatenate([target_list, target.to('cpu').detach().numpy()])            
            params_list1 = alpha1.to('cpu').detach().numpy() if not 'params_list1' in locals() else np.concatenate([params_list1, alpha1.to('cpu').detach().numpy()])
            params_list2 = alpha2.to('cpu').detach().numpy() if not 'params_list2' in locals() else np.concatenate([params_list2, alpha2.to('cpu').detach().numpy()])
            params_list3 = alpha3.to('cpu').detach().numpy() if not 'params_list3' in locals() else np.concatenate([params_list3, alpha3.to('cpu').detach().numpy()])
            params_list4 = alpha4.to('cpu').detach().numpy() if not 'params_list4' in locals() else np.concatenate([params_list4, alpha4.to('cpu').detach().numpy()])
            params_list5 = alpha5.to('cpu').detach().numpy() if not 'params_list5' in locals() else np.concatenate([params_list5, alpha5.to('cpu').detach().numpy()])
            params_mean1 += torch.sum(alpha1, 0)
            params_mean2 += torch.sum(alpha2, 0)
            params_mean3 += torch.sum(alpha3, 0)
            params_mean4 += torch.sum(alpha4, 0)
            params_mean5 += torch.sum(alpha5, 0)

            alpha1, alpha2, alpha3, alpha4, alpha5 = alpha1.expand(-1, z_size), alpha2.expand(-1, z_size), alpha3.expand(-1, z_size), alpha4.expand(-1, z_size), alpha5.expand(-1, z_size)
            z = alpha1*z1 + alpha2*z2 + alpha3*z3 + alpha4*z4 + alpha5*z5
            y = classifier(z)
            
            z_list = z if not 'z_list' in locals() else torch.cat([z_list, z])
            
            _, predict = torch.max(y.data, 1)
            test_correct = (predict == target).sum().item()

            loss = CE_loss(y, target)
            z_mean = (z1+z2+z3+z4+z5)/5            
            consistency_loss = (MSE_loss(z_mean, z1)+MSE_loss(z_mean, z2)+MSE_loss(z_mean, z3)+MSE_loss(z_mean, z4)+MSE_loss(z_mean, z5))/5
            loss = loss + args.consis_lambda*consistency_loss
            
            pred = y.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            
            pred_list = pred.to('cpu').detach().numpy() if not 'pred_list' in locals() else np.concatenate([pred_list, pred.to('cpu').detach().numpy()])
            test_loss += loss

        pred_list = np.array([item for l in pred_list for item in l ])
        test_loss /= len(test_loader.dataset)
        params_mean1 /= len(test_loader.dataset)
        params_mean2 /= len(test_loader.dataset)
        params_mean3 /= len(test_loader.dataset)
        params_mean4 /= len(test_loader.dataset)
        params_mean5 /= len(test_loader.dataset)
        
        # only when using saved model
        if test_model == True:
            random.seed(42) # must
            idx_list = np.arange(len(target_list))
            if len(target_list)>limit_num:
                idx_list = random.sample(list(idx_list), limit_num)
                draw_p1, draw_p2, draw_p3, draw_p4, draw_z = itemgetter(idx_list)(params_list1[:]), itemgetter(idx_list)(params_list2[:]), itemgetter(idx_list)(params_list3[:]), itemgetter(idx_list)(params_list4[:]), itemgetter(idx_list)(z_list[:])
                draw_p5, draw_target, draw_pred, draw_z1 = itemgetter(idx_list)(params_list5[:]), itemgetter(idx_list)(target_list), itemgetter(idx_list)(pred_list), itemgetter(idx_list)(z1_list[:])
                draw_z2, draw_z3, draw_z4, draw_z5 = itemgetter(idx_list)(z2_list[:]), itemgetter(idx_list)(z3_list[:]), itemgetter(idx_list)(z4_list[:]), itemgetter(idx_list)(z5_list[:])
            else:
                draw_p1, draw_p2, draw_p3, draw_p4 = params_list1[:], params_list2[:], params_list3[:], params_list4[:]
                draw_p5, draw_target, draw_pred, draw_z = params_list5[:], target_list, pred_list, z_list[:]
                draw_z1, draw_z2, draw_z3, draw_z4, draw_z5 = z1_list[:], z2_list[:], z3_list[:], z4_list[:], z5_list[:]
        
            read_save_png(draw_target, draw_pred, idx_list, data_name, epoch, dataset_path, "./{}/Sample_visualize_{}".format(data_name, data_name), "TEST", draw_p1, draw_p2, draw_p3, draw_p4, draw_p5)
            feature_by_class("TSNE", data_name, draw_z, draw_target, epoch, 100*correct/len(test_loader.dataset), "./{}".format(data_name), args.consis_lambda)
            alpha_number_line(idx_list, draw_p1.flatten(), draw_target, draw_pred, data_name, 'identity', 'others', epoch, 100. * correct / len(test_loader.dataset), dataset_path, args.consis_lambda, True, by_class=True)
    
    
        print('      Test set: Average loss: {:.8f}, Accuracy: {:>4}/{:<4} ({:>3.1f}%) Average Params: {}|{:.4f}, {}|{:.4f}, {}|{:.4f}, {}|{:.4f}, {}|{:.4f}'.format(
            test_loss, correct, len(test_loader.dataset), 100.*correct / len(test_loader.dataset), da1, params_mean1[0], da2, params_mean2[0], da3, params_mean3[0], da4, params_mean4[0], da5, params_mean5[0]))
        
        return test_loss, correct / len(test_loader.dataset), params_mean1[0], params_mean2[0], params_mean3[0], params_mean4[0], params_mean5[0]

def test_model():
    base_path = './result/Proposed_{}_{}-{}-{}-{}-{}_{}_{}'.format(data_name, args.da1, args.da2, args.da3, args.da4, args.da5, args.consis_lambda, epochs)
    ts_encoder1.load_state_dict(torch.load(base_path+'/ts_encoder1.pth', map_location='cuda:0'))
    ts_encoder2.load_state_dict(torch.load(base_path+'/ts_encoder2.pth', map_location='cuda:0'))
    ts_encoder3.load_state_dict(torch.load(base_path+'/ts_encoder3.pth', map_location='cuda:0'))
    ts_encoder4.load_state_dict(torch.load(base_path+'/ts_encoder4.pth', map_location='cuda:0'))
    ts_encoder5.load_state_dict(torch.load(base_path+'/ts_encoder5.pth', map_location='cuda:0'))
    gating5.load_state_dict(torch.load(base_path+'/gating5.pth', map_location='cuda:0'))
    classifier.load_state_dict(torch.load(base_path+'/classifier.pth', map_location='cuda:0'))
    test(epochs, True)
    exit(0)

if __name__ == "__main__":
    if args.frozen == True:
        test_model()
    
    best_loss, best_acc = 10e5, 0.
    for epoch in range(1, epochs+1):
        print('Epoch:{}/{}'.format(epoch, epochs))
        train_loss, train_acc = train(epoch)
        if epoch%25==0 or epoch==epochs or epoch==1:
            test_loss, test_acc, p1, p2, p3, p4, p5 = test(epoch)
        
            # save to csv file
            detached_train_acc, detached_train_loss = train_acc.to('cpu').detach().numpy().tolist(), train_loss.to('cpu').detach().numpy().tolist()
            detached_test_acc, detached_test_loss = test_acc.to('cpu').detach().numpy().tolist(), test_loss.to('cpu').detach().numpy().tolist()
            detached_p1, detached_p2, detached_p3, detached_p4, detached_p5 = p1.to('cpu').detach().numpy().tolist(), p2.to('cpu').detach().numpy().tolist(), p3.to('cpu').detach().numpy().tolist(), p4.to('cpu').detach().numpy().tolist(), p5.to('cpu').detach().numpy().tolist()
            update_csv_ts5(data_name, 'Proposed', detached_train_acc, detached_train_loss, detached_test_acc, detached_test_loss, epoch, \
                           da1, detached_p1, da2, detached_p2, da3, detached_p3, da4, detached_p4, da5, detached_p5, args.consis_lambda, now)
            
            if epoch==epochs:
                model_save_path = './result/Proposed_{}_{}-{}-{}-{}-{}_{}_{}/'.format(data_name, args.da1, args.da2, args.da3, args.da4, args.da5, args.consis_lambda, epoch)
                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)
                torch.save(ts_encoder1.state_dict(), model_save_path+'ts_encoder1.pth')
                torch.save(ts_encoder2.state_dict(), model_save_path+'ts_encoder2.pth')
                torch.save(ts_encoder3.state_dict(), model_save_path+'ts_encoder3.pth')
                torch.save(ts_encoder4.state_dict(), model_save_path+'ts_encoder4.pth')
                torch.save(ts_encoder5.state_dict(), model_save_path+'ts_encoder5.pth')
                torch.save(gating5.state_dict(), model_save_path+'gating5.pth')
                torch.save(classifier.state_dict(), model_save_path+'classifier.pth')
            
