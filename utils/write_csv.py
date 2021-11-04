# https://qiita.com/akabei/items/0eac37cb852ad476c6b9
import os, glob
import datetime
import pandas as pd

global clear_last
clear_last = 1
best_acc = 0.0
best_loss = 10e10
data_folder = "./result"


def update_csv_ts1(dataset_name, network_name, dataset_directory, train_acc, train_loss, test_acc, test_loss, current_epoch, all_epoch, da1, now, **kwargs):       
    global clear_last, best_acc, best_loss
    
    new_row = [dataset_name, network_name, dataset_directory, train_loss, test_loss, train_acc, test_acc, current_epoch, da1]
    save_path = '{}/result_csv/'.format(data_folder) + dataset_name + '_' + network_name + '_' + da1 + '_' + now + '_result.csv'
    
    if clear_last == 1: # for initial
        clear_last = 0
        csvcolumns = ['dataset_name', 'network_name', 'place', 'train_loss', 'vali/test_loss', 'train_acc', 'vali/test_acc', 'epoch', 'da1']
        
        df = pd.DataFrame(columns=csvcolumns)   
        if not os.path.exists('{}/result_csv/'.format(data_folder)):
            os.makedirs('{}/result_csv/'.format(data_folder))
        df.to_csv(save_path)
        
    df = pd.read_csv(save_path, index_col=0)
    series = pd.Series(new_row, index=df.columns)
    df = df.append(series, ignore_index=True)
    df.to_csv(save_path)
    
    if test_acc>best_acc:
        best_acc = test_acc
    
    if test_loss<best_loss:
        best_loss = test_loss

def update_csv_ts5(dataset_name, network_name, dataset_directory, train_acc, train_loss, test_acc, test_loss, current_epoch, all_epoch, da1, p1, da2, p2, da3, p3, da4, p4, da5, p5, consis_lambda, now, **kwargs):
    global clear_last, best_acc, best_loss
    
    new_row = [dataset_name, network_name, dataset_directory, train_loss, test_loss, train_acc, test_acc, current_epoch, da1, p1, da2, p2, da3, p3, da4, p4, da5, p5, consis_lambda]
    save_path = '{}/result_csv/'.format(data_folder) + dataset_name + '_' + network_name + '_' + da1 +'-' + da2 +'-' + da3 +'-' + da4 +'-' + da5 + '_' + str(consis_lambda) + '_' + now + '_result.csv'
    
    if clear_last == 1: # for initial
        clear_last = 0
        csvcolumns = ['dataset_name', 'network_name', 'place', 'train_loss', 'vali/test_loss', 'train_acc', 'vali/test_acc', 'epoch', 'da1', 'param1', 'da2', 'param2', 'da3', 'param3', 'da4', 'param4', 'da5', 'param5', 'consis_lambda']
        
        df = pd.DataFrame(columns=csvcolumns)   
        if not os.path.exists('{}/result_csv/'.format(data_folder)):
            os.makedirs('{}/result_csv/'.format(data_folder))
        df.to_csv(save_path)
        
    df = pd.read_csv(save_path, index_col=0)
    series = pd.Series(new_row, index=df.columns)
    df = df.append(series, ignore_index=True)
    df.to_csv(save_path)
    
    if test_acc>best_acc:
        best_acc = test_acc
    
    if test_loss<best_loss:
        best_loss = test_loss

def get_length_numofclass(dataset_name):
	
	if dataset_name == 'Crop':
		length, NumOfClass, NumOfTrain = 46, 24, 7200
	elif dataset_name == 'ElectricDevices':
		length, NumOfClass, NumOfTrain = 95, 7, 8926
	elif dataset_name == 'FordA':
		length, NumOfClass, NumOfTrain = 500, 2, 3601
	elif dataset_name == 'FordB':
		length, NumOfClass, NumOfTrain = 500, 2, 3636
	elif dataset_name == 'HandOutlines':
		length, NumOfClass, NumOfTrain = 2709, 2, 1000
	elif dataset_name == 'MelbournePedestrian_adjusted':
		length, NumOfClass, NumOfTrain = 24, 10, 1194
	elif dataset_name == 'NonInvasiveFetalECGThorax1':
		length, NumOfClass, NumOfTrain = 750, 42, 1800
	elif dataset_name == 'NonInvasiveFetalECGThorax2':
		length, NumOfClass, NumOfTrain = 750, 42, 1800
	elif dataset_name == 'PhalangesOutlinesCorrect':
		length, NumOfClass, NumOfTrain = 80, 2, 1800
	elif dataset_name == 'StarLightCurves':
		length, NumOfClass, NumOfTrain = 1024, 3, 1000
	elif dataset_name == 'TwoPatterns':
		length, NumOfClass, NumOfTrain = 128, 4, 1000
	elif dataset_name == 'Wafer':
		length, NumOfClass, NumOfTrain = 152, 2, 1000
	
	return length, NumOfClass, NumOfTrain
