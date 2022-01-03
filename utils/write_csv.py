import os, glob
import datetime
import pandas as pd

clear_last = 1

def update_csv_ts1(dataset_name, network_name, train_acc, train_loss, test_acc, test_loss, current_epoch, da1, now):       
    global clear_last
    
    new_row = [dataset_name, network_name, train_loss, test_loss, train_acc, test_acc, current_epoch, da1]
    save_path = './result/result_csv/' + dataset_name + '_' + network_name + '_' + da1 + '_' + now + '_result.csv'
    
    if clear_last == 1: # for initial
        clear_last = 0
        csvcolumns = ['dataset_name', 'network_name', 'train_loss', 'test_loss', 'train_acc', 'test_acc', 'epoch', 'da1']
        
        df = pd.DataFrame(columns=csvcolumns)   
        if not os.path.exists('./result/result_csv/'):
            os.makedirs('./result/result_csv/')
        df.to_csv(save_path)
        
    df = pd.read_csv(save_path, index_col=0)
    series = pd.Series(new_row, index=df.columns)
    df = df.append(series, ignore_index=True)
    df.to_csv(save_path)
    

def update_csv_ts5(dataset_name, network_name, train_acc, train_loss, test_acc, test_loss, current_epoch, da1, p1, da2, p2, da3, p3, da4, p4, da5, p5, consis_lambda, now):
    global clear_last
    
    new_row = [dataset_name, network_name, train_loss, test_loss, train_acc, test_acc, current_epoch, da1, p1, da2, p2, da3, p3, da4, p4, da5, p5, consis_lambda]
    save_path = './result/result_csv/' + dataset_name + '_' + network_name + '_' + da1 +'-' + da2 +'-' + da3 +'-' + da4 +'-' + da5 + '_' + str(consis_lambda) + '_' + now + '_result.csv'
    
    if clear_last == 1: # for initial
        clear_last = 0
        csvcolumns = ['dataset_name', 'network_name', 'train_loss', 'test_loss', 'train_acc', 'test_acc', 'epoch', 'da1', 'param1', 'da2', 'param2', 'da3', 'param3', 'da4', 'param4', 'da5', 'param5', 'consis_lambda']
        
        df = pd.DataFrame(columns=csvcolumns)   
        if not os.path.exists('./result/result_csv/'):
            os.makedirs('./result/result_csv/')
        df.to_csv(save_path)
        
    df = pd.read_csv(save_path, index_col=0)
    series = pd.Series(new_row, index=df.columns)
    df = df.append(series, ignore_index=True)
    df.to_csv(save_path)
    

def get_length_numofclass(dataset_name):
	
	if dataset_name == 'Crop':
		length, NumOfClass, NumOfTrain = 46, 24, 7200
	elif dataset_name == 'ElectricDevices':
		length, NumOfClass, NumOfTrain = 96, 7, 8926
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
    # 1000-500 samples
	elif dataset_name == 'UWaveGestureLibraryAll':
		length, NumOfClass, NumOfTrain = 945, 8, 896
	elif dataset_name == 'UWaveGestureLibraryX':
		length, NumOfClass, NumOfTrain = 315, 8, 896
	elif dataset_name == 'UWaveGestureLibraryY':
		length, NumOfClass, NumOfTrain = 315, 8, 896
	elif dataset_name == 'UWaveGestureLibraryZ':
		length, NumOfClass, NumOfTrain = 315, 8, 896
	elif dataset_name == 'Strawberry':
		length, NumOfClass, NumOfTrain = 235, 2, 613
	elif dataset_name == 'ProximalPhalanxOutlineCorrect':
		length, NumOfClass, NumOfTrain = 80, 2, 600
	elif dataset_name == 'MiddlePhalanxOutlineCorrect':
		length, NumOfClass, NumOfTrain = 80, 2, 600
	elif dataset_name == 'DistalPhalanxOutlineCorrect':
		length, NumOfClass, NumOfTrain = 80, 2, 600
	elif dataset_name == 'ShapesAll':
		length, NumOfClass, NumOfTrain = 512, 60, 600
	elif dataset_name == 'FaceAll':
		length, NumOfClass, NumOfTrain = 131, 14, 560
	elif dataset_name == 'PLAID_adjusted':
		length, NumOfClass, NumOfTrain = 1344, 11, 537
	elif dataset_name == 'EthanolLevel':
		length, NumOfClass, NumOfTrain = 1751, 4, 504
	elif dataset_name == 'SwedishLeaf':
		length, NumOfClass, NumOfTrain = 128, 15, 500
	elif dataset_name == 'ECG5000':
		length, NumOfClass, NumOfTrain = 140, 5, 500
	elif dataset_name == 'MixedShapesRegularTrain':
		length, NumOfClass, NumOfTrain = 1024, 5, 500
	
	return length, NumOfClass, NumOfTrain
