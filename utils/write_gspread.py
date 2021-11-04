# https://qiita.com/akabei/items/0eac37cb852ad476c6b9
import os, glob
import gspread
import datetime
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials

global clear_last
clear_last = 1
best_acc = 0.0
best_loss = 10e10
data_folder = "./data"

def sheet_existence_check(sheet_name):# とりあえず、存在していればエラー終了すれば良い
    ### open google spreadsheet ###
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    credentials = ServiceAccountCredentials.from_json_keyfile_name('time-series-da-1428f673f468.json', scope)
    gc = gspread.authorize(credentials)
    wk_sheet = gc.open('Time-series-reproduction').worksheet(sheet_name)

def update_gspread_ts1(sheet_name, dataset_name, network_name, dataset_directory, train_acc, train_loss, test_acc, test_loss, current_epoch, all_epoch, da1, now, **kwargs):
            
    global clear_last, best_acc, best_loss
    
    if current_epoch == 1:
        sheet_existence_check(sheet_name)
    
    new_row = [sheet_name, dataset_name, network_name, dataset_directory, train_loss, test_loss, train_acc, test_acc, current_epoch, da1]
    save_path = '{}/result_csv/'.format(data_folder) + sheet_name + '_' + dataset_name + '_' + network_name + '_' + da1 + '_' + now + '_result.csv'
    
    if clear_last == 1: # for initial
        clear_last = 0
        csvcolumns = ['sheet_name', 'dataset_name', 'network_name', 'place', 'train_loss', 'vali/test_loss', 'train_acc', 'vali/test_acc', 'epoch', 'da1']
        
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
        
def update_gspread_ts2(sheet_name, dataset_name, network_name, dataset_directory, train_acc, train_loss, test_acc, test_loss, current_epoch, all_epoch, params, da1, da2, consis_lambda, now, **kwargs):
    global clear_last, best_acc, best_loss
    
    if current_epoch == 1:
        sheet_existence_check(sheet_name)
    
    new_row = [sheet_name, dataset_name, network_name, dataset_directory, train_loss, test_loss, train_acc, test_acc, current_epoch, da1, params, da2, 1.-params, consis_lambda]
    save_path = '{}/result_csv/'.format(data_folder) + sheet_name + '_' + dataset_name + '_' + network_name + '_' + da1 +'_VS_' + da2 + '_' + str(consis_lambda) + '_' + now + '_result.csv'
    
    if clear_last == 1: # for initial
        clear_last = 0
        csvcolumns = ['sheet_name', 'dataset_name', 'network_name', 'place', 'train_loss', 'vali/test_loss', 'train_acc', 'vali/test_acc', 'epoch', 'da1', 'param1', 'da2', 'param2', 'consis_lambda']
        
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

def update_gspread_ts5(sheet_name, dataset_name, network_name, dataset_directory, train_acc, train_loss, test_acc, test_loss, current_epoch, all_epoch, da1, p1, da2, p2, da3, p3, da4, p4, da5, p5, consis_lambda, now, **kwargs):
    global clear_last, best_acc, best_loss
    
    if current_epoch == 1:
        sheet_existence_check(sheet_name)
    
    new_row = [sheet_name, dataset_name, network_name, dataset_directory, train_loss, test_loss, train_acc, test_acc, current_epoch, da1, p1, da2, p2, da3, p3, da4, p4, da5, p5, consis_lambda]
    save_path = '{}/result_csv/'.format(data_folder) + sheet_name + '_' + dataset_name + '_' + network_name + '_' + da1 +'-' + da2 +'-' + da3 +'-' + da4 +'-' + da5 + '_' + str(consis_lambda) + '_' + now + '_result.csv'
    
    if clear_last == 1: # for initial
        clear_last = 0
        csvcolumns = ['sheet_name', 'dataset_name', 'network_name', 'place', 'train_loss', 'vali/test_loss', 'train_acc', 'vali/test_acc', 'epoch', 'da1', 'param1', 'da2', 'param2', 'da3', 'param3', 'da4', 'param4', 'da5', 'param5', 'consis_lambda']
        
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


def get_length_numofclass(dataset_name, dataset_path):
    ### open google spreadsheet ###
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']

    credentials = ServiceAccountCredentials.from_json_keyfile_name('time-series-da-1428f673f468.json', scope)
    gc = gspread.authorize(credentials)
    wk_sheet = gc.open('Time-series-reproduction').worksheet("VGG(oba)") # get length & num-of-class from this sheet!
    
    ds_cell = wk_sheet.find(dataset_name)
    length = wk_sheet.cell(ds_cell.row, 4).value
    NumOfClass = wk_sheet.cell(ds_cell.row, 5).value 
    NumOfTrain = wk_sheet.cell(ds_cell.row, 2).value 
    
    return int(length), int(NumOfClass), int(NumOfTrain)
