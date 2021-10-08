import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import *
from PIL import Image

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import precision_recall_fscore_support
global fig_mx
global train_acc, train_loss, test_acc, test_loss, whether_first

train_accs = []
train_losses = []
test_accs = []
test_losses = []
epochs = []
whether_first = 1
data_folder = "./data"

import matplotlib
from mpl_toolkits.axes_grid1 import AxesGrid

# https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib
def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap
    

def draw_graph(epoch_all, epoch_now, train_acc, train_loss, test_acc, test_loss, training_name, save_place='./'):
        global whether_first, fig1, fig2
        train_accs.append(train_acc)
        train_losses.append(train_loss)
        test_accs.append(test_acc)
        test_losses.append(test_loss)
        epochs.append(epoch_now)
        
        if whether_first == 0:
                plt.close(fig1)
                plt.close(fig2)
        
        fig1 = plt.figure()
        plt.ylim(0.45, 1.0)
        plt.plot(epochs, train_accs, 'bo', label='Training acc')
        plt.plot(epochs, test_accs, 'b', label='Test acc')
        plt.title('Training and test accuracy; '+ training_name)
        plt.grid(b=True)
        plt.tick_params(labelsize=15)
        plt.legend(fontsize=15)
        
        fig2 = plt.figure()
        plt.plot(epochs, train_losses, 'ro', label='Training loss')
        plt.plot(epochs, test_losses, 'r', label='Test loss')
        plt.title('Training and test loss; ' + training_name)
        plt.grid(b=True)
        plt.tick_params(labelsize=15)
        plt.legend(fontsize=15)
        
        #plt.pause(0.1)
        
        whether_first = 0
        
        #if epoch_now >= epoch_all-1:
        fig1.savefig('Training and test accuracy; '+ training_name)
        fig2.savefig('Training and test loss; ' + training_name)
        fig1.savefig(save_place + 'Training and test accuracy; '+ training_name)
        fig2.savefig(save_place + 'Training and test loss; ' + training_name)
    
    
# https;//scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py   
def plot_confusion_matrix(y_true, y_pred, classes, save_caption,
                          save_place=None,
                          normalize=False,
                          cmap=plt.cm.Blues
                          ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    global fig_mx
    
    cm = confusion_matrix(y_true, y_pred)
    
    indices = precision_recall_fscore_support(y_true, y_pred, average="macro")
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    accuracy = accuracy_score(y_true, y_pred)
    accuracy_1err_ok = accuracy_score(y_true, y_pred) + accuracy_score(y_true, list(map(lambda x: x+1, y_pred))) + accuracy_score(y_true, list(map(lambda x: x-1, y_pred)))
    accuracy_2err_ok = accuracy_1err_ok + accuracy_score(y_true, list(map(lambda x: x+2, y_pred))) + accuracy_score(y_true, list(map(lambda x: x-2, y_pred)))
    mse = mean_squared_error(y_true, y_pred)
    
    if normalize:    
        title = 'Normalized:accuracy={:.3f},1psn={:.3f},2ppl={:.3f},MSE={:.3f}'.format(accuracy, accuracy_1err_ok, accuracy_2err_ok, mse)
    else:
        title = 'accuracy={:.3f},1psn={:.3f},2ppl={:.3f},MSE={:.3f}'.format(accuracy, accuracy_1err_ok, accuracy_2err_ok, mse)

    fig_mx, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmax=900)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center", fontsize=11,
                    color="white" if cm[i, j] > thresh else "black")
   
    fig_mx.tight_layout()
    fig_mx.savefig(save_place + 'confusion_matrix;' + os.path.basename(save_caption) + '.png')
    fig_mx.savefig('confusion_matrix;' + os.path.basename(save_caption) + '.png')
    plt.close(fig_mx)
    
    acc = 0
    cnt = 0
    for i in range(len(y_true)):
        acc += y_pred[i]
        cnt += 1
     
    result4 = "accuracy: {}\n".format(accuracy)
    result5 = "accuracy (accepting one person error): {}\n".format(accuracy_1err_ok)
    result6 = "accuracy (accepting two people error): {}\n".format(accuracy_2err_ok)
    result7 = "MAE: {}\n".format(mean_absolute_error(y_true, y_pred))
    result8 = "MSE: {}\n".format(mse)
    result9 = "RMSE: {}\n".format(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    f = open(save_place + 'result;' + os.path.basename(save_caption) + '.txt', 'w')
    f2 = open('result;' + os.path.basename(save_caption) + '.txt', 'w')
    f.write(result4)
    f2.write(result4)
    f.write(result5)
    f2.write(result5)
    f.write(result6)
    f2.write(result6)
    f.write(result7)
    f2.write(result7)
    f.write(result8)
    f2.write(result8)
    f.write(result9)
    f2.write(result9)
    f.close()
    f2.close()
    
    return ax

def draw_hist(epoch, params_list, folder_name, now, tag, fourEncoder=False):
    if not os.path.exists('{}/result_hist/{}_{}'.format(data_folder, folder_name, now)):
        os.makedirs('{}/result_hist/{}_{}'.format(data_folder, folder_name, now))
    imgs = []
    fig = plt.figure(figsize=(9, 5))
    bins_list = [i*0.01 for i in range(101)]
    plt.title('{}; {} '.format(folder_name.split('_')[0], tag)+ str(epoch))
    plt.ylim(0, params_list.shape[0])
    plt.yticks(np.arange(0, params_list.shape[0], int(params_list.shape[0]/10)))
    plt.xlim(0.0,1.0)
    plt.hist(params_list[:,0], alpha=0.5, bins=bins_list, label=folder_name.split('_')[2])
    plt.hist(params_list[:,1], alpha=0.5, bins=bins_list, label=folder_name.split('_')[4])
    if fourEncoder:
        plt.hist(params_list[:,2], alpha=0.5, bins=bins_list, label=folder_name.split('_')[6])
        plt.hist(params_list[:,3], alpha=0.5, bins=bins_list, label=folder_name.split('_')[8])
    plt.legend(loc="upper right")
    
    fig.savefig('{}/result_hist/{}_{}/'.format(data_folder, folder_name, now) + tag + str(epoch) + '.png')

if __name__ == '__main__': #this is for draw_graph debug
    epoch = 5
    train_acc4debug = [0.1, 0.2, 0.5, 0.6, 0.9]
    train_loss4debug = [5, 4, 3, 3, 1]
    test_acc4debug = [0.1, 0.3, 0.3, 0.5, 0.6]
    test_loss4debug = [5, 5, 4, 4, 3]

    for i in range(5):
        draw_graph(epoch, i, train_acc4debug[i], train_loss4debug[i], test_acc4debug[i], test_loss4debug[i], "debug")
        #pass

    pixels=[0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9]
    ppl_in_frame(pixels, 20, 25, 5, 'test', save_place=None, caltype="ave")

