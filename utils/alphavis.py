import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.cm as cm
import random; random.seed(42)
import numpy as np; np.random.seed(42)
import pylab, copy, csv, os, glob, gc, time
import torch

import pandas as pd
import seaborn as sns

import itertools
from sklearn.decomposition import PCA

def alpha_hist(savename, p1, p2, p3, p4, p5, target, da1, da2, da3, da4, da5):
    target = [str(i) for i in target]
    p1, p2, p3, p4, p5 = p1.flatten(), p2.flatten(), p3.flatten(), p4.flatten(), p5.flatten()
    data4plot = pd.DataFrame([p1,p2,p3,p4,p5,target], index=[da1,da2,da3,da4,da5,"class"]).T
    sns.pairplot(data4plot, hue="class", palette="tab10", kind="scatter", diag_kind="hist")
    plt.savefig(savename)


#def read_save_png(x_list, target_list, pred_list, idx_list, f_name, epoch, dataset_path, dirname, csvtag):
def read_save_png(target_list, pred_list, idx_list, f_name, epoch, dataset_path, dirname, csvtag, *args):
    cmap = plt.get_cmap("tab10")
    
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        with open(dataset_path+'/{}/{}_{}_Normalized.tsv'.format(f_name, f_name, csvtag)) as f:
            cols = csv.reader(f, delimiter='\t')
            all_cols = []
            for col in cols:
                all_cols.append(col)
                
            fig = plt.figure()
            plt.rcParams["font.size"]=20
            for i in range(len(idx_list)):
                idx = idx_list[i]
                cols = all_cols[idx]
                
                x = list(range(len(cols[1:])))
                col_float = [float(cols[i+1]) for i in range(len(cols[1:]))]
                ori_col_float = copy.copy(col_float)
                #col_float = magnitudeWarp(col_float)
                
                #plt.plot(x, ori_col_float, color="black")
                plt.plot(x, col_float, color=cmap(int(cols[0])-1), linewidth=3)
                
                plt.gca().axes.xaxis.set_visible(False)
                plt.gca().axes.yaxis.set_visible(False)
                plt.ylim(-1.0, 1.0)
                #plt.title(os.path.basename(f_name)+'\nTarget:'+str(target_list[i]+1)+' Pred:'+str(pred_list[i]+1), fontsize=18)
                fig.tight_layout()
                fmt = "pdf"
                if len(args)==1:
                    fig.savefig(dirname+"/{}_{}_{}_{}.{}".format(idx_list[i], args[0][i], target_list[i]+1, pred_list[i]+1, fmt))
                elif len(args)==0:
                    fig.savefig(dirname+"/{}_{}_{}.{}".format(idx_list[i], target_list[i]+1, pred_list[i]+1, fmt))
                elif len(args)==5:
                    fig.savefig(dirname+"/{}_{}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{}_{}.{}".format(-1, idx_list[i], args[0][i][0], args[1][i][0], args[2][i][0], args[3][i][0], args[4][i][0], target_list[i]+1, pred_list[i]+1, fmt))
                plt.cla()
    
    # if len(args)==1:
        # for i in range(len(idx_list)):
            # arr[i,0:480, 0:640,:] = plt.imread(dirname+"/{}_{}_{}_{}.png".format(idx_list[i], args[0][i], target_list[i]+1, pred_list[i]+1))[:,:,:3]
    
    return 0

def get_sample(f_name, num_of_class, dataset_path, csvtag):
    with open(dataset_path+'/{}/{}_{}_Normalized.tsv'.format(f_name, f_name, csvtag)) as f:
        cols = csv.reader(f, delimiter='\t')
        all_cols, class_checker, samples_by_class = [], [], []
        for col in cols:
            if not col[0] in class_checker:
                class_checker.append(col[0])
                samples_by_class.append(col)
                if(len(samples_by_class)==num_of_class):
                    x = list(range(len(col[1:])))
                    return x, samples_by_class

def alpha_number_line(idx_list, x, targets, preds, f_name, da1name, da2name, epoch, acc, dataset_path, consis_lambda, test_bool, by_class=False):
    
    if test_bool:
        dirname = "./alpha_{}_{}-vs-{}_{}/{}-test".format(f_name, da1name, da2name, consis_lambda, epoch)
        csvtag = "TEST"
    else:
        dirname = "./alpha_{}_{}-vs-{}_{}/{}".format(f_name, da1name, da2name, consis_lambda, epoch)
        csvtag = "VALIDATION"
    
    read_save_png(targets, preds, idx_list, f_name, epoch, dataset_path, dirname, csvtag, x)

    cmap = plt.get_cmap("tab10")
    num_of_class = max(targets)+1
    num_of_samples = len(targets)
    x_list = [ [] for i in range(num_of_class) ]
    
    for i in range(len(x)):
        x_list[targets[i]].append(x[i])
    
    fig = plt.figure(figsize=(26, 10), tight_layout=True)
    plt.rcParams["font.size"]=40 # 16
    bins_list = [i*0.01 for i in range(101)]
    #plt.title("{}, {:>5.0f}epoch, {:>5.0f}%(Acc), {:>5.0f}samples result".format(f_name, epoch, acc, num_of_samples))

    plt.axis("off")
    
    # samples for each class (1/2)
    ax_list = []
    #for i in range(num_of_class):
    #    ax_list.append(fig.add_subplot(2,num_of_class,num_of_class+i+1))
    
    # main (1/2)
    if by_class == False:
        yheight = int(1.5*num_of_samples/num_of_class)
        ax_list.append(fig.add_subplot(211, ylim=(0, yheight), yticks=np.arange(int(yheight/10), yheight+1, int(yheight/10)), xlim=(0.0,1.0))) # add grid
        for i in range(num_of_class):
            line = ax_list[-1].hist(x_list[i], bins=bins_list, color=cmap(i), alpha=0.3, label="Class {}".format(i+1))
        #ax_list[-1].grid()
        ax_list[-1].set_ylabel('more {} used'.format(da2name))
        ax_list[-1].set_xlabel('Ratio of Identity')
        ax_reversed = ax_list[-1].twinx()
        ax_reversed.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        ax_reversed.set_ylabel('more {} used'.format(da1name))
    else:
        yheight = int(num_of_samples/(min(40,1*num_of_class))) # 5 for wafer 1 for StarLC
        for i in range(num_of_class):
			#ax_list.append(fig.add_subplot(2,num_of_class,i+1, ylim=(0, yheight), yticks=np.arange(0, yheight, int(yheight/10)), xlim=(0.0,1.0))) # add grid
            #ax_list.append(fig.add_subplot(1,num_of_class,i+1, ylim=(0, yheight), yticks=np.arange(int(yheight/10), yheight+1, int(yheight/10)), xlim=(0.0,1.0))) # add grid
            ax_list.append(fig.add_subplot(1,num_of_class,i+1, ylim=(0, yheight), xlim=(0.0,1.0))) # no grid
            line = ax_list[-1].hist(x_list[i], bins=bins_list, color=cmap(i), label="Class {}".format(i+1))
            ax_list[-1].grid(alpha=0.5)
            ax_list[-1].set_ylabel('more {} used'.format(da2name))
            ax_list[-1].set_xlabel('Ratio of Identity')
            ax_list[-1].spines["right"].set_color("none")
            ax_list[-1].spines["left"].set_color("none")
            ax_list[-1].spines["top"].set_color("none")
            ax_reversed = ax_list[-1].twinx()
            ax_reversed.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
            ax_reversed.set_ylabel('more {} used'.format(da1name))
            ax_reversed.spines["right"].set_color("none")
            ax_reversed.spines["left"].set_color("none")
            ax_reversed.spines["top"].set_color("none")
            #if i!=0:
            ax_list[-1].tick_params(labelleft=False)
            ax_reversed.tick_params(labelleft=False)
            ax_list[-1].tick_params(left=False, right=False)
            ax_reversed.tick_params(left=False, right=False)
            
    
    # samples for each class (2/2)
    # cnt = 0
    # x, samples_by_class = get_sample(f_name, num_of_class, dataset_path, csvtag)
    # samples_by_order = []
    # for i in range(num_of_class):
        # for j in range(num_of_class):
            # if(int(samples_by_class[j][0])==i+1):
                # samples_by_order.append([float(samples_by_class[j][i+1]) for i in range(len(samples_by_class[j][1:]))])
    
    # plot_class_end = -1 if by_class==False else -num_of_class
    # for i in ax_list[:plot_class_end]:
        # i.plot(x, samples_by_order[cnt][:], color=cmap(cnt))
        # i.axes.xaxis.set_visible(False)
        # i.axes.yaxis.set_visible(False)
        # cnt+=1
    
    # main (2/2)
    if by_class == False:
        for i in range(num_of_class):
            ax_list[i].legend(bbox_to_anchor=(0, -0.1), borderaxespad=0, loc="upper left", ncol=1)
        
    ax_list[0].legend(bbox_to_anchor=(0, 1.0), borderaxespad=0, loc="upper left", ncol=1)
    ax_list[1].legend(bbox_to_anchor=(0, 1.0), borderaxespad=0, loc="upper left", ncol=1)
    ax_list[2].legend(bbox_to_anchor=(0, 1.0), borderaxespad=0, loc="upper left", ncol=1)
        
    fig.tight_layout()
    
    if test_bool:
        fig.savefig("./alpha_{}_{}-vs-{}_{}/{}-test.pdf".format(f_name, da1name, da2name, consis_lambda, epoch))
    else:
        fig.savefig("./alpha_{}_{}-vs-{}_{}/{}.pdf".format(f_name, da1name, da2name, consis_lambda, epoch))
    
    plt.clf()
    plt.close()

def alpha_number_line_for_slides(idx_list, x, targets, preds, f_name, da1name, da2name, epoch, acc, dataset_path, test_bool):
    
    if test_bool:
        dirname = "./alpha_{}_{}-vs-{}/{}-test".format(f_name, da1name, da2name, epoch)
        csvtag = "TEST"
    else:
        dirname = "./alpha_{}_{}-vs-{}/{}".format(f_name, da1name, da2name, epoch)
        csvtag = "VALIDATION"
    
    read_save_png(targets, preds, idx_list, f_name, epoch, dataset_path, dirname, csvtag, x)

    cmap = plt.get_cmap("tab10")
    num_of_class = max(targets)+1
    num_of_samples = len(targets)
    x_list = [ [] for i in range(num_of_class) ]
    
    for i in range(len(x)):
        x_list[targets[i]].append(x[i])
    
    fig = plt.figure(figsize=(20, 10), tight_layout=True)
    plt.rcParams["font.size"]=40
    spines = 5
    bins_list = [i*0.01 for i in range(101)]
    ax = plt.gca()
    ax.spines["right"].set_color("none")
    ax.spines["left"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_color("none")
    plt.axis("off")
    
    ax_list = []
    
    yheight = int(1.5*num_of_samples/num_of_class)
    yheight = 30
    ax_list.append(fig.add_subplot(111, ylim=(0, yheight), yticks=np.arange(0, yheight+1, int(yheight/10)), xticks = [0.2,0.4,0.6,0.8], xlim=(0.0,1.0))) # add grid
    for i in range(num_of_class):
        line = ax_list[-1].hist(x_list[i], bins=bins_list, color=cmap(i), alpha=0.3, label="{}".format(i+1))
    ax_list[-1].grid()
    
    ax_list[-1].spines["right"].set_linewidth(spines)
    ax_list[-1].spines["left"].set_linewidth(spines)
    ax_list[-1].spines["top"].set_linewidth(spines)
    ax_list[-1].spines["bottom"].set_linewidth(spines)
    
    fig.tight_layout()
    
    if test_bool:
        fig.savefig("./alpha_{}_{}-vs-{}/{}-test.png".format(f_name, da1name, da2name, epoch), transparent=True)
    else:
        fig.savefig("./alpha_{}_{}-vs-{}/{}.png".format(f_name, da1name, da2name, epoch), transparent=True)
    
    plt.clf()
    plt.close()

def alpha_pca(idx_list, targets, preds, f_name, da1name, p1, da2name, p2, da3name, p3, da4name, p4, da5name, p5, epoch, acc, consis_lambda, dataset_path, test_bool):
    
    csvtag = "TEST" if test_bool else "VALIDATION"
    num_of_class = max(targets)+1
    num_of_samples = len(targets)
    cmap = plt.get_cmap("tab10")
    test = [[p1[i,0],p2[i,0],p3[i,0],p4[i,0],p5[i,0]] for i in range(len(p1))]
    
    pca = PCA(n_components=2)
    pca.fit(test)
    feature = pca.transform(test)
    
    fig = plt.figure(figsize=(16, 10), tight_layout=True)
    plt.rcParams["font.size"]=16
    plt.title("{}, {:>5.0f}epoch, {:>5.0f}%(Acc), {:>5.0f}samples result".format(f_name, epoch, acc, num_of_samples))
    ax = plt.gca()
    ax.spines["right"].set_color("none")
    ax.spines["left"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_color("none")
    plt.axis("off")
        
    # samples for each class (1/2)
    ax_list = []
    for i in range(num_of_class):
        ax_list.append(fig.add_subplot(2,num_of_class,num_of_class+i+1))
    
    # main (1/2)
    ax_list.append(fig.add_subplot(211, xlabel="PC1", ylabel="PC2"))
    for i in range(len(feature[:,1])):
        ax_list[-1].scatter(feature[:, 0], feature[:, 1], alpha=0.8, color=cmap(targets))
    ax_list[-1].grid()
    
    # samples for each class (2/2)
    cnt = 0
    x, samples_by_class = get_sample(f_name, num_of_class, dataset_path, csvtag)
    samples_by_order = []
    for i in range(num_of_class):
        for j in range(num_of_class):
            if(int(samples_by_class[j][0])==i+1):
                samples_by_order.append([float(samples_by_class[j][i+1]) for i in range(len(samples_by_class[j][1:]))])
    
    for i in ax_list[:-1]:
        i.plot(x, samples_by_order[cnt][:], color=cmap(cnt))
        i.axes.xaxis.set_visible(False)
        i.axes.yaxis.set_visible(False)
        cnt+=1
    
    # main (2/2)
    fig.tight_layout()
    
    dirname = "./alpha_{}_{}-{}-{}-{}-{}_{}/".format(f_name, da1name, da2name, da3name, da4name, da5name, consis_lambda)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    
    if test_bool:
        fig.savefig(dirname+"{}-test.png".format(epoch))
    else:
        fig.savefig(dirname+"{}.png".format(epoch))
    
    plt.clf()
    plt.close()


if __name__=="__main__":
    f_name = "ElectricDevices"
    da1 = "identity"
    da2 = "jitter"
    epoch = 1
    acc = 0
    f_list = glob.glob("../alpha/alpha_{}_{}-vs-{}/{}/*.png".format(f_name, da1, da2, epoch))
    
    idx_list, draw_params, draw_target, draw_pred = [], [], [], []
    
    for f in f_list:
        idx = os.path.basename(f).split("_")[0]
        param = os.path.basename(f).split("_")[1].split(".png")[0]
        target = os.path.basename(f).split("_")[2]
        pred = os.path.basename(f).split("_")[3][:-4]
        
        idx_list.append(int(idx))
        draw_params.append(float(param))
        draw_target.append(int(target))
        draw_pred.append(int(pred))
    
    alpha_number_line(idx_list, draw_params, draw_target, draw_pred, f_name, da1, da2, epoch, acc)
    
    numdata = 100
	
    p1 = np.random.rand(numdata)
    p2 = np.random.rand(numdata)
    p3 = np.random.rand(numdata)
    p4 = np.random.rand(numdata)
    p5 = np.random.rand(numdata)
    target_rand = np.random.rand(numdata)
    threshold = np.random.rand()

    target = []

    for i in range(numdata):
        denominator = p1[i]+p2[i]+p3[i]+p4[i]+p5[i]
        p1[i] = p1[i]/denominator
        p2[i] = p2[i]/denominator
        p3[i] = p3[i]/denominator
        p4[i] = p4[i]/denominator
        p5[i] = p5[i]/denominator
        target.append("A") if target_rand[i]>threshold else target.append("B")

    alpha_hist(p1, p2, p3, p4, p5, target, "Identity", "Jittering", "MagnitudeWarp", "WindowWarp", "TimeWarp")
