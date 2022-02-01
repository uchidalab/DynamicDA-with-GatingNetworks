import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.cm as cm
import random; random.seed(42)
import numpy as np; np.random.seed(42)
import pylab, copy, csv, os, glob, gc, time
import torch

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

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

                plt.plot(x, col_float, color=cmap(int(cols[0])-1), linewidth=3)
                
                plt.gca().axes.xaxis.set_visible(False)
                plt.gca().axes.yaxis.set_visible(False)
                plt.ylim(-1.0, 1.0)
                fig.tight_layout()
                fmt = "pdf"
                if len(args)==1:
                    fig.savefig(dirname+"/{}_{}_{}_{}.{}".format(idx_list[i], args[0][i], target_list[i]+1, pred_list[i]+1, fmt))
                elif len(args)==0:
                    fig.savefig(dirname+"/{}_{}_{}.{}".format(idx_list[i], target_list[i]+1, pred_list[i]+1, fmt))
                elif len(args)==5: # -1, index, DA1, DA2, DA3, DA4, DA5, target, pred, format
                    fig.savefig(dirname+"/{}_{}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{}_{}.{}".format(-1, idx_list[i], args[0][i][0], args[1][i][0], args[2][i][0], args[3][i][0], args[4][i][0], target_list[i]+1, pred_list[i]+1, fmt))
                    
                plt.cla()

            if len(args)==5: # save alphas in csv
                args = np.vstack([np.array(args[0][:][:]).flatten(), np.array(args[1][:][:]).flatten(), np.array(args[2][:][:]).flatten() ,np.array(args[3][:][:]).flatten() ,np.array(args[4][:][:]).flatten(), target_list, pred_list, idx_list])
                args = zip(*args)
                args = pd.DataFrame(args, dtype='float')
                args.to_csv(os.path.dirname(dirname)+"/{}.csv".format(f_name))
    
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
    dirname = "./{}/alpha_{}_{}-vs-{}_{}/".format(f_name, f_name, da1name, da2name, consis_lambda)
    dirname += "{}-test".format(epoch) if test_bool else "/{}".format(epoch)
    csvtag = "TEST" if test_bool else "VALIDATION"
    
    read_save_png(targets, preds, idx_list, f_name, epoch, dataset_path, dirname, csvtag, x)

    cmap = plt.get_cmap("tab10")
    num_of_class, num_of_samples = max(targets)+1, len(targets)
    x_list = [ [] for i in range(num_of_class) ]
    
    for i in range(len(x)):
        x_list[targets[i]].append(x[i])
    
    fig = plt.figure(figsize=(26, 10), tight_layout=True)
    plt.rcParams["font.size"]=40
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
        ax_list[-1].set_xlabel('Ratio of {}'.format(da1name))
        ax_reversed = ax_list[-1].twinx()
        ax_reversed.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        ax_reversed.set_ylabel('more {} used'.format(da1name))
    else:
        yheight = int(num_of_samples/(min(40,9*num_of_class))) # 5 for wafer 1 for StarLC
        for i in range(num_of_class):
			#ax_list.append(fig.add_subplot(2,num_of_class,i+1, ylim=(0, yheight), yticks=np.arange(0, yheight, int(yheight/10)), xlim=(0.0,1.0))) # add grid
            #ax_list.append(fig.add_subplot(1,num_of_class,i+1, ylim=(0, yheight), yticks=np.arange(int(yheight/10), yheight+1, int(yheight/10)), xlim=(0.0,1.0))) # add grid
            ax_list.append(fig.add_subplot(1,num_of_class,i+1, ylim=(0, yheight), xlim=(0.0,1.0))) # no grid
            
            line = ax_list[-1].hist(x_list[i], bins=bins_list, color=cmap(i), label="Class {}".format(i+1))
            ax_list[-1].grid(alpha=0.5)
            ax_list[-1].set_ylabel('more {} used'.format(da2name))
            ax_list[-1].set_xlabel('Ratio of {}'.format(da1name))
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
    else:
        for i in range(num_of_class):
            ax_list[i].legend(bbox_to_anchor=(0, 1.0), borderaxespad=0, loc="upper left", ncol=1)
        
    fig.tight_layout()
    
    if test_bool:
        fig.savefig("./{}/alpha_{}_{}-vs-{}_{}/{}-test.pdf".format(f_name, f_name, da1name, da2name, consis_lambda, epoch))
    else:
        fig.savefig("./{}/alpha_{}_{}-vs-{}_{}/{}.pdf".format(f_name, f_name, da1name, da2name, consis_lambda, epoch))
    
    plt.clf()
    plt.close()

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.cm as cm
import random; random.seed(42)
import numpy as np; np.random.seed(42)
import pylab, copy, csv, os, glob, gc, time
import torch

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

import itertools
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def feature_by_class(vis_tag, f_name, z, target, epoch, acc, feature_img_path, consis_lambda):
    cmap = plt.get_cmap("tab10") # "terrain" & set num_of_class
    z = z.to('cpu').detach().numpy()

    if vis_tag == "PCA":
        vis = PCA(n_components=2)
        vis.fit(z) # test must be "num_of_samples(=5DAx64samples) X dimension"
        feature = vis.transform(z)
    else:
        vis = TSNE(n_components=2)
        feature = vis.fit_transform(z)
    
    fig = plt.figure(figsize=(16, 10), tight_layout=True)
    plt.rcParams["font.size"]=40 # 16
    #plt.title("{}, {:>5.0f}epoch, {:>5.0f}%(Acc), {:>5.0f}samples result, lambda={}".format(f_name, epoch, acc, z.shape[0], consis_lambda))
    ax = plt.gca()
    ax.spines["right"].set_color("none")
    ax.spines["left"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_color("none")
    plt.axis("off")
    
    # main
    class_count = []
    ax_list = []
    if vis_tag == "PCA":
        ax_list.append(fig.add_subplot(111, xlabel="PC1", ylabel="PC2"))
    else:
        ax_list.append(fig.add_subplot(111))
    for i in range(len(feature[:,1])):
        if target[i] not in class_count:
            ax_list[-1].scatter(feature[i, 0], feature[i, 1], alpha=0.75, color=cmap(target[i]), label=str(target[i]+1))
            class_count.append(target[i])
        else:
            ax_list[-1].scatter(feature[i, 0], feature[i, 1], alpha=0.75, color=cmap(target[i]))
    ax_list[-1].grid()
    ax_list[-1].set_ylim([-70,70])
    ax_list[-1].set_xlim([-60,60])
    ax_list[-1].axes.xaxis.set_visible(False)
    ax_list[-1].axes.yaxis.set_visible(False)
    #ax_list[-1].legend(bbox_to_anchor=(0, -0.1), borderaxespad=0, loc="upper left", ncol=max(target)+1)
    fig.tight_layout()

    fig.savefig(feature_img_path+"/by-class_{}_{}_{}_{:.0f}_{}.pdf".format(vis_tag, f_name, epoch, acc, consis_lambda))
    
    plt.clf()
    plt.close()
