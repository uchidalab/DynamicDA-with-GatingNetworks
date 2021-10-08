import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.cm as cm
import random; random.seed(42)
import numpy as np; np.random.seed(42)
import pylab, copy, csv, os, glob, gc, time
import torch

import itertools
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def feature_pca(vis_tag, f_name, da1name, z1, mean1, da2name, z2, mean2, da3name, z3, mean3, da4name, z4, mean4, da5name, z5, mean5, epoch, acc, feature_img_path, consis_lambda):
    cmap = plt.get_cmap("tab10")

    z_stack = torch.cat([z1,z2,z3,z4,z5])
    z_stack = z_stack.to('cpu').detach().numpy()
    label_list = [da1name, da2name, da3name, da4name, da5name]
    
    if vis_tag == "PCA":
        vis = PCA(n_components=2)
        vis.fit(z_stack) # test must be "num_of_samples(=5DAx64samples) X dimension"
        feature = vis.transform(z_stack)
    else:
        vis = TSNE(n_components=2)
        feature = vis.fit_transform(z_stack)

    
    fig = plt.figure(figsize=(16*2,10*2), tight_layout=True) # figsize=(16,10)
    plt.rcParams["font.size"]=40 # 16
    #plt.title("{}, {:>5.0f}epoch, {:>5.0f}%(Acc), {:>5.0f}samples result, lambda={}\n{}|{:.4f}, {}|{:.4f}, {}|{:.4f}, {}|{:.4f}, {}|{:.4f}".format(f_name, epoch, acc, z1.shape[0], consis_lambda, da1name, mean1, da2name, mean2, da3name, mean3, da4name, mean4, da5name, mean5))
    ax = plt.gca()
    ax.spines["right"].set_color("none")
    ax.spines["left"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_color("none")
    plt.axis("off")
    
    # main
    ax_list = []
    if vis_tag == "PCA":
        ax_list.append(fig.add_subplot(111, xlabel="PC1", ylabel="PC2"))
    else:
        ax_list.append(fig.add_subplot(111))
    for i in range(len(feature[:,1])):
        if i%z1.shape[0]==0:
            ax_list[-1].scatter(feature[i, 0], feature[i, 1], alpha=0.5, color=cmap(i//z1.shape[0]), label=label_list[i//z1.shape[0]])
        else:
            ax_list[-1].scatter(feature[i, 0], feature[i, 1], alpha=0.5, color=cmap(i//z1.shape[0]))
    #ax_list[-1].grid()
    ax_list[-1].axes.xaxis.set_visible(False)
    ax_list[-1].axes.yaxis.set_visible(False)
    #ax_list[-1].legend(bbox_to_anchor=(0, -0.1), borderaxespad=0, loc="upper left", ncol=z1.shape[0], markerscale=9.0, fontsize=80)
    fig.tight_layout()

    fig.savefig(feature_img_path+"/{}_{}_{}_{:.0f}_{}.pdf".format(vis_tag, f_name, epoch, acc, consis_lambda))
    
    plt.clf()
    plt.close()

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

def feature_samples(f_name, idx_list, z, target, pred, da, p, epoch, acc, feature_img_path, consis_lambda):
    cmap = plt.get_cmap("tab10") # "terrain" & set num_of_class
    z = z.to('cpu').detach().numpy()
    target = target[:,np.newaxis].astype(int)
    idx_list = np.array(idx_list)[:,np.newaxis]
    for_csv = np.hstack([idx_list, z])
    for_csv = np.hstack([target+1, for_csv])
    
    save_dir = "./feature_vis_{}_{}".format(f_name, da)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    np.savetxt(save_dir+'/features.csv', for_csv, delimiter=',')
    
    fig = plt.figure()
    
    for i in range(len(target)):
        x = list(range(512))
        values = z[i]
        plt.plot(x, values, color=cmap(target[i][0]))
        fig.tight_layout()
        fig.savefig(save_dir+"/{}_{}_{}.png".format(p[i][0], idx_list[i][0], target[i][0]+1))
        plt.cla()
    
    return 0
    

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
