from os.path import dirname, abspath
import sys
import numpy as np
import matplotlib.pylab as plt
from matplotlib import gridspec
import matplotlib.animation as animation
### Move to parent directory
parent_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(parent_dir)
### Import module
from src.util import fft

class numerical_exp():

    def __init__(self, rcParams_dict={
                                    'font.family':'Times New Roman',
                                    'mathtext.fontset':'stix',
                                    'font.size':15,
                                    'xtick.direction':'in',
                                    'ytick.direction':'in',
                                    'axes.linewidth':1.5,
                                    'xtick.major.size':8,
                                    'ytick.major.size':8,}, 
                savefig=False, save_dir=None, title=None, file_type='png'):
        for key in rcParams_dict.keys(): plt.rcParams[str(key)] = rcParams_dict[str(key)]
        self.save = savefig
        self.dir = save_dir
        self.name = title
        self.ext = file_type

    def figure01(self, X, T, n_plt, cmap="seismic", file_name=''):
        aspect = n_plt/(X.shape[1]*80)
        t_min = T[0]
        t_max = T[n_plt - 1] if n_plt <= len(T) else T[-1]
        spec = gridspec.GridSpec(ncols=1, nrows=1)
        fig = plt.figure(figsize=(10, 3))
        ax = fig.add_subplot(spec[0])
        ax.imshow(X[:n_plt].T, aspect=aspect, cmap=cmap, 
                extent=[t_min, t_max, 1, X.shape[1]], 
                vmax=np.max(X[:n_plt]), vmin=np.min(X[:n_plt]))
        ax.set_ylabel(r'$X(t)$')
        ax.set_xlabel(r'Time [s]')
        ax.tick_params(labelleft=False, left=False)
        if self.save: plt.savefig(self.dir+self.name+'numerical_exp_fig01'+file_name+'.'+self.ext, bbox_inches="tight")  
        plt.show()

    def figure02(self, fractal_dim_list, intrinsic_dim_list, k_list, label_list, 
                c=('b', 'r', 'g', 'm'), m=('o', 's', '^', 'v'), file_name=''):
        spec = gridspec.GridSpec(ncols=1, nrows=1)
        fig = plt.figure(figsize=(10, 3))
        ax = fig.add_subplot(spec[0])
        for i in range(len(fractal_dim_list)): ax.axhline(y=fractal_dim_list[i], linestyle='--', c='k')
        for i in range(len(fractal_dim_list)): ax.plot(k_list, intrinsic_dim_list[i], marker=m[i], c=c[i], ms=5, label=label_list[i])
        ax.set_xlabel(r'Number of Nearest Neighbor $k$')
        ax.set_ylabel(r'Estimated dimension $\hat{D}$')
        fig.legend(loc="upper center", facecolor="white", edgecolor="white", bbox_to_anchor=(0.5, 1.06), ncol=len(fractal_dim_list))
        if self.save: plt.savefig(self.dir+self.name+'numerical_exp_fig02'+file_name+'.'+self.ext, bbox_inches="tight")  
        plt.show()

    def figure03(self, loss_valid_list, label_list, threshold_value=0.01, 
                c=('g', 'b', 'r', 'm', 'c'), 
                ls=((0, (2, 1)), (0, (5, 1)), (0, (8, 1)), (0, (2, 1, 1, 1)), (0, (5, 1, 1, 1))),
                y_lim=(0, None), file_name='') :
        spec = gridspec.GridSpec(ncols=1, nrows=1)
        fig = plt.figure(figsize=(10, 3))
        ax = fig.add_subplot(spec[0])
        ax.axhline(y=threshold_value, linestyle='-', c='k')
        for i in range(len(loss_valid_list)): ax.plot(np.arange(1, len(loss_valid_list[i])+1), loss_valid_list[i], lw=2, c=c[i], linestyle=ls[i], label=label_list[i])
        ax.set_xlabel(r'Epoch')
        ax.set_ylabel(r'Loss')
        ax.set_ylim(y_lim[0], y_lim[1])
        fig.legend(loc="upper center", facecolor="white", edgecolor="white", bbox_to_anchor=(0.5, 1.06), ncol=len(loss_valid_list))
        if self.save: plt.savefig(self.dir+self.name+'numerical_exp_fig03'+file_name+'.'+self.ext, bbox_inches="tight")  
        plt.show()

    def figure04(self, X_input, X_output, T, n_plt, cmap12="seismic", cmap3="seismic", file_name=''):
        aspect = n_plt/(X_input.shape[1]*80)
        t_min = T[0]
        t_max = T[n_plt - 1] if n_plt <= len(T) else T[-1]
        spec = gridspec.GridSpec(ncols=1, nrows=3, height_ratios=[1, 1, 1])
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(spec[0])
        ax.imshow(X_input[:n_plt].T, aspect=aspect, cmap=cmap12, 
                extent=[t_min, t_max, 1, X_input.shape[1]], 
                vmax=np.max(X_input[:n_plt]), vmin=np.min(X_input[:n_plt]))
        ax.set_ylabel(r'$X(t)$')
        ax.tick_params(labelleft=False, labelbottom=False, left=False)
        ax = fig.add_subplot(spec[1])
        ax.imshow(X_output[:n_plt].T, aspect=aspect, cmap=cmap12, 
                extent=[t_min, t_max, 1, X_output.shape[1]], 
                vmax=np.max(X_input[:n_plt]), vmin=np.min(X_input[:n_plt]))
        ax.set_ylabel(r'$\hat{X}(t)$')
        ax.tick_params(labelleft=False, labelbottom=False, left=False)
        ax = fig.add_subplot(spec[2])
        ax.imshow((X_output[:n_plt] - X_input[:n_plt]).T, aspect=aspect, cmap=cmap3, 
                extent=[t_min, t_max, 1, X_output.shape[1]], 
                vmax=np.max(np.abs(X_input[:n_plt])), vmin=-np.max(np.abs(X_input[:n_plt])))
        ax.set_ylabel(r'$\hat{X}(t)-X(t)$')
        ax.set_xlabel(r'Time [s]')
        ax.tick_params(labelleft=False, left=False)
        if self.save: plt.savefig(self.dir+self.name+'numerical_exp_fig04'+file_name+'.'+self.ext, bbox_inches="tight")  
        plt.show()

    def figure05(self, x, T, n_plt, n_attractor, sort=True, file_name=''):
        if sort: x = x[:, np.argsort(np.var(x, axis=0))[::-1]] 
        spec = gridspec.GridSpec(ncols=2, nrows=3, height_ratios=[1, 1, 1], width_ratios=[1.5, 1], wspace=0.05)
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(spec[0, 0])
        ax.plot(T[:n_plt], x[:, 0][:n_plt], '-', c='k')
        ax.set_xlim(T[:n_plt][0], T[:n_plt][-1])
        ax.set_ylabel(r'$x_{1}$')
        ax.tick_params(labelbottom=False)
        ax = fig.add_subplot(spec[1, 0])
        ax.plot(T[:n_plt], x[:, 1][:n_plt], '-', c='k')
        ax.set_xlim(T[:n_plt][0], T[:n_plt][-1])
        ax.set_ylabel(r'$x_{2}$')
        ax.tick_params(labelbottom=False)
        ax = fig.add_subplot(spec[2, 0])
        ax.plot(T[:n_plt], x[:, 2][:n_plt], '-', c='k')
        ax.set_xlim(T[:n_plt][0], T[:n_plt][-1])
        ax.set_ylabel(r'$x_{3}$')
        ax.set_xlabel(r'Time [s]')
        ax = fig.add_subplot(spec[:, 1], projection='3d')
        ax.plot(x[:n_attractor, 0], x[:n_attractor, 1], x[:n_attractor, 2], '.', ms=1, c='k')
        ax.set_xlabel(r'$x_{1}$', labelpad=-15)
        ax.set_ylabel(r'$x_{2}$', labelpad=-15)
        ax.set_zlabel(r'$x_{3}$', labelpad=-15)
        ax.tick_params(labelbottom=False, labelleft=False, labelright=False)
        #ax.set_box_aspect((1,1,1))
        if self.save: plt.savefig(self.dir+self.name+'numerical_exp_fig05'+file_name+'.'+self.ext, bbox_inches="tight")  
        plt.show()
    