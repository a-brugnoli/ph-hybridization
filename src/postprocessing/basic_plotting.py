import firedrake as fdrk
import matplotlib.pyplot as plt
from matplotlib import rcParams

SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('text', usetex= True) 

rcParams.update({'figure.autolayout': True})
rcParams['text.latex.preamble']=r"\usepackage{amsmath}\usepackage{bm}"
rcParams["legend.loc"] = 'best'


def tricontourf(field_2d, title=None, save_path=None):
    fig = plt.figure()
    axes = fig.add_subplot(111)
    contf = fdrk.tricontourf(field_2d, axes=axes)
    fig.colorbar(contf, ax=axes)
    if title is not None:
        axes.set_title(title)
    if save_path is not None:
        fig.savefig(save_path, dpi='figure', format='eps')
        

def trisurf(field_2d, title=None, save_path=None):
    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    surf = fdrk.trisurf(field_2d, axes=axes)
    fig.colorbar(surf, ax=axes)
    if title is not None:
        axes.set_title(title)
    if save_path is not None:
        fig.savefig(save_path, dpi='figure', format='eps')
      

def plot_signal(t_vec, signal_vec, title=None, save_path=None):
    plt.figure()
    plt.plot(t_vec, signal_vec)
    plt.grid(color='0.8', linestyle='-', linewidth=.5)
    plt.xlabel(r'Time')
    if title is not None:
        plt.title(title)
    if save_path is not None:
        plt.savefig(save_path, dpi='figure', format='eps')



def plot_signals(t_vec, *signals_vec, **options):
    plt.figure()
    for count, signal in enumerate(signals_vec):
        if "legend" in options:
            plt.plot(t_vec, signal, label=options["legend"][count])
    plt.grid(color='0.8', linestyle='-', linewidth=.5)
    plt.xlabel(r'Time')
    plt.legend()
    if "title" in options:
        plt.title(options["title"])
    if "save_path" in options:
        plt.savefig(options["save_path"], dpi='figure', format='eps')
