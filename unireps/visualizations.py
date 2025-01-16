import matplotlib.pyplot as plt
from . import outputs
from . import similarity

def layer_by_layer_plot(layer_similarity_mat, model_1='', model_2='', show_max=False):
    fig = plt.figure()
    ax = fig.add_subplot()

    cax = ax.matshow(layer_similarity_mat, vmin=0, vmax=1)
    fig.colorbar(cax, fraction=0.025)
    
    ax.xaxis.set_label_position('top')
    ax.set_xlabel(model_2)
    ax.set_ylabel(model_1)

    if show_max:
        ax.plot(range(layer_similarity_mat.shape[1]), layer_similarity_mat.max(dim=0).indices, c='red')

    return fig