import matplotlib.pyplot as plt
from . import outputs
from . import similarity

def layer_by_layer_plot(layer_similarity_mat, x_label='', y_label='', show_max=False):
    fig = plt.figure()
    ax = fig.add_subplot()

    cax = ax.imshow(layer_similarity_mat, vmin=0, vmax=1, interpolation="nearest", aspect='equal')
    fig.colorbar(cax, fraction=0.025)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if show_max:
        ax.plot(range(layer_similarity_mat.shape[1]), layer_similarity_mat.max(dim=0).indices, c='red')

    ax.invert_yaxis()

    return fig