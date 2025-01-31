import matplotlib.pyplot as plt
import numpy as np
import torch
from . import outputs
from . import similarity

def model_display_name(model_name):
    return model_name.split('/')[-1]

def layer_by_layer_plot(layer_similarity_mat, x_label='', y_label='', show_max=False, cmap='inferno'):
    fig = plt.figure()
    ax = fig.add_subplot()

    cax = ax.imshow(layer_similarity_mat, vmin=0, vmax=1, interpolation="nearest", aspect='equal', cmap=cmap)
    fig.colorbar(cax, fraction=0.02)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if show_max:
        ax.plot(range(layer_similarity_mat.shape[1]), layer_similarity_mat.max(dim=0).indices, c='red')

    ax.invert_yaxis()

    return fig


def get_from_all_mknn(mknns, m1, m2):
    if (m1,m2) in mknns:
        return mknns[(m1,m2)][1:,1:]
    else:
        return mknns[(m2,m1)][1:,1:].T

def big_mat_plot(mknns, mat_model_names, tick_spacing=10, invert_yaxis=True, rotate_model_names=True, figsize=(10,10), cmap='inferno'):
    big_mat = torch.cat([torch.cat([get_from_all_mknn(mknns, m1, m2).T for m2 in mat_model_names]).T for m1 in mat_model_names])
    # big_mat = np.ma.array(big_mat, mask=np.tri(big_mat.shape[0], big_mat.shape[1], k=0).T)

    model_layers = np.array([get_from_all_mknn(mknns, m, mat_model_names[0]).shape[0] for m in mat_model_names])

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()

    cax = ax.imshow(big_mat, vmin=0, vmax=1, interpolation="nearest", aspect='equal', cmap=cmap)
    # fig.colorbar(cax, fraction=0.02)

    # Layer ticks
    tick_positions = np.array([])
    tick_labels = np.array([], dtype=np.int64)
    depth = 0
    for d in model_layers:
        layers = np.arange(tick_spacing, d-tick_spacing, tick_spacing)
        tick_positions = np.concatenate((tick_positions, layers + depth))
        tick_labels = np.concatenate((tick_labels, layers))
        depth += d

    ax.xaxis.set_ticks(tick_positions, tick_labels)
    ax.yaxis.set_ticks(tick_positions, tick_labels)

    # Model ticks
    model_tick_positions = model_layers.cumsum()
    model_tick_positions = np.insert(model_tick_positions, 0, 0)
    model_tick_positions = (model_tick_positions[1:] + model_tick_positions[:-1])/2

    display_model_names = [model_display_name(n) for n in mat_model_names]

    if invert_yaxis:
        sec = ax.secondary_xaxis(location='top')
    else:
        sec = ax.secondary_xaxis(location=0)

    sec.xaxis.set_ticks(model_tick_positions, display_model_names)

    if rotate_model_names:
        sec.tick_params('x', length=20, width=0, rotation=90)
    else:
        sec.tick_params('x', length=20, width=0, rotation=0)

    secy = ax.secondary_yaxis(location=0)
    secy.yaxis.set_ticks(model_tick_positions, display_model_names)

    if rotate_model_names:
        secy.tick_params('y', length=20, width=0, rotation=0)
    else:
        secy.tick_params('y', length=20, width=0, rotation=90)

    for label in secy.get_yticklabels():
        label.set_verticalalignment('center')

    # Seperators
    seperator_positions = model_layers.cumsum()[:-1]

    if invert_yaxis:
        sec2 = ax.secondary_xaxis(location='top')
    else:
        sec2 = ax.secondary_xaxis(location=0)

    sec2.xaxis.set_ticks(seperator_positions - 0.5, labels=[])
    sec2.tick_params('x', length=40, width=1, colors='gray')

    secy2 = ax.secondary_yaxis(location=0)
    secy2.yaxis.set_ticks(seperator_positions - 0.5, labels=[])
    secy2.tick_params('y', length=40, width=1, colors='gray')

    if invert_yaxis:
        ax.xaxis.tick_top()
        ax.invert_yaxis()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    return fig