__version__ = '0.8'

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
from sklearn.metrics.ranking import auc as area_under_curve


#FIXME is this correct?
def visualize_unaries(weights, unaries):
    # feat1=unaries[:,0] or X[0][0][:,0] for first feature of first x
    # w: weights for every linear combination of features per class
    # neg=w[0] * unaries[:,0] + w[1] * unaries[:,1] + w[2] * unaries[:,2]
    # pos=w[3] * unaries[:,0] + w[4] * unaries[:,1] + w[5] * unaries[:,2]

    n_states=2
    n_features = unaries.shape[1]

    negatives = []
    positives = []

    # class0 = w0*f0 + w1*f1 + w2*f2
    # class1 = w3*f0 + w4*f1 + w5*f2
    w_cnt=0
    for state in xrange(n_states):
        for f in xrange(n_features):
            wf = weights[w_cnt] * unaries[:,f]
            w_cnt += 1
            if state == 0:
                negatives.append(wf.reshape(512,512))
            if state == 1:
                positives.append(wf.reshape(512,512))
    return positives, negatives


def visualize_fore_and_background(weights, unaries):
    pos, neg = visualize_unaries(weights, unaries)
    return np.sum(np.asarray(pos), axis=0), np.sum(np.asarray(neg), axis=0)


def visualize_edges(edges, edge_features, feature_idx=0, dims=(512,512)):
    """ Visualize the edge potentials of the factor graph. There will be two
    resulting images, one for horizontal, one for vertical edges between pixels.

    :param edges: list of edges (tuples consisting of pixel indices)
    :param edge_features: computed potentials for each edge
    :param feature_idx: choose the feature to visualize (feature_idx[0] corresponds to bias)

    :return: two images, visualizing horizontal and vertical edges resp.
    """
    assert edges.shape[0] == edge_features.shape[0]
    assert feature_idx <= edge_features.shape[1]
    img_hx, img_hy, img_vx, img_vy = dims[0]-1, dims[1], dims[0], dims[1]-1
    img_h = np.zeros(img_hx * img_hy)
    img_v = np.zeros(img_vx * img_vy)
    features = edge_features[:, feature_idx]
    cnt_h = 0
    cnt_v = 0
    for i in xrange(edges.shape[0]): # for every edge (that is, every node pair)
        #check for horizontal edge
        if edges[i][0]+1 == edges[i][1]:
            img_h[cnt_h] = features[i]
            cnt_h += 1
        else:
            img_v[cnt_v] = features[i]
            cnt_v += 1
    return img_h.reshape((img_hy, img_hx)), img_v.reshape((img_vy, img_vx))


def plot_it_like_ferran(gt, pred):
    assert gt.shape == pred.shape
    dims = gt.shape
    rgb_img = np.zeros([dims[0], dims[1], 3])
    rgb_img[:,:,0] = gt
    rgb_img[:,:,1] = pred
    rgb_img[:,:,2] = gt
    return rgb_img

def show_graph(edges):
    import networkx as nx
    G = nx.Graph()
    G.add_edges_from(edges)
    nx.draw(G)
    plt.show()

def rf_roc_curve(probs, Y_test, out_path=None, return_axis=False, title=None):
    fpr, tpr, _ = roc_curve(Y_test, probs[:, 1], pos_label=1)
    roc_auc = roc_auc_score(Y_test, probs[:, 1]) # NOTE class weights can be used here instead of balancing test data
    if out_path is not None:
        fig = plt.figure()
        fig.add_axes(plot_roc_curve_on_axis(fpr, tpr, roc_auc, title=title))
        plt.savefig(out_path)
        print "--> saved ROC curve to", out_path
        plt.close('all')
    if return_axis:
        return plot_roc_curve_on_axis(fpr, tpr, roc_auc)
    return fpr, tpr, roc_auc

def rf_pr_curve(probs, Y_test, out_path=None, return_axis=False, title=None):
    precision, recall, _ = precision_recall_curve(Y_test, probs[:, 1], pos_label=1)
    if out_path is not None:
        fig = plt.figure()
        fig.add_axes(plot_pr_curve_on_axis(precision, recall, title=title))
        plt.savefig(out_path)
        print "--> saved PR curve to", out_path
        plt.close('all')
    if return_axis:
        return plot_pr_curve_on_axis(precision, recall)
    return precision, recall

def plot_roc_curve_on_axis(fpr, tpr, auc=None, color='b', linestyle='solid', label='', title='', label_random='', ax=None):
    if ax == None:
        ax = plt.gca()
    if auc is None:
        auc = area_under_curve(fpr, tpr, reorder=True)
    ax.plot(fpr, tpr, label='%s (AUC: %.02f)' % (label, auc*100), linestyle=linestyle, color=color)
    ax.plot([0, 1], [0, 1], label=label_random, color='k', linestyle=':', linewidth=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title(title)
    ax.legend(loc=4, fancybox=True)
    return ax

def plot_pr_curve_on_axis(precision, recall, color='b', linestyle='solid', label='', title='', label_random='', ax=None):
    if ax is None:
        ax = plt.gca()
    ax.plot(precision, recall, label=label, linestyle=linestyle, color=color)
    ax.plot([1, 0], [0, 1], label=label_random, color='k', linestyle=':', linewidth=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc=3, fancybox=True)
    return ax

def sort_legend(ax, loc=0):
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0])) # (c) stackoverflow
    ax.legend(handles, labels, loc=loc)
    return ax

def sort_legend_stupid_labelings(ax):
    ax = sort_legend(ax)
    h, l = ax.get_legend_handles_labels()
    l, h = zip(*sorted(zip(l, h), key=lambda t: t[0]))
    h = h[-8:] + h[:-8] # don't ask
    l = l[-8:] + l[:-8] # (please)
    # h = h[-3:] + h[:-3]
    # l = l[-3:] + l[:-3]
    ax.legend(h, l, loc=0, fancybox=True)
    return ax

def plot_ssvm_learning_on_axis(trained_model, title=None, time=True, show_primal_dots_label=False, axes=None, linewidth=1,
                               linestyle=None, color=None, label=None):
        """Plot optimization curves and cache hits.

        Create a plot summarizing the optimization / learning process of an SSVM.
        It plots the primal and cutting plane objective (if applicable) and also
        the target loss on the training set against training time.
        For one-slack SSVMs with constraint caching, cached constraints are also
        contrasted against inference runs.

        Parameters
        -----------
        ssvm : object
            SSVM learner to evaluate. Should work with all learners.

        time : boolean, default=True
            Whether to use wall clock time instead of iterations as the x-axis.

        Notes
        -----
        Warm-starting a model might mess up the alignment of the curves.
        So if you warm-started a model, please don't count on proper alignment
        of time, cache hits and objective.

        (copied from pystruct and adjusted to print on axes plus adding extra plotting parameters)
        """

        ssvm = trained_model
        print(ssvm)
        if hasattr(ssvm, 'base_ssvm'):
            ssvm = ssvm.base_ssvm
        print("Iterations: %d" % len(ssvm.objective_curve_))
        print("Objective: %f" % ssvm.objective_curve_[-1])
        inference_run = None
        if hasattr(ssvm, 'cached_constraint_'):
            inference_run = ~np.array(ssvm.cached_constraint_)
            print("Gap: %f" %
                  (np.array(ssvm.primal_objective_curve_)[inference_run][-1] -
                   ssvm.objective_curve_[-1]))

        if time and hasattr(ssvm, 'timestamps_'):
            print("loading timestamps")
            inds = np.array(ssvm.timestamps_)
            inds = inds[2:len(ssvm.objective_curve_) + 1] / 60.
            inds = np.hstack([inds, [inds[-1]]])
            axes[0].set_xlabel('training time (min)')
        else:
            inds = np.arange(len(ssvm.objective_curve_))
            axes[0].set_xlabel('QP iterations')

        axes[0].set_title("Objective", y=1.02)
        axes[0].plot(inds, ssvm.objective_curve_, label="C=%s (dual)"%label, linestyle='--', color=color, linewidth=linewidth)
        axes[0].set_yscale('log')
        if hasattr(ssvm, "primal_objective_curve_"):
            axes[0].plot(inds, ssvm.primal_objective_curve_,
                         label="C=%s (cached primal)"%label if inference_run is not None
                         else "C=%s (primal)"%label, color=color, linewidth=linewidth)
        if inference_run is not None:
            inference_run = inference_run[:len(ssvm.objective_curve_)]
            if show_primal_dots_label:
                label = "C=%s (primal)"%label
            else:
                label = ''
            axes[0].plot(inds[inference_run],
                         np.array(ssvm.primal_objective_curve_)[inference_run],
                         '.', label=label, color=color)
        axes[0].legend(loc=4, fancybox=True, fontsize='small')
        if time and hasattr(ssvm, "timestamps_"):
            axes[1].set_xlabel('training time (min)')
        else:
            axes[1].set_xlabel('QP iterations')
        try:
            axes[1].plot(inds[::ssvm.show_loss_every], ssvm.loss_curve_, color=color, linewidth=linewidth)
        except:
            axes[1].plot(ssvm.loss_curve_, color=color, linewidth=linewidth)
        axes[1].set_title("Training error", y=1.02)
        axes[1].set_yscale('log')
        return axes


def ssvm_plot_learning(trained_model, title=None, time=True, axes=None):
        """Plot optimization curves and cache hits.

        Create a plot summarizing the optimization / learning process of an SSVM.
        It plots the primal and cutting plane objective (if applicable) and also
        the target loss on the training set against training time.
        For one-slack SSVMs with constraint caching, cached constraints are also
        contrasted against inference runs.

        Parameters
        -----------
        ssvm : object
            SSVM learner to evaluate. Should work with all learners.

        time : boolean, default=True
            Whether to use wall clock time instead of iterations as the x-axis.

        Notes
        -----
        Warm-starting a model might mess up the alignment of the curves.
        So if you warm-started a model, please don't count on proper alignment
        of time, cache hits and objective.

        (copied from pystruct and adjusted to print on axes)
        """

        ssvm = trained_model
        print(ssvm)
        if hasattr(ssvm, 'base_ssvm'):
            ssvm = ssvm.base_ssvm
        print("Iterations: %d" % len(ssvm.objective_curve_))
        print("Objective: %f" % ssvm.objective_curve_[-1])
        inference_run = None
        if hasattr(ssvm, 'cached_constraint_'):
            inference_run = ~np.array(ssvm.cached_constraint_)
            print("Gap: %f" %
                  (np.array(ssvm.primal_objective_curve_)[inference_run][-1] -
                   ssvm.objective_curve_[-1]))
        if hasattr(ssvm, "loss_curve_"):
            n_plots = 2
            fig, axes = plt.subplots(1, 2)
        else:
            n_plots = 1
            fig, axes = plt.subplots(1, 1)
            axes = [axes]

        fig.set_size_inches(12, 5)
        if title is not None:
            fig.suptitle(title)

        if time and hasattr(ssvm, 'timestamps_'):
            print("loading timestamps")
            inds = np.array(ssvm.timestamps_)
            inds = inds[2:len(ssvm.objective_curve_) + 1] / 60.
            inds = np.hstack([inds, [inds[-1]]])
            axes[0].set_xlabel('training time (min)')
        else:
            inds = np.arange(len(ssvm.objective_curve_))
            axes[0].set_xlabel('QP iterations')

        axes[0].set_title("Objective")
        axes[0].plot(inds, ssvm.objective_curve_, label="dual")
        axes[0].set_yscale('log')
        if hasattr(ssvm, "primal_objective_curve_"):
            axes[0].plot(inds, ssvm.primal_objective_curve_,
                         label="cached primal" if inference_run is not None
                         else "primal")
        if inference_run is not None:
            inference_run = inference_run[:len(ssvm.objective_curve_)]
            axes[0].plot(inds[inference_run],
                         np.array(ssvm.primal_objective_curve_)[inference_run],
                         'o', label="primal")
        axes[0].legend(loc=4, fancybox=True, fontsize='small')
        if n_plots == 2:
            if time and hasattr(ssvm, "timestamps_"):
                axes[1].set_xlabel('training time (min)')
            else:
                axes[1].set_xlabel('QP iterations')
            try:
                axes[1].plot(inds[::ssvm.show_loss_every], ssvm.loss_curve_)
            except:
                axes[1].plot(ssvm.loss_curve_)
            axes[1].set_title("Training error")
            axes[1].set_yscale('log')
        return fig, axes


