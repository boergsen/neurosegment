__version__ = '0.1'
import matplotlib
matplotlib.use('Agg')
import os
import matplotlib.pyplot as plt
from pystruct.learners import NSlackSSVM, OneSlackSSVM
from pystruct.models import EdgeFeatureGraphCRF

from core.methods import create_2d_edge_graph, create_out_path, compute_class_weights
from core.visuals import ssvm_plot_learning, plot_it_like_ferran
from core.data.calcium_imaging import JaneliaData


def create_XY(subvolumes, which_gt, feats_unary, feats_pairwise):
    img_x, img_y = subvolumes[0].dims
    edges = create_2d_edge_graph(img_x, img_y)

    X = []
    Y = []
    # An instance ``x_i`` is represented as a tuple ``(node_features, edges, edge_features)``
    for sub in subvolumes:
        print sub.name
        unaries = sub.ssvm_pack_unary_features(feats_unary, bias=True, normalize=True)
        edge_features = sub.ssvm_pack_pairwise_features(feats_pairwise, bias=True)
        x = (unaries, edges, edge_features)
        X.append(x)
    # Labels ``y_i`` have to be an array of shape (n_nodes,)
    for sub in subvolumes:
        gt = sub.get(which_gt).astype('uint8')
        y = gt.reshape(img_x * img_y)
        Y.append(y)
    return X, Y

def ssvm_construct_Y(ground_truth, subvolumes):
    img_x, img_y = subvolumes[0].dims
    Y = []
    for sub in subvolumes:
        gt = sub.get(ground_truth).astype('uint8')
        # Labels ``y_i`` have to be an array of shape (n_nodes,)
        y = gt.reshape(img_x * img_y)
        Y.append(y)
    return Y

def ssvm_construct_X(subvolumes, feats_unary, feats_pairwise, edges):
    X = []
    for sub in subvolumes:
        print sub.name
        unaries = sub.ssvm_pack_unary_features(feats_unary, bias=False, normalize=True)
        edge_features = sub.ssvm_pack_pairwise_features(feats_pairwise, bias=False)
        x = (unaries, edges, edge_features)
        X.append(x)
    return X


def doit(out_dir):
    def fit_model(X_train, Y_train):
        try:
            print '* fitting model...'
            model.fit(X_train, Y_train)
            return True
        except:
            return False

    ssvm_c = [50, 25, 15, 10, 5, 3, 1, .75, .5, .25, .1, .05, .01, .005, .001, .0005, .0001, .00005, .000001]
    ssvm_tol = -10 # default -10: stop only if no more constraints can be found
    ssvm_iter = 50
    ssvm_inactive_thresh = 1e-3
    which_ssvm = 'nslack'
    which_solver = ('ogm', {'alg': 'gc'})
    which_combo = which_ssvm + '_ogm-' + which_solver[1]['alg']
    out_dir = os.path.join(out_dir, 'ssvm_baseline', which_combo)
    create_out_path(out_dir, except_on_exist=False)
    cd = JaneliaData(dummy_data=False)

    #train_subs = cd.dummy_subs[:2]
    #test_subs = cd.dummy_subs[2:4]
    #lesub = cd.dummy_subs[4:]

    train_subs = cd.ssvm_train_set
    test_subs = cd.ssvm_test_set
    lesub = [cd.demo_volume]

    feats_unaries = ['feats_pm_++'] #'feats_pm_setA_green', 'feats_pm_setB_green', 'feats_pm_setC_green', 'feats_pm_setD_green', 'feats_pm_setE_green', 'feats_pm_setG_green', 'feats_pm_setH_green', 'feats_pm_setI_green', 'feats_pm_setJ_green', 'pseudo_rgb_green']
    feats_pairwise = ['feats_xcorr_green']

    gts = ['gt_active_++', 'gt_active_+++', 'gt_active_+++x']

    gt=gts[1]
    lesubX , lesubY = create_XY(lesub, gt, feats_unaries, feats_pairwise)

    X_train, Y_train = create_XY(train_subs, gt, feats_unaries, feats_pairwise)
    X_test, Y_test = create_XY(test_subs, gt, feats_unaries, feats_pairwise)

    Y = ssvm_construct_Y(gt, cd.nice_subs)
    class_weights = compute_class_weights(Y)
    crf_graph = EdgeFeatureGraphCRF(n_states=2, inference_method=which_solver, class_weight=class_weights)

    model = NSlackSSVM(crf_graph, verbose=1, C=ssvm_c[0], max_iter=ssvm_iter, n_jobs=8, tol=ssvm_tol, show_loss_every=1,
                       inactive_threshold=ssvm_inactive_thresh, inactive_window=50, batch_size=1000)

    cnt = 0
    while not fit_model(X_train, Y_train):
        cnt += 1
        try:
            model.C = ssvm_c[cnt]
            print 'trying again with', ssvm_c[cnt]
        except:
            cnt = None
            break
    if cnt is None:
        print 'SHIT!'
    else:
        print 'SUCCESS!'
        print ssvm_c[cnt]

    print '* scoring model...'
    score = model.score(X_test, Y_test)
    print score

    print '* saving results for test run'
    fig, _ = ssvm_plot_learning(model, time=False)
    fig.savefig(os.path.join(out_dir, 'ssvm_plot_learning.png'))
    plt.close('all')

    print '* saving prediction for demo volume'
    pred=model.predict(lesubX)
    lesub_vol = lesub[0]
    rgb_pred = plot_it_like_ferran(lesub_vol.get(gt), pred[0].reshape(512, 512))
    plt.imsave(os.path.join(out_dir, '%s_rgb_pred.png'% lesub_vol.name), rgb_pred)
    print '\nDone.'
    print 'Schmutz alarm...'
    print 'C:', ssvm_c
    IPython.embed()


if __name__ == '__main__':
    import sys
    import IPython
    args=sys.argv[1:]
    out_dir = args[0]
    doit(out_dir)
