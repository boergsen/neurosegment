__author__ = 'akiem'
__version__ = 'V3'

import matplotlib
matplotlib.use('Agg')
from core.methods import compute_class_weights, normalize_image_to_zero_one, create_2d_edge_graph, create_out_path
from core.visuals import *
from core.data.calcium_imaging import JaneliaData
from os.path import join as join_path
import cPickle as pickle
from pystruct.models import EdgeFeatureGraphCRF
from pystruct.learners import NSlackSSVM
from sklearn import grid_search
import IPython

def create_unary_feats_from_pseudo_rgb(subvolume):
    mean = normalize_image_to_zero_one(subvolume.get('mean_green')).reshape((512*512, -1))
    max = normalize_image_to_zero_one(subvolume.get('max_green')).reshape((512*512, -1))
    std = normalize_image_to_zero_one(subvolume.get('std_green')).reshape((512*512, -1))
    bias = np.ones_like(std)
    return np.hstack([bias, mean, max, std])

def create_XY(subvolumes, which_gt, feats_pairwise='feats_xcorr_green'):
    img_x, img_y = subvolumes[0].dims
    X = []
    Y = []
    # An instance ``x_i`` is represented as a tuple ``(node_features, edges, edge_features)``
    for sub in subvolumes:
        print sub.name
        unaries = create_unary_feats_from_pseudo_rgb(sub)
        pms = sub.get('feats_pm_++')
        unaries = np.hstack([unaries, pms])
        edge_features = sub.get(feats_pairwise)
        edges = create_2d_edge_graph(img_x, img_y)
        x = (unaries, edges, edge_features)
        X.append(x)
    # Labels ``y_i`` have to be an array of shape (n_nodes,)
    for sub in subvolumes:
        gt = sub.get(which_gt).astype('uint8')
        y = gt.reshape(img_x * img_y)
        Y.append(y)
    return X, Y


which_gt = 'gt_active_++'
out_dir = 'out/db_vizzz/grid_search%s'%__version__
create_out_path(out_dir, except_on_exist=False)
n_cv_folds = 3
jd = JaneliaData()
train_vols = jd.ssvm_train_set
test_vols = jd.ssvm_test_set
nice_vols = jd.nice_subs
demo_vol = jd.demo_volume

X_train, Y_train = create_XY(train_vols, which_gt)
X_test, Y_test = create_XY(test_vols, which_gt)
class_weights = compute_class_weights(Y_train)


parameters = {
    'C' : [10, .5, .1, .01, .001, .0001],
    'tol' : [-10, .01, .1, 10],
    'inactive_threshold' : [1e-5, 1e-2],
    'batch_size' : [100, 1000]
#    'inactive_window' : [50]
}

# CRF with different solvers
#crf_dd = EdgeFeatureGraphCRF(inference_method=('ogm', {'alg':'dd'}), class_weight=class_weights)
#crf_qpbo = EdgeFeatureGraphCRF(inference_method='qpbo', class_weight=class_weights)
crf_gc = EdgeFeatureGraphCRF(n_states=2, n_features=X_train[0][0].shape[1], n_edge_features=X_train[0][2].shape[1],
                             inference_method=('ogm', {'alg':'gc'}), class_weight=class_weights)
print crf_gc

ssvm = NSlackSSVM(crf_gc, verbose=0, max_iter=42, n_jobs=1, show_loss_every=0, inactive_window=50)
ssvm_gs = grid_search.GridSearchCV(ssvm, parameters, cv=n_cv_folds, verbose=2, error_score=0.000042, n_jobs=8)
try:
    ssvm_gs.fit(X_train, Y_train)
except:
    print 'komischer rank error terror'
    pickle.dump(ssvm_gs, open(join_path(out_dir, 'ssvm_grid_search%s.p'%__version__), 'w'))
    IPython.embed()

print 'Number of CV folds:', n_cv_folds
print 'Ground truth used:', which_gt
print 'SSVM grid scores:', ssvm_gs.grid_scores_
#print ssvm_gs.best_estimator_
print 'GridSearch best params:', ssvm_gs.best_params_
print 'GridSearch best score:', ssvm_gs.best_score_
print 'Done.'
IPython.embed()
