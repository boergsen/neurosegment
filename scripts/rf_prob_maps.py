__version__ = '1.02'

import os
import cPickle as pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import ShuffleSplit

from core.methods import compute_ilastik_features_RGB, balance_data, create_out_path, shuffle
from core.data.calcium_imaging import JaneliaData
from core.visuals import rf_roc_curve, rf_pr_curve

# global
label_sets = {
              'A' : [('++'), ('bg')], # very active vs background only
              'B' : [('++', '+'), ('bg')], # all positive vs background only
              'C' : [('++', '+'), ('xx', 'x')], # all positive versus all negative only
              'D' : [('++', '+'), ('-', 'xx', 'x', 'bg')], # all positive versus all negative (including background) (gt+++)
              'E' : [('++', '+', '-'), ('x', 'xx', 'bg')], # positive plus uncertain versus inactive plus background (gt+++x)
              'F' : [('++', '+', '-', 'x', 'xx'), ('bg')], # original Janelia GT
              'G' : [('++'), ('-')], # best vs uncertain
              'H' : [('++'), ('x')], # best vs autolabels
              'I' : [('++'), ('+')], # best vs second-best
              'J' : [('++'), ('+', '-', 'x', 'xx', 'bg')], # (gt++)
             }

def rf_create_features(subvolumes, channel, verbose=0):
    if verbose > 0: print 'creating features...'
    dim_x, dim_y = subvolumes[0].dims
    n_features = subvolumes[0].get('feats_ilastik_pseudoRGB_%s' % channel).shape[2]
    n_subvolumes = len(subvolumes)
    n_samples = n_subvolumes * dim_x * dim_y
    X = np.ndarray((n_samples, n_features), dtype=np.float32)
    for i in xrange(n_subvolumes):
        s = subvolumes[i]
        if verbose > 1: print "    * processing subvolume:", s.name
        try:
            feats = s.get('feats_ilastik_pseudoRGB_%s' % channel)
            if verbose > 1: print '      - loaded ilastik features'
        except:
            rgb = s.get('pseudo_rgb_%s' % channel)
            feats = compute_ilastik_features_RGB(rgb)
            if verbose > 1: print "      - computed ilastik features on each channel"
        if verbose > 1: print "      - append to feature set"
        feats = feats.reshape((dim_x * dim_y, -1))
        start = i * feats.shape[0]
        end = start + feats.shape[0]
        X[start:end, :] = feats
    return X

def rf_create_label_sets(subvolumes, positive_label_set, negative_label_set):
    labels = subvolumes[0].activity_labels
    labels_short = subvolumes[0].activity_labels_short
    dimx, dimy = subvolumes[0].dims
    if (positive_label_set is not 'full'):
        if isinstance(positive_label_set, str):
            positive_label_set = [positive_label_set]
        for label in positive_label_set:
            assert (label in labels) or (label in labels_short), \
                ValueError("Set with positive labels ill-defined.")
            if negative_label_set is not 'bg':
                assert label not in negative_label_set, \
                    ValueError("Positive label %s also specified as negative label!" % label)
    else:
        assert negative_label_set is 'bg', ValueError("'full' needs negative labels to be only background.")
        positive_label_set = labels
    if negative_label_set is not 'bg':
        for label in negative_label_set:
            assert (label in labels) or (label in labels_short) or (label == 'bg'), \
                ValueError("Set with negative labels ill-defined.")
            assert label not in positive_label_set, \
                ValueError("Negative label %s also specified as positive label!" % label)
    # indices will correspond index in full|nice|dummy X
    positive_pixel_idxs = []
    negative_pixel_idxs = []
    for i in xrange(len(subvolumes)):
        sub = subvolumes[i]
        # adjust indices according to their new index in X
        start = i * dimx * dimy
        pixel_idxs_pos = start + sub.get_pixel_indices_by_activity_labels(positive_label_set)
        positive_pixel_idxs.append(pixel_idxs_pos)
        pixel_idxs_neg = start + sub.get_pixel_indices_by_activity_labels(negative_label_set)
        negative_pixel_idxs.append(pixel_idxs_neg)
    return np.unique(np.concatenate(positive_pixel_idxs)).astype(np.uint32), np.unique(np.concatenate(negative_pixel_idxs)).astype(np.uint32)


def rf_save_features_and_label_sets_to_disk(h5out_x_ilastik, h5out_y_ilastik, subvolumes, label_sets, channel):
    assert isinstance(label_sets, dict)
    import os
    # save X once
    if not os.path.exists(h5out_x_ilastik):
        with h5.File(h5out_x_ilastik) as h5out:
            X = rf_create_features(subvolumes, channel=channel)
            h5out.create_dataset(name='X', data=X)
    else:
        print 'X already saved to', h5out_x_ilastik

    if not os.path.exists(h5out_y_ilastik): # Y's are identical for both channels
        # save in HDF5 group for label set: one Y for each set
        for key in label_sets.keys():
            set = label_sets[key]

            print 'exctract labels for set', set
            pos, neg = rf_create_label_sets(subvolumes, positive_label_set=set[0], negative_label_set=set[1])
            print 'number of positive labels in Y:', len(pos)
            print 'number of negative labels in Y:', len(neg)
            label_set_mask = shuffle(np.concatenate((pos, neg)))

            print 'create Y'
            n_samples = len(subvolumes) * subvolumes[0].dims[0] * subvolumes[0].dims[1]
            Y = np.empty((n_samples), dtype=np.uint8)
            Y[pos] = 1
            Y[neg] = 0

            with h5.File(h5out_y_ilastik) as h5out:
                print 'save to disk'
                grp = h5out.create_group(name=str(key))
                grp.create_dataset(name='Y', data=Y)
                grp.create_dataset(name='shuffled_mask', data=label_set_mask)
                grp.create_dataset(name='pos_mask', data=pos)
                grp.create_dataset(name='neg_mask', data=neg)
                grp.create_dataset(name='labels', data=str(set))
    else:
        print 'Y already saved to', h5out_y_ilastik
    print 'Done.'

def rf_load_Y(h5in_y_ilastik, label_set_key, get_mask=True):
    with h5.File(h5in_y_ilastik, 'r') as hin:
        print 'load Y for set', label_set_key
        grp = hin[label_set_key]
        Y = np.asarray(grp['Y'])
        mask = np.asarray(grp['shuffled_mask'])
        if get_mask:
            return Y, mask
        else:
            return Y[mask]

def rf_test_different_labelings(out_dir, h5in_x_ilastik, h5in_y_ilastik, max_balanced_class_size, data, channel, test_size=.75, n_folds=3,
                                n_trees=150, n_cpus=8):
    def compute_stats_for_rf_model(rf, xtest, ytest):
        print 'compute accuracy on test set'
        score = rf.score(xtest, ytest)
        print 'acc:', score
        print 'predict probabilites on test set'
        probs = rf.predict_proba(xtest)
        print 'compute ROC and PR curves'
        fpr, tpr, auc = rf_roc_curve(probs, ytest)
        prec, reca = rf_pr_curve(probs, ytest)
        return {'acc' : float(score),
                'fpr' : list(fpr),
                'tpr' : list(tpr),
                'roc_auc': float(auc),
                'precision_threshs' : list(prec),
                'recall_threshs' : list(reca),
                'n_pos_test' : len(np.where(ytest==1)[0]),
                'n_neg_test' : len(np.where(ytest==0)[0])}

    # stuff
    results = {}
    out_dir = os.path.join(out_dir, 'rf_labelings_%s_%s' % (data, channel), 'rf_test_label_sets_max%d' % max_balanced_class_size)
    create_out_path(out_dir)
    jd = JaneliaData(only_demo_volume=True)
    demo_sub = jd.demo_volume
    dim_x, dim_y = demo_sub.dims

    with h5.File(h5in_x_ilastik, 'r') as hin:
        print 'load X'
        print '  shape:', hin['X'].shape
        X = np.asarray(hin['X'])

    for set, labels in label_sets.iteritems():
        results[set] = {}

        # set output paths
        label_set_out_dir = os.path.join(out_dir, 'label set %s'%set)
        create_out_path(label_set_out_dir)
        cv_out_dir = os.path.join(label_set_out_dir, 'cv_runs')
        create_out_path(cv_out_dir)

        print 'load Y'
        Y, index_mask = rf_load_Y(h5in_y_ilastik, set, get_mask=True)
        n_samples = len(index_mask)
        print 'n_samples for set %s: %d' % (set, n_samples)

        sp = ShuffleSplit(X.shape[0], n_iter=n_folds, random_state=42, test_size=test_size)
        cv_cnt = 0
        best_score = 0
        best_model = None
        for train, test in iter(sp):
            #NOTE: label classes that are neither present in pos. or neg. label set will be excluded completely
            print np.sum((train.shape, test.shape))
            train = np.lib.arraysetops.intersect1d(index_mask, train)
            train = np.random.permutation(train)
            test = np.lib.arraysetops.intersect1d(index_mask, test)
            test = np.random.permutation(test)
            print np.sum((train.shape, test.shape))

            print '\n====  CV fold %d ====' % cv_cnt
            print 'take train and test split samples from X and Y'
            xtrain, ytrain, xtest, ytest = X[train, :], Y[train], X[test, :], Y[test]
            print 'balance training set'
            xtrain, ytrain = balance_data(xtrain, ytrain, max_balanced_class_size, verbose=1)
            print 'train random forest'
            rf = RandomForestClassifier(n_estimators=n_trees, n_jobs=n_cpus)
            rf.fit(xtrain, ytrain)
            print 'compute stats for unbalanced test set'
            stats_unbal =  compute_stats_for_rf_model(rf, xtest, ytest)
            print 'compute stats for balanced test set'
            xtest, ytest = balance_data(xtest, ytest, verbose=1)
            stats_bal = compute_stats_for_rf_model(rf, xtest, ytest)
            print 'append results'
            for tmp in [stats_bal, stats_unbal]:
                tmp['n_pos_train'] = len(np.where(ytrain==1)[0])
                tmp['n_neg_train'] = len(np.where(ytrain==0)[0])
            stats = {
                'channel' : channel,
                'cv_fold' : cv_cnt,
                'set' : set,
                'labels' : labels,
                'results_balanced_test' : stats_bal,
                'results_unbalanced_test' : stats_unbal
            }
            tmp = results[set]
            tmp[cv_cnt] = stats

            if stats['results_unbalanced_test']['acc'] > best_score:
                best_score = stats['results_unbalanced_test']['acc']
                best_model = rf

            print 'save model for current CV run'
            pickle.dump(rf, open(os.path.join(cv_out_dir, 'cv%d_random_forest.pickle'%cv_cnt), 'w'))
            print '--> pickled random forest'
            cv_cnt += 1

        print 'generate vizzz for demo subvolume'
        feats = jd.demo_volume.get('feats_ilastik_pseudoRGB_green')
        sub_pm = best_model.predict_proba(feats.reshape((dim_x * dim_y, -1)))
        sub_pm = sub_pm.reshape(dim_x, dim_y, -1)
        plt.imsave(os.path.join(cv_out_dir, '%s_pm0_gist.png' % demo_sub.name), sub_pm[:, :, 0], cmap='gist_heat', dpi=300)
        plt.imsave(os.path.join(cv_out_dir, '%s_pm1_gist.png' % demo_sub.name), sub_pm[:, :, 1], cmap='gist_heat', dpi=300)
        plt.imsave(os.path.join(cv_out_dir, '%s_pm0.png' % demo_sub.name), sub_pm[:, :, 0], dpi=300)
        plt.imsave(os.path.join(cv_out_dir, '%s_pm1.png' % demo_sub.name), sub_pm[:, :, 1], dpi=300)
        labeled_pm_ax = demo_sub.plot_activity_labels(image=sub_pm[:, :, 1], cmap='gray')
        plt.savefig(os.path.join(cv_out_dir, '%s_labeled_pm_modelscore=%f.png' % (demo_sub.name, best_score)), dpi=300)
        # pilf = plot_it_like_ferran(demo_gt, sub_pm[:, :, 1])
        # plt.imsave(join_path(test_out_dir, '%s_rgb_pred_score=%f.png' % (name_tag, sub_score)), pilf)
        plt.close('all')
    pickle.dump(results, open(os.path.join(out_dir, 'results.pickle'), 'w'))
    print '--> dumped results to', out_dir
    print 'Done, done and done.'





def process_command_line():
    import argparse
    # create a parser instance
    parser = argparse.ArgumentParser(description="Train random forests on different activity label sets. \n"
                                                 "OUTPUT:\n"
                                                 "   1) pickled random forest models to generate probability heat maps "
                                                 "      for every subvolume later on.\n"
                                                 "   2) pickled dictionaries with results (accuracy, roc auc, tpr, fpr, ...) "
                                                 "      evaluating different activity label sets used as ground truth",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # add arguments to parser
    req = parser.add_argument_group('required arguments')
    req.add_argument('--out_dir', type=str, default=False,
                     help='Folder where results will be saved to.')
    req.add_argument('--features_dir', type=str, default='data/full/features',
                     help='Folder with precomputed ilastik features (will be generated there if none found).')
    parser.add_argument('--channel', type=str, default='green', choices=['green', 'red'],
                        help='Data from which Calcium imaging channel to use.')
    parser.add_argument('--data',  type=str, default='nice', choices=['dummy', 'nice', 'full'],
                        help='Control quality and amount of subvolumes used. "nice": only subs with at least '
                             'one active neuron. "dummy": 5 demo volumes. "full": all subvolumes.')
    parser.add_argument('--cv_folds', type=int, default=3,
                        help='Number of cross-validation folds.')
    parser.add_argument('--n_trees', type=int, default=150,
                        help='Number of trees in forest.')
    parser.add_argument('--n_cpus', type=int, default=-1,
                        help='Number of CPUs to be used for training/testing.')
    parser.add_argument('--size_test_set', type=float, default=.75,
                        help='Amount of data used for testing (train size = 1 - test size).')
    return parser.parse_args()


if __name__ == '__main__':
    import sys
    args = process_command_line()

    if args.out_dir and args.features_dir:
        results_dir=args.out_dir
        features_dir=args.features_dir
        data=args.data
        channel=args.channel
        n_folds=args.cv_folds
        n_cpus=args.n_cpus
        n_trees=args.n_trees
        try:
            h5_x_ilastik = os.path.join(features_dir, 'X_ilastik_%s_%s.h5'%(data, channel))
            h5_y_ilastik = os.path.join(features_dir, 'Y_ilastik_%s.h5'%data)

            if not os.path.exists(h5_x_ilastik) or not os.path.exists(h5_y_ilastik):
                print 'No labels or instances found in %s. Create data...' % features_dir
                jd = JaneliaData(dummy_data=False, only_nice_volumes=False)
                sub_sets = { 'nice' : jd.nice_subs, 'dummy' : jd.dummy_subs, 'full' : jd.subvolumes }
                subvolumes = sub_sets[data]
                rf_save_features_and_label_sets_to_disk(h5out_x_ilastik=h5_x_ilastik, h5out_y_ilastik=h5_y_ilastik,
                                                        subvolumes=subvolumes, label_sets=label_sets, channel=channel)
            max_samples = [5000, 25000, 50000, 75000] # used for balancing the train set
            for size in max_samples:
                rf_test_different_labelings(results_dir, h5_x_ilastik, h5_y_ilastik, max_balanced_class_size=size,
                                            data=data, channel=channel, n_folds=n_folds, n_cpus=n_cpus,
                                            n_trees=args.n_trees, test_size=args.size_test_set)
            print 'Horray!'
            sys.exit(0)
        except: # something went wrong
            print 'Uuuups...'
            sys.exit(1)
    else:
        print 'Specify output directory for results and in/out directory used for storing ilastik features.'
        sys.exit(1)
