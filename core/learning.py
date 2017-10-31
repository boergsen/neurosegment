__version__ = '0.5'

import cPickle as pickle
import h5py as h5
import json
import numpy as np
import os
import sys

import matplotlib.pyplot as plt
from pystruct.learners import NSlackSSVM, OneSlackSSVM, FrankWolfeSSVM
from pystruct.models import EdgeFeatureGraphCRF
from pystruct.utils import SaveLogger as pystruct_logger
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score

from methods import create_2d_edge_graph, compute_class_weights
from core.data.calcium_imaging import CalciumDataAbstract as CalciumData
from core.data.subvolumes import SubvolumeAbstract as Subvolume


class ClassifierPipeline(object):

    model = None
    stats = None
    X_train = None
    Y_train = None
    X_test = None
    Y_test = None

    def predict_on_subvolume(self, subvolume, channel, verbosity):
        raise NotImplementedError('Method has to be implemented by subclasses.')

    def dump_train_test_data(self, h5_out):
        #assert self.has_data_loaded(), RuntimeError('Data not initialized.')
        assert not os.path.isfile(h5_out), IOError('Output H5 file already exists: %s' % h5_out)
        with h5.File(h5_out, 'w') as h5out:
            h5out.create_dataset(name='X_train', data=self.X_train)
            h5out.create_dataset(name='Y_train', data=self.Y_train)
            h5out.create_dataset(name='X_test', data=self.X_test)
            h5out.create_dataset(name='Y_test', data=self.Y_test)
        print '--> dumped test and train sets to', h5_out


class RandomForestPipeline(ClassifierPipeline):

    # "Constructors": call one of the class methods to start a pipeline and return its instance

    @classmethod
    def init_model_from_file(cls, pickled_forest):
        pipeline_obj = cls()
        pipeline_obj._start(load_pickled_forest_path=pickled_forest)
        return pipeline_obj

    @classmethod
    def init_model_from_scratch(cls, calcium_data, which_gt, out_dir, n_trees=255, n_jobs=-1, test_size=.75, only_nice_volumes=False,
                                balance_train_data=True, balance_test_data=False, exclude_inactive_neurons=True,
                                exclude_labels=False, per_class_sample_size=-1, channel='green'):
        pipeline_obj = cls()
        pipeline_obj._start(out_dir=out_dir, calcium_data=calcium_data, which_gt=which_gt, n_trees=n_trees, n_jobs=n_jobs, test_size=test_size,
                            only_nice_volumes=only_nice_volumes, balance_train_data=balance_train_data,
                            balance_test_data=balance_test_data, exclude_inactive_neurons=exclude_inactive_neurons,
                            exclude_labels=exclude_labels, per_class_sample_size=per_class_sample_size,
                            load_train_test_path=False, load_pickled_forest_path=False, channel=channel)
        return pipeline_obj

    @classmethod
    def load_model(cls, pickled_forest_path, h5_train_test_path):
        pipeline_obj = cls()
        pipeline_obj._start(load_train_test_path=h5_train_test_path, load_pickled_forest_path=pickled_forest_path)
        return pipeline_obj

    def _start(self, out_dir=None, calcium_data=None, which_gt=None, n_trees=255, n_jobs=-1, test_size=.75, only_nice_volumes=False,
                 balance_train_data=True, balance_test_data=False, exclude_inactive_neurons=True, exclude_labels=False, per_class_sample_size=-1,
                 load_train_test_path=False, load_pickled_forest_path=False, channel='green'):

        print '\n===== Random forest pipeline ====='

        if load_pickled_forest_path: # path given? False by default.
            try:
                print '* loading random forest from disk...'
                self.model = pickle.load(open(load_pickled_forest_path, 'r'))
            except:
                raise IOError('Random forest pickle file not found: %s' % load_pickled_forest_path)

        if load_train_test_path: # path given? False by default.
            try:
                print '* loading test/train data from disk'
                with h5.File(load_train_test_path) as data:
                    X_train, Y_train, X_test, Y_test = np.asarray(data['X_train']), np.asarray(data['Y_train']), \
                                                   np.asarray(data['X_test']), np.asarray(data['Y_test'])
                print '* X_train:', X_train.shape, 'X_test:', X_test.shape, 'Y_train:', Y_train.shape, 'Y_test:', Y_test.shape
            except:
                raise IOError('No HDF5 file found in %s. Create test/train data first (--> init model from scratch).' % load_train_test_path)
        else:
            print '* creating test/train data (test size: %.02f)' % test_size
            assert issubclass(type(calcium_data), CalciumData), AssertionError('specify a valid Calcium data subclass instance')

            X_train, Y_train, X_test, Y_test, self.stats = calcium_data.create_train_test_data_random_forest(
                which_gt=which_gt, test_size=test_size, only_nice_volumes=only_nice_volumes,
                exclude_inactive_neurons=exclude_inactive_neurons, exclude_labels=exclude_labels,
                balance_train_data=balance_train_data, balance_test_data=balance_test_data,
                per_class_sample_size=per_class_sample_size, channel=channel)
        self.X_train, self.Y_train, self.X_test, self.Y_test = X_train, Y_train, X_test, Y_test

        if not load_pickled_forest_path:
            print '* training random forest'
            rf = RandomForestClassifier(n_estimators=n_trees, n_jobs=n_jobs)
            rf.fit(X_train, Y_train)
            self.model = rf
            print '\n--> model trained'
            self.plot_learning(out_dir)
            self.stats['test_score'] = self.test_set_score()
            print '--> accuracy on test set:', str(self.stats['test_score'])

            if self.stats is not None:
                self.stats['n_trees'] = n_trees
                # with open(out_dir + 'random_forest_summary.txt', 'w') as f:
                #     for line in self.print_summary():
                #         f.writelines(line + '\n')
                with open(out_dir + 'random_forest_stats.json', 'w') as json_out:
                    json_out.write(json.dumps(self.stats, indent=0, sort_keys=True))
                print '--> saved random forest stats to', out_dir

    def is_trained(self):
        if self.model is not None:
            return True
        else:
            return False

    def has_data_loaded(self):
        if self.X_train is None or self.X_test is None or self.Y_train is None or self.Y_test is None:
            return False
        else:
            return True

    def predict_on_subvolume(self, subvolume, channel='green', verbosity=0):
        assert self.is_trained(), RuntimeError('No model trained/loaded!')
        assert issubclass(type(subvolume), Subvolume), AssertionError('specify a valid Subvolume subclass instance')
        if verbosity > 0: print '* generating probability map'
        dim_x, dim_y = subvolume.dims
        if verbosity > 0: print '  * loading pseudoRGB ilastik features for', subvolume.name
        feats = subvolume.get('feats_ilastik_pseudoRGB_%s' % channel)
        if verbosity > 0: print '  * predicting probabilities'
        pm = self.model.predict_proba(feats.reshape((dim_x * dim_y, -1)))
        return pm.reshape(dim_x, dim_y, -1)

    def test_set_score(self):
        """Returns the mean accuracy on the test data and labels."""
        return self.model.score(self.X_test, self.Y_test)

    def dump_model(self, pickle_out):
        assert self.is_trained(), RuntimeError('Train model first!')
        assert not os.path.isfile(pickle_out), IOError('Output file already exists: %s' % pickle_out)
        pickle.dump(self.model, open(pickle_out, 'w'))
        print '--> dumped forest to', pickle_out


    def print_summary(self):
        return [
            '\n===== Summary =====',
             '* test set size: %s' % str(self.stats['test_size']),
             '* ground truth used for labels: %s' % str(self.stats['gt']),
             '* number of overall samples in X (before balancing): %s' % str(self.stats['n_samples']),
             '* maximum number of samples per class (for balancing): %s' % str(self.stats['per_class_sample_size']),
             '* (train set) number of pos. samples (foreground): %s' % str(self.stats['n_pos_samples_train']),
             '* (train set) number of neg. samples (background): %s' % str(self.stats['n_neg_samples_train']),
             '* (test set)n number of pos. samples (foreground): %s' % str(self.stats['n_pos_samples_test']),
             '* (test set) number of neg. samples (background): %s' % str(self.stats['n_neg_samples_test']),
             '* (bool) only_nice_volumes: %s' % str(self.stats['bool_only_nice_volumes']),
             '* (bool) exclude_inactive_roi: %s' % str(self.stats['bool_exclude_inactive_roi']),
             '* (bool) balanced_train_set: %s' % str(self.stats['bool_balanced_train_set']),
             '* (bool) balanced_test_set: %s' % str(self.stats['bool_balanced_test_set']),
             '* Calcium channel: %s' % str(self.stats['channel']),
             '* number of features computed per channel %s' % str(self.stats['n_features_per_channel'])
        ]

    def plot_learning(self, out_path):
        print "--> saving ROC curve"
        probs = self.model.predict_proba(self.X_test)
        fpr, tpr, _ = roc_curve(self.Y_test, probs[:, 1], pos_label=1)
        roc_auc = roc_auc_score(self.Y_test, probs[:, 1])
        self.stats['fpr'] = list(fpr)
        self.stats['tpr'] = list(tpr)
        self.stats['roc_auc'] = float(roc_auc)

        #tag = str(self.model)
        #print fpr.shape, tpr.shape
        #print fpr, tpr

        fig = plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %s)' % str(roc_auc)[:6])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Random forest predictions (pos. class) (ROC)')
        #plt.suptitle(tag)
        plt.legend(loc="lower right")
        plt.savefig(out_path + 'plot_random_forest_prob_maps_roc.png')
        plt.close(fig)

        print "--> saving PR curve"
        precision, recall, _ = precision_recall_curve(self.Y_test, probs[:, 1], pos_label=1)
        self.stats['precision'] = list(precision)
        self.stats['recall'] = list(recall)
        #print precision, recall
        fig2 = plt.figure()
        plt.plot(precision, recall, label='PR curve')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR curve on RF prob. maps')
        #plt.suptitle(tag)
        plt.savefig(out_path + 'plot_random_forest_prob_maps_pr.png')
        plt.close(fig2)

def score_random_forests(test_different_labelings_rf_dir):
    rf_test_folders = [dir for dir in os.listdir(test_different_labelings_rf_dir) if dir.startswith('rf')]
    scores = {}
    with open(os.path.join(test_different_labelings_rf_dir, 'random_forest_scores.txt')) as scores_out:
        for test in rf_test_folders:
            print 'collect underpants for:',  test
            forest_path = os.path.join(test_different_labelings_rf_dir, test, 'random_forest.pickle')
            train_test_path = os.path.join(test_different_labelings_rf_dir, test, 'train_test_splits.h5')
            rf_pipeline = RandomForestPipeline.load_model(pickled_forest_path=forest_path, h5_train_test_path=train_test_path)
            print 'compute score on test set...'
            score = rf_pipeline.test_set_score()
            print score
            scores_out.write('%s\t%.05f\n' % (str(test), float(score)))
            scores_out.flush()
            scores[test] = score
    print scores
    print 'save scores to', test_different_labelings_rf_dir
    scores_out_path = os.path.join(test_different_labelings_rf_dir, 'random_forest_scores.json')
    assert not os.path.isfile(scores_out_path), IOError('Output file already exists: %s' % scores_out_path)
    with open(scores_out_path, 'w') as json_out:
        json_out.write(json.dumps(scores, indent=0, sort_keys=True))
    print 'Done.'

class PixelwiseSSVMPipeline(ClassifierPipeline):

    crf_graph = None

    @classmethod
    def init_model_from_scratch(cls, out_dir, calcium_data=None, test_size=.75, feats_unary='feats_pm',
                                feats_pairwise='feats_xcorr_green', which_gt='gt', test_gt=None, dump_ssvm=True, which_ssvm='nslack',
                                ssvm_c=0.3, ssvm_tol=0.001, ssvm_iter=30, which_solver=('ogm', {'alg': 'gc'}),
                                no_class_weights=False, dummy_data=False, only_nice_volumes=False, verbosity=0):
        pipeline_obj = cls()
        pipeline_obj._start(out_dir, calcium_data=calcium_data, test_size=test_size, feats_unary=feats_unary,
                            feats_pairwise=feats_pairwise, which_gt=which_gt, test_gt=test_gt, dump_ssvm=dump_ssvm,
                            which_ssvm=which_ssvm, ssvm_c=ssvm_c, ssvm_tol=ssvm_tol, ssvm_iter=ssvm_iter, which_solver=which_solver,
                            verbosity=verbosity, no_class_weights=no_class_weights, dummy_data=dummy_data, only_nice_volumes=only_nice_volumes)
        return pipeline_obj

    def _start(self, out_dir, calcium_data=None, test_size=.75, feats_unary='feats_pm', feats_pairwise='feats_xcorr_green',
               which_gt='gt', test_gt=None, dump_ssvm=False, which_ssvm='nslack', ssvm_c=0.3, ssvm_tol=0.001, ssvm_iter=30, verbosity=0,
               which_solver=('ogm', {'alg': 'gc'}), no_class_weights=False, dummy_data=False, only_nice_volumes=False):

        self.stats = {'feats_unary' : feats_unary,
                      'feats_pairwise' : feats_pairwise,
                      'which_gt' : which_gt,
                      'test_size' : test_size}

        print '===== CRF ====='
        print "* build edge graph"
        img_x, img_y = calcium_data.subvolumes[0].dims
        edges = create_2d_edge_graph(img_x, img_y)

        if only_nice_volumes:
            print "* choosing only nice volumes"
            subvolumes = [sub for sub in calcium_data.subvolumes if sub.is_nice()]
        else:
            subvolumes = calcium_data.subvolumes

        print "* load features and labels"
        X, Y = calcium_data._create_features_and_labels_ssvm(subvolumes, which_gt, feats_unary, feats_pairwise, edges, verbosity)

        if len(subvolumes) == 2:
            print '[info] only 2 volumes present in data, using one for testing the other for training'
            X_train, X_test, Y_train, Y_test = [X[0]], [X[1]], [Y[0]], [Y[1]]
        else:
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)

        if no_class_weights:
            print "* create graphical model"
            crf_graph = EdgeFeatureGraphCRF(inference_method=which_solver)
            self.class_weights_str = 'no'
        else:
            print "* calculate class weights: number of instances in X: %s" % len(X)
            class_weights = compute_class_weights(Y)
            print "* create graphical model"
            crf_graph = EdgeFeatureGraphCRF(inference_method=which_solver, class_weight=class_weights)
            self.class_weights_str = 'yes'

        print '===== SSVM ====='
        if dump_ssvm:
            print '* setting up logger'
            logger = pystruct_logger(out_dir + 'ssvm_model.logger', save_every=10)
        else:
            logger = None

        if which_ssvm == 'nslack':
            print "* creating NSlackSSVM"
            model = NSlackSSVM(crf_graph, verbose=verbosity, C=ssvm_c, max_iter=ssvm_iter, n_jobs=-1, tol=ssvm_tol, show_loss_every=1,
                              inactive_threshold=1e-3, inactive_window=10, batch_size=1000, logger=logger)
            ssvm_name = 'NSlackSSVM'

        if which_ssvm == 'oneslack':
            print "* creating OneSlackSSVM"
            model = OneSlackSSVM(crf_graph, verbose=verbosity, C=ssvm_c, max_iter=ssvm_iter, n_jobs=-1, tol=ssvm_tol,
                                show_loss_every=1, inactive_threshold=1e-3, inactive_window=10, logger=logger)
            ssvm_name = 'OneSlackSSVM'

        if which_ssvm == 'frankwolfe':
            print "* creating FrankWolfeSSVM"
            model = FrankWolfeSSVM(crf_graph, verbose=verbosity, C=ssvm_c, max_iter=ssvm_iter, tol=ssvm_tol, line_search=True,
                                  check_dual_every=10, do_averaging=True, sample_method='perm', random_state=None, logger=logger)
            ssvm_name = 'FrankWolfeSSVM'

        print '* fitting model...'
        model.fit(X_train, Y_train)

        print '* scoring model...'
        self.score = model.score(X_test, Y_test)

        print '* saving object properties'
        self.crf_graph = crf_graph
        self.model = model
        self.X = X
        self.Y = Y
        self.edges = edges
        self.X_train, self.X_test, self.Y_train, self.Y_test = X_train, X_test, Y_train, Y_test
        self.img_x, self.img_y = img_x, img_y
        self.subvolumes = subvolumes
        self.ssvm_name = ssvm_name

        print '--> model trained'

    def predict_on_subvolume(self, subvolume, channel='green', verbosity=0):
        assert issubclass(type(subvolume), Subvolume), AssertionError('specify a valid Subvolume subclass instance')
        unary_feats = subvolume.get(self.stats['feats_unary'])
        pairwise_feats = subvolume.get(self.stats['feats_pairwise'])
        x = (unary_feats, self.edges, pairwise_feats)
        prediction = self.model.predict([x])
        return prediction[0].reshape(subvolume.dims)

    def score_subvolume(self, subvolume):
        assert issubclass(type(subvolume), Subvolume), AssertionError('specify a valid Subvolume subclass instance')
        unary_feats = subvolume.get(self.stats['feats_unary'])
        pairwise_feats = subvolume.get(self.stats['feats_pairwise'])
        x = (unary_feats, self.edges, pairwise_feats)
        gt = subvolume.get(self.stats['which_gt']).astype('uint8')
        y = gt.reshape(subvolume.dims[0] * subvolume.dims[1])
        return self.model.score([x], [y])

    def write_summary(self, out_path):
        with open(out_path + 'summary.txt', 'w') as summary_file:
            summary_file.write('\n=== Results ===\n')
            summary_file.write('Score on test set: %s\n' % str(self.score))
            summary_file.write('Test set size: %s\n' % str(self.stats['test_size']))
            summary_file.write('Number of volumes used: %s\n' % str(len(self.subvolumes)))
            summary_file.write('Number of overall samples (pixels): %d\n' % int(len(self.X)*self.img_x*self.img_y))
            summary_file.write('Number of positive samples in train/test set: %d/%d\n' % (int(np.sum(self.Y_train)), int(np.sum(self.Y_test))))
            summary_file.write('Number of negative samples in train/test set: %d/%d\n' %
                               (int(self.img_x*self.img_y - np.sum(self.Y_train)), int(self.img_x*self.img_y - np.sum(self.Y_test))))
            summary_file.write('\n=== CRF ===\n')
            summary_file.write('Unary features: %s\n' % self.stats['feats_unary'])
            summary_file.write('Pairwise features: %s\n' % self.stats['feats_pairwise'])
            summary_file.write('\n=== SSVM ===\n')
            summary_file.write('Ground truth: %s\n' % self.stats['which_gt'])
            summary_file.write('Class weights: %s\n' % self.class_weights_str)
            summary_file.write('Pystruct settings:\n')
            params = self.model.get_params()
            for k,v in params.iteritems():
                summary_file.write('    %s: %s\n' % (str(k), str(v)))
            summary_file.write('\n=== Data ===\n')
            for sub in self.subvolumes:
                stats, n_rois = sub.get_roi_activity_label_counts()
                summary_file.write('%s\t#rois: %3d\tactivity: ++ %3d\t+ %3d\tx %3d\t- %3d\t-- %3d\n' %
                                   (sub.name, n_rois, stats[0], stats[1], stats[2], stats[3], stats[4]))

    def write_scores(self, out_path, scores_dict):
        with open(out_path + 'subvolume_scores.txt', 'w') as out:
            for k,v in scores_dict.iteritems():
                    out.write('%s\t%s\n' % (str(k), str(v)))

    def plot_learning(self, time=True):
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
        """
        assert self.model is not None, RuntimeWarning('Train pipeline first.')

        ssvm = self.model
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

            axes[1].set_title("Training Error")
            axes[1].set_yscale('log')
        return fig, axes

if __name__ == "__main__":
    import IPython
    sys.exit(IPython.embed())