__version__ = '0.6'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from core.learning import PixelwiseSSVMPipeline
from core.data.calcium_imaging import JaneliaData
from core.data.rois import JaneliaRoi as Roi
from core.methods import create_2d_edge_graph, create_out_path
from core.visuals import ssvm_plot_learning, plot_it_like_ferran
from pystruct.utils import SaveLogger as pystruct_logger
from pystruct.models import EdgeFeatureGraphCRF
from pystruct.learners import NSlackSSVM, FrankWolfeSSVM, OneSlackSSVM
from scipy import ndimage
import json
import h5py as h5
import os
from os.path import join as join_path
from core.methods import compute_class_weights
from sklearn.cross_validation import ShuffleSplit
import cPickle as pickle

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
        unaries = sub.get(feats_unary)
        edge_features = sub.get(feats_pairwise)
        if feats_pairwise == 'feats_xcorr_green':
            edge_features = 1 - edge_features # remove negativity in xcorr
        # An instance ``x_i`` is represented as a tuple ``(node_features, edges, edge_features)``
        x = (unaries, edges, edge_features)
        X.append(x)
    return X

def test_different_labelings_ssvm(out_dir, pm_names, except_on_exist=False):
    def exclude_pm(pm):
        bad_test_sizes = ['0.90', '0.50', '0.10']
        if pm.split('test-size=')[1] in bad_test_sizes:
            return True
        else:
            return False

    jd = JaneliaData()

    out_path = out_dir + 'test_different_labelings_ssvm/'
    create_out_path(out_path=out_path, except_on_exist=except_on_exist)

    labelings = ['gt_active_++', 'gt_active_+++', 'gt_active_+++x']
    test_sizes = [.75, .25]
    ssvm_c = [5, .3, .1, .01, .001]
    failed_combos = []
    for active_gt in labelings:
        for feats_unary in pm_names:
            if exclude_pm(feats_unary): continue # testing all just takes too long (>1300h)
            for test_size in test_sizes:
                for c in ssvm_c:
                    out_name = 'ssvm_gt=%s_unaries=%s_test-size=%.02f_c=%.1e/' % (active_gt, feats_unary, test_size, c)
                    out_dir = out_path + out_name
                    try:
                        create_out_path(out_dir, except_on_exist=True)
                    except IOError:
                        print "SKIPPING combo '%s', already exists." % out_name
                        continue
                    print '===== Starting pixelwise SSVM pipeline  ====='
                    print '(%s)' % out_dir
                    try:
                        ssvm_baseline_pipe = PixelwiseSSVMPipeline.\
                             init_model_from_scratch(out_dir=out_dir, calcium_data=jd, test_size=test_size,
                                                     feats_unary=feats_unary, feats_pairwise='feats_xcorr_green',
                                                     which_gt=active_gt, ssvm_iter=50, dump_ssvm=True, ssvm_c=c,
                                                     only_nice_volumes=True)
                    except:
                        print 'Combo %s failed, removing its out folder.' % out_name
                        failed_combos.append(out_name)
                        os.rmdir(out_dir)
                        continue

                    print "* saving output"
                    fig, _ = ssvm_baseline_pipe.plot_learning(time=False)
                    fig.savefig(out_dir + 'ssvm_plot_learning.png', dpi=300)
                    plt.close('all')
                    ssvm_baseline_pipe.write_summary(out_dir)

                    print "* predicting on subvolumes and computing scores..."
                    sub_scores = {}
                    out_dir_sub = out_dir + 'predictions/'
                    create_out_path(out_dir_sub, except_on_exist=True)

                    for sub in jd.subvolumes:
                        print '  -', sub.name
                        pred = ssvm_baseline_pipe.predict_on_subvolume(sub)
                        gt = sub.get(active_gt).astype('uint8')
                        rgb_vis = plot_it_like_ferran(gt, pred)
                        plt.imsave(out_dir_sub + '%s_ssvm_prediction.png' % sub.name, pred)
                        plt.imsave(out_dir_sub + '%s_%s.png' % (sub.name, active_gt), gt)
                        plt.imsave(out_dir_sub + '%s_rgb_vis.png' % sub.name, rgb_vis)
                        sub_scores[sub.name] = ssvm_baseline_pipe.score_subvolume(sub)

                    print "* saving subvolume scores"
                    ssvm_baseline_pipe.write_scores(out_dir, sub_scores)
    print 'There were issues, the following combos got skipped:'
    for combo in failed_combos:
        print combo
    print '\nDone.'



def test_prediction_activity(out_dir, calcium_data, logged_ssvm_path, which_gt='gt', feats_unary='feats_pm', feats_pairwise='feats_xcorr_green'):

        img_x, img_y = 512, 512

        logger = pystruct_logger(logged_ssvm_path)
        ssvm = logger.load()
        print ssvm

        print "Build edge graph"
        edges = create_2d_edge_graph(img_x, img_y)

        for sub in calcium_data.subvolumes:

            print 'Preparing output directory for', sub.name
            out_folder = 'test_prediction_activity/%s/' % sub.name
            create_out_path(out_path=out_dir, except_on_exist=False)

            print 'Loading stuff for subvolume', sub.name
            gt = sub.get(which_gt).astype('uint8')
            unary_feats = sub.get(feats_unary)
            pairwise_feats = sub.get(feats_pairwise)
            x_single = (unary_feats, edges, pairwise_feats)
            y_single = gt.reshape(img_x * img_y)

            print "\tPredicting on subvolume"
            pred=ssvm.predict([x_single])
            pred=pred[0].reshape(img_x, img_y)

            print '\tScoring subvolume prediction'
            score = ssvm.score([x_single], [y_single])
            print 'Score on subvolume:', score

            print 'Getting connected components'
            cc_pred, cnt_obj_pred = ndimage.label(pred)
            cc_gt, cnt_obj_gt = ndimage.label(gt)

            print 'cc:', cnt_obj_pred

            print 'Saving vizzzz'
            for cc in xrange(1, cnt_obj_pred + 1):
                label_indices_xy = np.where(cc_pred == cc)
                tmp = np.zeros_like(gt)
                tmp[label_indices_xy] = 1
                tmp = tmp.reshape(img_x * img_y)
                label_indices_rshp = np.where(tmp == 1)
                label_indices_xy = np.asarray(label_indices_xy)
                label_polygons_pppx = sub.mpl_get_polygon_patches(['active_very_certain', 'active_mod_certain', 'uncertain'])
                label_polygons_ppp = sub.mpl_get_polygon_patches(['active_very_certain', 'active_mod_certain'])
                label_polygons_pp = sub.mpl_get_polygon_patches(['active_very_certain'])
                label_polygons = [label_polygons_pp, label_polygons_ppp, label_polygons_pppx]
                roi = Roi(indices=label_indices_rshp, xy=label_indices_xy, subvolume=sub, janelia_id=cc,
                          meta_info='Spike trains of predicted neurons for %s (score: %.02f)' % (sub.name, score))
                fig = roi.plot_like_annotation_labeler(prediction_image=pred, label_polygons=label_polygons)
                plt.savefig(out_dir + '%d.png' % roi.id, dpi=600)
                plt.close()

            plt.imsave(out_dir + 'cc_pred.png', cc_pred)
            plt.imsave(out_dir + 'cc_gt.png', cc_gt)
            print '\n============================================\n'
        print 'Done.'

def test_different_c_and_iter(out_dir, feats_unary, feats_pairwise, replace_stuff_on_disk=False, which_gt='gt'):

    out_path = out_dir + 'test_different_c_and_iter/'
    create_out_path(out_path)

    cd = JaneliaData()
    img_x, img_y = cd.subvolumes[0].dims

    ssvm_pipe = PixelwiseSSVMPipeline.init_model_from_scratch(out_dir=out_path, calcium_data=cd, which_gt=which_gt, feats_unary=feats_unary,
                                                          feats_pairwise=feats_pairwise)

    ssvm = ssvm_pipe.ssvm

    X_train, Y_train = ssvm_pipe.X_train, ssvm_pipe.Y_train

    C = [5, .1, .01, .001, .0001]

    for val in C:
        for iter in xrange(2, ssvm.max_iter):
            print 'Setting C parameter to %.4f' % val
            ssvm.C = val

            print "Setting max_iter parameter to %d" % iter
            ssvm.max_iter=iter

            # reset pystruct's internal list with loss values [BUG?!]
            ssvm.loss_curve_= []

            print "Fitting %s..." % ssvm_pipe.ssvm_name
            ssvm.fit(X_train, Y_train)

            print "Predicting on test volume (an229719_2013_12_02_06003)"
            sub_test = cd.get_subvolume_by_name('an229719_2013_12_02_06003')
            unaries_test = sub_test.get(feats_unary)
            pairwise_test = sub_test.get(feats_pairwise)
            x_test = (unaries_test, ssvm_pipe.edges, pairwise_test)
            y_test = sub_test.get(which_gt).astype('uint8').reshape(img_x * img_y)
            p_test = ssvm.predict([x_test])

            print "Plotting results..."
            fig=plt.figure()
            title = "%s,C=%.1e,max_iter=%d,score=%.4f,loss=%.4f,solver=%s" \
                    % (ssvm_pipe.ssvm_name, val, iter, ssvm.score([x_test], [y_test]), ssvm.loss_curve_[-1], which_solver)
            fig.suptitle(title)

            ax1=fig.add_subplot(221)
            ax1.set_title('primal objective')
            ax1.plot(ssvm.primal_objective_curve_)
            ax2=fig.add_subplot(222)
            ax2.set_title('loss')
            ax2.plot(ssvm.loss_curve_)
            ax3=fig.add_subplot(223)
            ax3.set_title('prediction')
            ax3.set_xticks([])
            ax3.set_yticks([])
            ax3=sub_test.plot_activity_labels(image=p_test[0].reshape(img_x, img_y),
                                                labels=['active_very_certain', 'active_mod_certain', 'uncertain'],
                                                ax=ax3)
            ax4=fig.add_subplot(224)
            ax4.set_title('ground truth')
            ax4.set_xticks([])
            ax4.set_yticks([])
            ax4 = sub_test.plot_labeled_rois(ax=ax4)
            fig.savefig(out_dir + 'test_different_c_' + title + '.png')
            plt.close('all')
    print '\nDone.'


def test_final_segmentations(out_dir):
    test_size = .75
    #ssvm_c = [5, .3, .1, .01, .001]
    ssvm_c = [50]
    ssvm_tol = 0.001
    ssvm_iter = 50
    which_ssvm = 'nslack'
    which_solver = ('ogm', {'alg': 'gc'})
    which_combo = which_ssvm + '_ogm-' + which_solver[1]['alg']

    out_dir = join_path(out_dir, 'test_final_segmentations', which_combo)
    create_out_path(out_dir, except_on_exist=False)
    cd = JaneliaData(dummy_data=False, only_nice_volumes=True)
    subs = cd.subvolumes
    demo_volumes = ['an197522_2013_03_10_13002', 'an197522_2013_03_08_06003', 'an229719_2013_12_05_03004', 'an229719_2013_12_05_07003', 'an229719_2013_12_05_08004']

    img_x, img_y = cd.subvolumes[0].dims
    edges = create_2d_edge_graph(img_x, img_y)

    feats_unaries = ['feats_pm_++', 'feats_pm_+++', 'feats_pm_+++x']
    feats_pairwise = 'feats_xcorr_green'

    train_labelings = ['gt_active_++', 'gt_active_+++', 'gt_active_+++x']
    test_labelings = ['gt_active_++', 'gt_active_+++', 'gt_active_+++x']
    failed_combos = []

    sp = ShuffleSplit(len(subs), random_state=42, test_size=test_size)
    train_idxs, test_idxs = next(iter(sp))

    for feats_unary in feats_unaries:
        X = ssvm_construct_X(subs, feats_unary, feats_pairwise, edges)

        for train_labeling in train_labelings:
            Y_full_train = ssvm_construct_Y(train_labeling, subs)
            class_weights = compute_class_weights(Y_full_train)
            crf_graph = EdgeFeatureGraphCRF(inference_method=which_solver, class_weight=class_weights)

            for test_labeling in test_labelings:
                X_train = [X[i] for i in train_idxs]
                Y_train = [Y_full_train[i] for i in train_idxs]
                Y_full_test = ssvm_construct_Y(test_labeling, subs)

                X_test = [X[i] for i in test_idxs]
                Y_test = [Y_full_test[i] for i in test_idxs]

                for c in ssvm_c:
                    try:
                        test_name = '%s_u=%s_p=%s_train-gt=%s_test-gt=%s_C=%.1e' % (which_ssvm, feats_unary, feats_pairwise, train_labeling, test_labeling, c)
                        test_out_dir = join_path(out_dir, test_name)
                        create_out_path(test_out_dir, except_on_exist=True)

                        print '\n=================================================================='
                        print '-->' + test_name
                    except IOError as e:
                        print e
                        continue

                    try:
                        logger = pystruct_logger(join_path(test_out_dir, 'ssvm_model.logger'), save_every=10)

                        if which_ssvm == 'nslack':
                            print "* creating NSlackSSVM"
                            model = NSlackSSVM(crf_graph, verbose=1, C=c, max_iter=ssvm_iter, n_jobs=-1, tol=ssvm_tol, show_loss_every=1,
                                              inactive_threshold=1e-3, inactive_window=10, batch_size=1000, logger=logger)
                            ssvm_name = 'NSlackSSVM'

                        if which_ssvm == 'oneslack':
                            print "* creating OneSlackSSVM"
                            model = OneSlackSSVM(crf_graph, verbose=1, C=c, max_iter=ssvm_iter, n_jobs=-1, tol=ssvm_tol,
                                                show_loss_every=1, inactive_threshold=1e-3, inactive_window=10, logger=logger)
                            ssvm_name = 'OneSlackSSVM'

                        if which_ssvm == 'frankwolfe':
                            print "* creating FrankWolfeSSVM"
                            model = FrankWolfeSSVM(crf_graph, verbose=1, C=c, max_iter=ssvm_iter, tol=ssvm_tol, line_search=True,
                                                  check_dual_every=10, do_averaging=True, sample_method='perm', random_state=None, logger=logger)
                            ssvm_name = 'FrankWolfeSSVM'

                        print '* fitting model...'
                        model.fit(X_train, Y_train)
                        print '--> model trained'

                        print '* scoring model...'
                        score = model.score(X_test, Y_test)
                        print score

                        print '* saving results for test run'
                        fig, _ = ssvm_plot_learning(model, title=test_name, time=False)
                        fig.savefig(join_path(test_out_dir, 'ssvm_plot_learning.png'), dpi=300)
                        plt.close('all')
                        print '--> saved learning plots'

                        ### save results to JSON ###
                        with open(join_path(test_out_dir, 'ssvm_stats.json'), 'w') as json_out:
                            stats = {
                                'test_accuracy' : float(score),
                                'test_size' : float(test_size),
                                'ssvm_c' : float(c),
                                'ssvm_iter' : ssvm_iter,
                                'ssvm_tol' : float(ssvm_tol),
                                'feats_unary' : feats_unary,
                                'feats_pairwise' : feats_pairwise,
                                'which_ssvm' : which_ssvm,
                                'which_solver' : which_solver[0],
                                'which_alg' : which_solver[1]['alg'],
                                'train_labeling' : train_labeling,
                                'test_labeling' : test_labeling
                            }
                            json_out.write(json.dumps(stats, indent=0, sort_keys=True))
                        print '--> saved SSVM stats to JSON'

                    except:
                        print 'Combo %s failed, removing its out folder.' % test_out_dir
                        failed_combos.append(test_out_dir)
                        os.rmdir(test_out_dir)
                        create_out_path('FAILED_' + test_out_dir)
                        continue

                    print '--> saving vizzz for demo volumes and computing scores...'
                    demo_out_dir = join_path(test_out_dir, 'demo_vols')
                    create_out_path(demo_out_dir, except_on_exist=True)
                    demo_scores = {}
                    for demo_sub in demo_volumes:
                        demo_sub = cd.get_subvolume_by_name(demo_sub)
                        print '    -', demo_sub.name
                        unary_feats = demo_sub.get(feats_unary)
                        pairwise_feats = demo_sub.get(feats_pairwise)
                        x = (unary_feats, edges, pairwise_feats)
                        pred = model.predict([x])
                        pred = pred[0].reshape(img_x, img_y)

                        gt_train = demo_sub.get(train_labeling).astype('uint8')
                        gt_test = demo_sub.get(test_labeling).astype('uint8')
                        rgb_vis_train = plot_it_like_ferran(gt_train, pred)
                        rgb_vis_test = plot_it_like_ferran(gt_test, pred)

                        plt.imsave(join_path(demo_out_dir, '%s_ssvm_prediction.png' % demo_sub.name), pred)
                        plt.imsave(join_path(demo_out_dir, '%s_%s.png' % (demo_sub.name, train_labeling)), gt_train)
                        plt.imsave(join_path(demo_out_dir, '%s_%s.png' % (demo_sub.name, test_labeling)), gt_test)
                        plt.imsave(join_path(demo_out_dir, '%s_rgb_vis_train.png' % demo_sub.name), rgb_vis_train)
                        plt.imsave(join_path(demo_out_dir, '%s_rgb_vis_test.png' % demo_sub.name), rgb_vis_test)

                        labeled_pm_ax = demo_sub.plot_activity_labels(image=pred, cmap='gray')
                        plt.savefig(join_path(demo_out_dir, '%s_labeled_pred.png' % demo_sub.name), dpi=300)
                        plt.close('all')

                        score_train = model.score([x], [gt_train.reshape(img_x * img_y)])
                        score_test = model.score([x], [gt_test.reshape(img_x * img_y)])
                        scores = {
                            'accuracy_train' : float(score_train),
                            'accuracy_test' : float(score_test)
                        }
                        demo_scores[demo_sub.name] = scores

                    with open(join_path(test_out_dir, 'demo_vol_scores.json'), 'w') as json_out:
                        json_out.write(json.dumps(demo_scores, indent=0, sort_keys=True))
                    print '--> saved demo volume scores to JSON'

    print '\nDone.'

    print 'There were issues, the following combos got skipped:'
    for combo in failed_combos:
        print combo

def test_cherry_segmentations(out_dir):
    out_dir = os.path.join(out_dir, 'test_cherry_segmentations')
    test_size = .8
    ssvm_c = [100, 50, 5, .3, .1, .01, .001, .0001]
    ssvm_tol = -10
    ssvm_iter = 50
    ssvms = ['nslack', 'oneslack']
    which_solver = ('ogm', {'alg': 'gc'})
    #which_solver = 'ad3'
    cd = JaneliaData()
    subs = cd.ssvm_data_cherry_picked


    feats_unaries = ['feats_pm_setA_green', 'feats_pm_setJ_green', 'feats_pm_setD_green', 'feats_pm_setE_green']
    feats_pairwise = 'feats_xcorr_green'

    train_labelings = ['gt_active_++', 'gt_active_+++', 'gt_active_+++x']
    failed_combos = []

    sp = ShuffleSplit(len(subs), random_state=1, test_size=test_size)
    train_idxs, test_idxs = next(iter(sp))
    for which_ssvm in ssvms:
        if isinstance(which_solver, tuple):
            which_combo = which_ssvm + '_ogm-' + which_solver[1]['alg']
        else:
            which_combo = which_ssvm + '_' + which_solver
        combo_out_dir = join_path(out_dir, which_combo)
        create_out_path(combo_out_dir, except_on_exist=False)

        for feats_unary in feats_unaries:
            X = [_sub.ssvm_create_x(feats_unary, feats_pairwise, bias_u=False, bias_p=False) for _sub in subs]
            X_train = [X[i] for i in train_idxs]

            for train_labeling in train_labelings:
                Y_full_train = [_sub.ssvm_create_y(train_labeling) for _sub in subs]
                Y_train = [Y_full_train[i] for i in train_idxs]
                class_weights = compute_class_weights(Y_train)
                crf_graph = EdgeFeatureGraphCRF(inference_method=which_solver, class_weight=class_weights)

                for c in ssvm_c:
                    try:
                        test_name = '%s_u=%s_p=%s_train-gt=%s_C=%.1e' % (which_ssvm, feats_unary, feats_pairwise, train_labeling, c)
                        test_out_dir = join_path(combo_out_dir, test_name)
                        create_out_path(test_out_dir, except_on_exist=True)
                        print '\n=================================================================='
                        print test_name
                        print '\n=================================================================='
                    except IOError as e: # omit already computed results on script re-run
                        print e
                        continue

                    try:
                        logger = pystruct_logger(join_path(test_out_dir, 'ssvm_model.logger'), save_every=10)

                        if which_ssvm == 'nslack':
                            print "* creating NSlackSSVM"
                            model = NSlackSSVM(crf_graph, verbose=1, C=c, max_iter=ssvm_iter, n_jobs=-1, tol=ssvm_tol, show_loss_every=1,
                                               inactive_threshold=1e-3, inactive_window=10, batch_size=10000, logger=logger)
                            ssvm_name = 'NSlackSSVM'

                        if which_ssvm == 'oneslack':
                            print "* creating OneSlackSSVM"
                            model = OneSlackSSVM(crf_graph, verbose=1, C=c, max_iter=ssvm_iter, n_jobs=-1, tol=ssvm_tol,
                                                 show_loss_every=1, inactive_threshold=1e-3, inactive_window=10, logger=logger)
                            ssvm_name = 'OneSlackSSVM'

                        # if which_ssvm == 'frankwolfe':
                        #     print "* creating FrankWolfeSSVM"
                        #     model = FrankWolfeSSVM(crf_graph, verbose=1, C=c, max_iter=ssvm_iter, tol=ssvm_tol, line_search=True,
                        #                            check_dual_every=10, do_averaging=True, sample_method='perm', random_state=None, logger=logger)
                        #     ssvm_name = 'FrankWolfeSSVM'

                        print '* fitting model...'
                        model.fit(X_train, Y_train)
                    except:
                        print 'Combo %s failed, removing its out folder.' % test_out_dir
                        failed_combos.append(test_out_dir)
                        os.rmdir(test_out_dir)
                        create_out_path(os.path.join('%s_FAILED'%test_out_dir))
                        continue
                    print '* saving learning curves for current C'
                    fig, _ = ssvm_plot_learning(model, title=test_name, time=False)
                    fig.savefig(join_path(test_out_dir, 'ssvm_plot_learning.png'))
                    plt.close('all')
                    print '--> saved learning plots'
                    ### save results to JSON ###
                    with open(join_path(test_out_dir, 'ssvm_stats.json'), 'w') as json_out:
                        stats = {
                                'test_size' : float(test_size),
                                'ssvm_c' : float(c),
                                'ssvm_iter' : ssvm_iter,
                                'ssvm_tol' : float(ssvm_tol),
                                'feats_unary' : feats_unary,
                                'feats_pairwise' : feats_pairwise,
                                'which_ssvm' : which_ssvm,
                                'which_solver' : which_solver[0],
                                'which_alg' : which_solver[1]['alg'],
                                'train_labeling' : train_labeling,
                                'loss_curve' : model.loss_curve_,
                                'objective_curve' : model.objective_curve_,
                                'primal_objective_curve' : model.primal_objective_curve_,
                                'learned_weights' : list(model.w)
                            }
                        json_out.write(json.dumps(stats, indent=0, sort_keys=False))
                    print '--> saved SSVM stats to JSON'

    print 'There were issues, the following combos got skipped:'
    for combo in failed_combos:
        print combo
    pickle.dump(failed_combos, open(os.path.join(out_dir, 'failed_combos.p'), 'w'))
    print '\nDone.'

def test_crazy_segmentations(out_dir):
    out_dir = os.path.join(out_dir, 'test_crazy_segmentations')
    test_size = .8
    ssvm_c = [5, .3, .0001]
    ssvm_tol = -10
    ssvm_iter = 30
    ssvms = ['nslack', 'oneslack']
    which_solver = ('ogm', {'alg': 'gc'})
    #which_solver = 'ad3'
    cd = JaneliaData()
    subs = cd.ssvm_data_cherry_picked


    feats_unaries = ['feats_pm_setA_green', 'feats_pm_setJ_green', 'feats_pm_setD_green', 'feats_pm_setE_green', 'feats_pm_setA_red', 'feats_pm_setJ_red', 'feats_pm_setD_red', 'feats_pm_setE_red']
    feats_pairwise = ['feats_xcorr_green']

    train_labelings = ['gt_active_++', 'gt_active_+++', 'gt_active_+++x']
    failed_combos = []

    sp = ShuffleSplit(len(subs), random_state=1, test_size=test_size)
    train_idxs, test_idxs = next(iter(sp))
    for which_ssvm in ssvms:
        if isinstance(which_solver, tuple):
            which_combo = which_ssvm + '_ogm-' + which_solver[1]['alg']
        else:
            which_combo = which_ssvm + '_' + which_solver
        combo_out_dir = join_path(out_dir, which_combo)
        create_out_path(combo_out_dir, except_on_exist=False)

        X = [_sub.ssvm_create_x(feats_unaries, feats_pairwise, bias_u=False, bias_p=False) for _sub in subs]
        X_train = [X[i] for i in train_idxs]

        for train_labeling in train_labelings:
            Y_full_train = [_sub.ssvm_create_y(train_labeling) for _sub in subs]
            Y_train = [Y_full_train[i] for i in train_idxs]
            class_weights = compute_class_weights(Y_train)
            crf_graph = EdgeFeatureGraphCRF(inference_method=which_solver, class_weight=class_weights)

            for c in ssvm_c:
                try:
                    test_name = '%s_u=%s_p=%s_train-gt=%s_C=%.1e' % (which_ssvm, 'AJDE_gr', feats_pairwise[0], train_labeling, c)
                    test_out_dir = join_path(combo_out_dir, test_name)
                    create_out_path(test_out_dir, except_on_exist=True)
                    print '\n=================================================================='
                    print test_name
                    print '\n=================================================================='
                except IOError as e: # omit already computed results on script re-run
                    print e
                    continue

                try:
                    logger = pystruct_logger(join_path(test_out_dir, 'ssvm_model.logger'), save_every=10)

                    if which_ssvm == 'nslack':
                        print "* creating NSlackSSVM"
                        model = NSlackSSVM(crf_graph, verbose=1, C=c, max_iter=ssvm_iter, n_jobs=-1, tol=ssvm_tol, show_loss_every=1,
                                           inactive_threshold=1e-3, inactive_window=10, batch_size=1000, logger=logger)
                        ssvm_name = 'NSlackSSVM'

                    if which_ssvm == 'oneslack':
                        print "* creating OneSlackSSVM"
                        model = OneSlackSSVM(crf_graph, verbose=1, C=c, max_iter=ssvm_iter, n_jobs=-1, tol=ssvm_tol,
                                             show_loss_every=1, inactive_threshold=1e-3, inactive_window=10, logger=logger)
                        ssvm_name = 'OneSlackSSVM'

                    print '* fitting model...'
                    model.fit(X_train, Y_train)
                except:
                    print 'Combo %s failed, removing its out folder.' % test_out_dir
                    failed_combos.append(test_out_dir)
                    os.rmdir(test_out_dir)
                    create_out_path(os.path.join('%s_FAILED'%test_out_dir))
                    continue
                print '* saving learning curves for current C'
                fig, _ = ssvm_plot_learning(model, title=test_name, time=False)
                fig.savefig(join_path(test_out_dir, 'ssvm_plot_learning.png'))
                plt.close('all')
                print '--> saved learning plots'
                ### save results to JSON ###
                with open(join_path(test_out_dir, 'ssvm_stats.json'), 'w') as json_out:
                    stats = {
                            'test_size' : float(test_size),
                            'ssvm_c' : float(c),
                            'ssvm_iter' : ssvm_iter,
                            'ssvm_tol' : float(ssvm_tol),
                            'feats_unary' : feats_unaries,
                            'feats_pairwise' : feats_pairwise,
                            'which_ssvm' : which_ssvm,
                            'which_solver' : which_solver[0],
                            'which_alg' : which_solver[1]['alg'],
                            'train_labeling' : train_labeling,
                            'loss_curve' : model.loss_curve_,
                            'objective_curve' : model.objective_curve_,
                            'primal_objective_curve' : model.primal_objective_curve_,
                            'learned_weights' : list(model.w)
                        }
                    json_out.write(json.dumps(stats, indent=0, sort_keys=False))
                print '--> saved SSVM stats to JSON'

    print 'There were issues, the following combos got skipped:'
    for combo in failed_combos:
        print combo
    pickle.dump(failed_combos, open(os.path.join(out_dir, 'failed_combos.p'), 'w'))
    print '\nDone.'

if __name__ == '__main__':
    import sys
    try:
        args = sys.argv[1:]
        out_dir = args[0]
        test = args[1]
        rest = args[2:]
    except:
        print 'specify <out_dir> <test>'
        sys.exit()

    if test == 'ssvm':
        with h5.File(out_dir + 'test_different_labelings_rf/generated_pms.h5')  as f:
            pm_names = list(f['pm_names'])
        test_different_labelings_ssvm(out_dir=out_dir, pm_names=pm_names)
    if test == 'ssvm_segment':
        test_final_segmentations(out_dir=out_dir)
    if test == 'ssvm_segment_cherries':
        test_cherry_segmentations(out_dir=out_dir)
    if test == 'ssvm_segment_crazy':
        test_crazy_segmentations(out_dir=out_dir)
