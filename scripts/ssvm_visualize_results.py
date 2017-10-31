import os
import sys
from IPython import embed
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
from core.data.calcium_imaging import JaneliaData
from core.methods import clear_ticks, create_2d_edge_graph
from core.visuals import plot_it_like_ferran, ssvm_plot_learning
from pystruct.utils.logging import SaveLogger as pystruct_logger
import IPython
sns.set(style='ticks', palette='muted', color_codes=True, context='notebook')

def process_results(test_final_segmentations_dir, demo_sub_name, ssvm_alg):
    demo_sub_name = '%s_rgb_vis_train.png'%demo_sub_name
    # collect data on disk
    underpants = []
    for root, _, files in os.walk(os.path.join(test_final_segmentations_dir, ssvm_alg), topdown=True):
        for name in files:
            underpants.append(os.path.join(root, name))
    model_paths = [path for path in underpants if path.endswith('ssvm_model.logger')]
    json_stat_paths = [path for path in underpants if path.endswith('ssvm_stats.json')]
    demo_sub_rgb_pred_paths = [path for path in underpants if path.endswith(demo_sub_name)]
    model_paths.sort()
    json_stat_paths.sort()
    demo_sub_rgb_pred_paths.sort()
    return model_paths, json_stat_paths, demo_sub_rgb_pred_paths

# Problem: too few data points
def plot_ssvm_score_vs_test_size(test_different_labelings_ssvm_dir):
        results = { # for three different label sets, aka pm++, pm+++, pm+++-
            'J' : None,
            'D' : None,
            'E' : None
        }

        # collect data on disk
        underpants = []
        for root, _, files in os.walk(test_different_labelings_ssvm_dir, topdown=False):
            for name in files:
                underpants.append(os.path.join(root, name))
        summaries = [path for path in underpants if path.endswith('summary.txt')]

        # extract relevant prob maps
        setJ = 'feats_test_pm_gt=gt_active_++_test-size=0.75'
        setD = 'feats_test_pm_gt=gt_active_+++_test-size=0.75'
        setE = 'feats_test_pm_gt=gt_active_+++x_test-size=0.75'
        c = '1.0e-02'
        sumJ = [sum for sum in summaries if sum.split('unaries=')[1].startswith(setJ)]
        sumD = [sum for sum in summaries if sum.split('unaries=')[1].startswith(setD)]
        sumE = [sum for sum in summaries if sum.split('unaries=')[1].startswith(setE)]

        sumJ = [sum for sum in sumJ if (sum.split('ssvm_gt=')[1].startswith('gt_active_++_') and sum.split('_c=')[1].startswith(c))]
        sumD = [sum for sum in sumD if (sum.split('ssvm_gt=')[1].startswith('gt_active_++_') and sum.split('_c=')[1].startswith(c))]
        sumE = [sum for sum in sumE if (sum.split('ssvm_gt=')[1].startswith('gt_active_++_') and sum.split('_c=')[1].startswith(c))]
        sumJ.sort()
        sumD.sort()
        sumE.sort()
        scoresJ = [float(open(f).readlines()[2].split('test set: ')[1].strip()) for f in sumJ]
        scoresD = [float(open(f).readlines()[2].split('test set: ')[1].strip()) for f in sumD]
        scoresE = [float(open(f).readlines()[2].split('test set: ')[1].strip()) for f in sumE]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        sizes = [25, 50, 75, 90]
        ax.plot(sizes, scoresJ, label='J')
        ax.plot(sizes[:-1], scoresE, label='E')
        ax.plot(sizes, scoresD, label='D')
        ax.set_xlabel('Test set size (percent)')
        ax.set_ylabel('$1-Loss$ (averaged)')
        ax.legend(loc=0)
        return fig


def convert_gtsymbols_to_numbers(png_paths, sort=True):
    png_paths = [path.replace('_++_', '_0_') for path in png_paths]
    png_paths = [path.replace('_+++_', '_1_') for path in png_paths]
    png_paths = [path.replace('_+++x_', '_2_') for path in png_paths]
    if sort: png_paths.sort()
    return png_paths


def convert_gtnumbers_to_symbols(png_paths):
    png_paths = [path.replace('_0_', '_++_') for path in png_paths]
    png_paths = [path.replace('_1_', '_+++_' ) for path in png_paths]
    png_paths = [path.replace('_2_', '_+++x_') for path in png_paths]
    return png_paths


def convert_gtnumber_to_symbol(path):
    path = path.replace('_0_', '_++_')
    path = path.replace('_1_', '_+++_' )
    path = path.replace('_2_', '_+++x_')
    return path


def convert_gtsymbol_to_number(path):
    path = path.replace('_++_', '_0_')
    path = path.replace('_+++_', '_1_')
    path = path.replace('_+++x_', '_2_')
    return path

#old
def plot_final_segmentations_for_demo_sub_from_PNG(test_final_segmentations_dir, demo_sub_name='an197522_2013_03_10_13002', ssvm_alg='nslack', C=0.3):
    _, _, demo_sub_rgb_pred_paths = process_results(test_final_segmentations_dir, demo_sub_name, ssvm_alg)
    png_paths = [path for path in demo_sub_rgb_pred_paths if ((float(path.split('_C=')[1].split('/')[0]) == C) and
                                                              path.split('_test-gt=')[1].startswith('gt_active_+++x_'))]
    png_paths = convert_gtsymbols_to_numbers(png_paths)
    fig =plt.figure(figsize=(8, 8))
    # fig.suptitle('C=%.03f'%C)
    #fig.subplots_adjust(bottom=0.025, left=0.025, top = 0.975, right=0.975)
    layout = [(3,3,1), (3,3,2), (3,3,3), (3,3,4), (3,3,5), (3,3,6), (3,3,7), (3,3,8), (3,3,9)]
    for nrows, ncols, plot_number in layout:
        ax = fig.add_subplot(nrows, ncols, plot_number)
        rgb_pred =  plt.imread(convert_gtnumbers_to_symbols(png_paths)[plot_number - 1])
        ax.imshow(rgb_pred)
        clear_ticks(ax)
    plt.tight_layout()
    return fig


def show_constraints_for_minimum_loss(model, train_gt):
    from operator import itemgetter
    min_loss = min(enumerate(model.loss_curve_), key=itemgetter(1))[0]
    min_constr = model.constraints_[min_loss]
    fig_ = plt.figure()
    fig_.suptitle('constraints of iter with lowest loss: %d (of %d)' % (min_loss, len(model.loss_curve_)))
    n_rows_cols = int(np.ceil(np.sqrt(len(min_constr))))
    for constr_i in xrange(len(min_constr)):
        ax_ = fig_.add_subplot(n_rows_cols, n_rows_cols, constr_i)
        constr = min_constr[constr_i][0].reshape(512,512)
        rgb_vis = plot_it_like_ferran(train_gt, constr)
        ax_.imshow(rgb_vis)
    clear_ticks(fig_.get_axes())
    return fig_


def show_constraints_for_loss_iter(model, iter, train_gt):
    min_constr = model.constraints_[iter]
    fig_ = plt.figure()
    fig_.suptitle('constraints of loss for iter: %d (of %d)' % (iter, len(model.loss_curve_)))
    n_rows_cols = int(np.ceil(np.sqrt(len(min_constr))))
    for constr_i in xrange(len(min_constr)):
        ax_ = fig_.add_subplot(n_rows_cols, n_rows_cols, constr_i)
        constr = min_constr[constr_i][0].reshape(512,512)
        rgb_vis = plot_it_like_ferran(train_gt, constr)
        ax_.imshow(rgb_vis)
    clear_ticks(fig_.get_axes())
    return fig_

#old
def plot_demo_sub_best_pred_with_different_gt(test_final_segmentations_dir, pred_i, subvolume, C, ssvm_alg):
    _, _, demo_sub_rgb_pred_paths = process_results(test_final_segmentations_dir, subvolume.name, ssvm_alg)
    png_paths = [path for path in demo_sub_rgb_pred_paths if ((float(path.split('_C=')[1].split('/')[0]) == C) and
                                                              path.split('_test-gt=')[1].startswith('gt_active_+++x_'))]
    png_paths = convert_gtsymbols_to_numbers(png_paths)

    # model
    model_path = convert_gtnumbers_to_symbols(png_paths)[pred_i]
    print 'Model:', model_path
    logger = pystruct_logger(model_path)
    model = logger.load()
    print model

    # features
    name_unary = model_path.split('_u=')[1].split('_p=')[0]
    print 'Prob map:', name_unary
    name_pairwise = 'feats_xcorr_green'

    print 'Predicting...'
    img_x, img_y = 512, 512
    edges = create_2d_edge_graph(img_x, img_y)
    unary_feats = subvolume.get(name_unary)
    pairwise_feats = subvolume.get(name_pairwise)
    x = (unary_feats, edges, pairwise_feats)
    pred = model.predict([x])
    pred = pred[0].reshape(img_x, img_y)

    gt0 = subvolume.get('gt_active_++').astype('uint8')
    gt1 = subvolume.get('gt_active_+++').astype('uint8')
    gt2 = subvolume.get('gt_active_+++x').astype('uint8')
    gtJ = subvolume.get('gt').astype('uint8')
    gts = [gt0, gt1, gt2, gtJ]
    fig = plt.figure(figsize=(8,8))

    for i in xrange(len(gts)):
        ax_ = fig.add_subplot(4, 1, i)
        rgb_vis = plot_it_like_ferran(gts[i], pred)
        ax_.imshow(rgb_vis)
    clear_ticks(fig.get_axes())
    return fig

#old
def plot_final_segmentations_for_demo_sub(test_final_segmentations_dir, subvolume,  ssvm_alg='nslack', C=5):
    img_x, img_y = 512, 512
    edges = create_2d_edge_graph(img_x, img_y)
    model_paths, _, _ = process_results(test_final_segmentations_dir, subvolume.name, ssvm_alg)
    try:
        png_paths = [path for path in model_paths if ((float(path.split('_C=')[1].split('/')[0]) == C) and
                                                      path.split('_test-gt=')[1].startswith('gt_active_++_'))]
        test_name = 'test_final_segmentations'
    except:
        png_paths = [path for path in model_paths if float(path.split('_C=')[1].split('/')[0]) == C]
        test_name = 'test_cherry_segmentations'
    png_paths = convert_gtsymbols_to_numbers(png_paths)
    fig =plt.figure(figsize=(8, 8))
    # fig.suptitle('C=%.03f'%C)
    #fig.subplots_adjust(bottom=0.025, left=0.025, top = 0.975, right=0.975)
    layout = [(3,3,1), (3,3,2), (3,3,3), (3,3,4), (3,3,5), (3,3,6), (3,3,7), (3,3,8), (3,3,9)]
    for nrows, ncols, plot_number in layout:
        print '\n'
        # model
        model_path = convert_gtnumbers_to_symbols(png_paths)[plot_number - 1]
        print 'Model:', model_path
        logger = pystruct_logger(model_path)
        model = logger.load()
        print model

        # features
        name_unary = model_path.split('_u=')[1].split('_p=')[0]
        print 'Prob map:', name_unary
        name_pairwise = 'feats_xcorr_green'

        # ground truth
        if test_name == 'test_final_segmentations':
            name_traingt = model_path.split('_train-gt=')[1].split('_test-gt=')[0]
        else: # test_cherry_segmentations
            name_traingt = model_path.split('_train-gt=')[1].split('_C')[0]
        print 'GT:', name_traingt

        print 'Predicting...'
        unary_feats = subvolume.get(name_unary)
        pairwise_feats = subvolume.get(name_pairwise)
        x = (unary_feats, edges, pairwise_feats)
        pred = model.predict([x])
        pred = pred[0].reshape(img_x, img_y)

        gt_train = subvolume.get(name_traingt).astype('uint8')
        rgb_vis = plot_it_like_ferran(gt_train, pred)

        ax = fig.add_subplot(nrows, ncols, plot_number)
        ax.imshow(rgb_vis)
        clear_ticks(ax)
        # fig_loss = show_constraints_for_minimum_loss(model, gt_train)
        # name = convert_gtsymbols_to_numbers(['out/min_loss_constraints/%s_%s_%s_%s.png'%(subvolume.name, name_unary, name_traingt, ssvm_alg)])[0]
        # for i in xrange(len(model.constraints_)):
        #     fig_iter = show_constraints_for_loss_iter(model, i, gt_train)
        #     name = convert_gtsymbols_to_numbers(['out/loss_constraints_iter/%s_%s_%s_%s_iter%d.png'%(subvolume.name, name_unary, name_traingt, ssvm_alg, i)])[0]
        #     fig_iter.savefig(name, dpi=300)
        # plt.close('all')
    plt.tight_layout()
    return fig


def plot_rgb_segmentations_for_demo_sub(model_paths, subvolume):
    print 'Plotting the following models for subvolume', subvolume.name
    for bla in model_paths: print ' ', bla

    img_x, img_y = subvolume.dims
    fig =plt.figure(figsize=(10, 13))
    fig.suptitle('%s')
    n_rows = len(model_paths)/3
    n_cols = 3
    cnt_plot = 0
    for i in xrange(len(model_paths)):
        cnt_plot += 1
        print '\n'
        # model
        model_path = convert_gtnumber_to_symbol(model_paths[i])
        logger = pystruct_logger(model_path)
        model = logger.load()
        print model
        # features
        name_unary = model_path.split('_u=')[1].split('_p=')[0]
        print name_unary
        if name_unary == 'AJDE_gr':
            feats_unary = ['feats_pm_setA_green', 'feats_pm_setJ_green', 'feats_pm_setD_green', 'feats_pm_setE_green',
                           'feats_pm_setA_red', 'feats_pm_setJ_red', 'feats_pm_setD_red', 'feats_pm_setE_red']
            name_unary = 'JADE-gr'
        else:
            feats_unary = name_unary
            name_unary = 'pm' + name_unary.split('_green')[0][-1]
        print 'Unaries:', name_unary
        name_pairwise = 'feats_xcorr_green'
        # ground truth
        name_traingt = model_path.split('_train-gt=')[1].split('_C')[0]
        print 'train GT:', name_traingt

        print 'Predicting...'
        x = subvolume.ssvm_create_x(feats_unary, name_pairwise, bias_u=False, bias_p=False)
        pred = model.predict([x])
        pred = pred[0].reshape(img_x, img_y)
        gt_train = subvolume.get(name_traingt).astype('uint8')
        rgb_vis = plot_it_like_ferran(gt_train, pred)
        ax = fig.add_subplot(n_rows, n_cols, cnt_plot)
        if cnt_plot == 1:
            ax.set_title('train gt: gt++', fontweight='bold', fontsize='medium', y=1.08)
        elif cnt_plot == 2:
            ax.set_title('train gt: gt+++', fontweight='bold', fontsize='medium', y=1.08)
        elif cnt_plot == 3:
            ax.set_title('train gt: gt+++-', fontweight='bold', fontsize='medium', y=1.08)
        if cnt_plot%n_cols == 1:
            ax.set_ylabel("unaries: %s" % name_unary, fontweight='bold', fontsize='medium', labelpad=13)
        ax.imshow(rgb_vis)
        clear_ticks(ax)
        if cnt_plot == 1:
            fig.suptitle('Prediction for %s (C=%s)'%(sub.name, str(model.C)))
    # plt.tight_layout()
    return fig


def table_cherry_score_models_with_different_gt(test_cherry_segmentations_dir, cherry_subs, ssvm_alg='nslack', C=5):
    from sklearn.cross_validation import ShuffleSplit

    test_gts = ['gt_active_++', 'gt_active_+++', 'gt_active_+++x', 'gt']
    pattern = '&&   %s && %0.2f & %0.2f & %0.2f & %0.2f \\\\'
    rows = []
    scores = {}

    # prepare underpants
    sp = ShuffleSplit(len(cherry_subs), random_state=1, test_size=.8)
    _, test_idxs = next(iter(sp))
    model_paths, _, _ = process_results(test_cherry_segmentations_dir, demo_sub_name=None, ssvm_alg=ssvm_alg)
    model_paths = [path for path in model_paths if float(path.split('_C=')[1].split('/')[0]) == C]
    model_paths = convert_gtsymbols_to_numbers(model_paths)

    for model_path in model_paths:
        # model combo
        model_path = convert_gtnumber_to_symbol(model_path)
        logger = pystruct_logger(model_path)
        model = logger.load()
        # features, gt, label set
        name_unary = model_path.split('_u=')[1].split('_p=')[0]
        name_pairwise = 'feats_xcorr_green'
        name_traingt = model_path.split('_train-gt=')[1].split('_C')[0]
        label_set = name_unary.split('feats_pm_set')[1][0]
        combo = label_set + convert_gtsymbol_to_number(model_path.split('train-gt=gt_active')[1].split('C=')[0])[-2] # excuse me?
        print '=== Model:', model_path
        print '    Prob map:', name_unary
        print '    GT:', name_traingt
        print '    Combo key:', combo
        print ' + Building X...'
        X = [_sub.ssvm_create_x(name_unary, name_pairwise, bias_u=False, bias_p=False) for _sub in cherry_subs]
        X_test = [X[i] for i in test_idxs]

        for test_gt in test_gts:
            print ' + Building Y with test GT:', test_gt
            Y_test = [[_sub.ssvm_create_y(test_gt) for _sub in cherry_subs][i] for i in test_idxs]
            score = model.score(X_test, Y_test)
            print '   Score:', score
            try:
                scores[combo].append(score)
            except KeyError:
                scores[combo] = [score]

    combos = scores.keys() # keys are like A0, A1, A2, D0, D1, ...
    combos.sort()
    for c in combos:
        row =  pattern % (c[0], scores[c][0]*100, scores[c][1]*100, scores[c][2]*100, scores[c][3]*100)
        print row
        rows.append(row)
        if int(c[1]) == 2:
            print '\n'
    print '\nDone for C=', C
    return scores, rows


def table_crazy_score_models_with_different_gt(model_paths, cherry_subs):
    from sklearn.cross_validation import ShuffleSplit
    feats_unary = ['feats_pm_setA_green', 'feats_pm_setJ_green', 'feats_pm_setD_green', 'feats_pm_setE_green',
                           'feats_pm_setA_red', 'feats_pm_setJ_red', 'feats_pm_setD_red', 'feats_pm_setE_red']
    name_pairwise = 'feats_xcorr_green'
    test_gts = ['gt_active_++', 'gt_active_+++', 'gt_active_+++x', 'gt']
    pattern = '&&   %s && %0.2f & %0.2f & %0.2f & %0.2f \\\\'
    rows = []
    scores = {}

    # prepare underpants
    sp = ShuffleSplit(len(cherry_subs), random_state=1, test_size=.8)
    _, test_idxs = next(iter(sp))
    print 'Building X...'
    X = [_sub.ssvm_create_x(feats_unary, name_pairwise, bias_u=False, bias_p=False) for _sub in cherry_subs]
    X_test = [X[i] for i in test_idxs]

    for model_path in model_paths:
        # model combo
        model_path = convert_gtnumber_to_symbol(model_path)
        logger = pystruct_logger(model_path)
        model = logger.load()
        # features, gt, label set
        combo = 'JADE-gr' + convert_gtsymbol_to_number(model_path.split('train-gt=gt_active')[1].split('C=')[0])[-2] # excuse me?
        print '=== Model:', model_path
        print ' + Combo key:', combo
        for test_gt in test_gts:
            print ' + Building Y with test GT:', test_gt
            Y_test = [[_sub.ssvm_create_y(test_gt) for _sub in cherry_subs][i] for i in test_idxs]
            score = model.score(X_test, Y_test)
            print '   Score:', score
            try:
                scores[combo].append(score)
            except KeyError:
                scores[combo] = [score]
    combos = scores.keys() # keys are AJDE_gr0, AJDE_gr1, AJDE_gr2
    combos.sort()
    for c in combos:
        row =  pattern % (c[0], scores[c][0]*100, scores[c][1]*100, scores[c][2]*100, scores[c][3]*100)
        print row
        rows.append(row)
    print '\nDone for C=', model.C # all models specified by model_paths should have the same C
    return scores, rows


def plot_rgb_vizzz_demo_vol_with_different_gt(model_paths, subvolume):
    img_x, img_y = subvolume.dims

    print 'load ground truth for RGB vizzz'
    gt0 = subvolume.get('gt_active_++').astype('uint8')
    gt1 = subvolume.get('gt_active_+++').astype('uint8')
    gt2 = subvolume.get('gt_active_+++x').astype('uint8')
    gtJ = subvolume.get('gt').astype('uint8')
    gts = [gt0, gt1, gt2, gtJ]

    fig =plt.figure()
    n_rows = len(model_paths)
    n_cols = 5
    cnt_plot = 0
    print 'max_plots:', n_rows*n_cols
    print 'nrows', n_rows
    for path in model_paths:
        print path

    for i in xrange(len(model_paths)):
        cnt_plot += 1
        # underpants
        model_path = convert_gtnumber_to_symbol(model_paths[i])

        name_unary = model_path.split('_u=')[1].split('_p=')[0]
        if name_unary == 'AJDE_gr': # crazy
            feats_unary = ['feats_pm_setA_green', 'feats_pm_setJ_green', 'feats_pm_setD_green', 'feats_pm_setE_green',
                           'feats_pm_setA_red', 'feats_pm_setJ_red', 'feats_pm_setD_red', 'feats_pm_setE_red']
            name_unary = 'JADE-gr'
            combo = name_unary + '_gt' + model_path.split('train-gt=gt_active_')[1].split('_C=')[0]
            fig.set_size_inches(12,6)
        else: # cherry
            feats_unary = name_unary
            label_set = name_unary.split('feats_pm_set')[1][0]
            combo = 'pm' + label_set + model_path.split('train-gt=gt_active')[1].split('_C=')[0]
            fig.set_size_inches(12,12)
        if combo.endswith('x'):
            combo = combo.split('x')[0] + '-'

        name_pairwise = 'feats_xcorr_green'
        name_traingt = model_path.split('_train-gt=')[1].split('_C')[0]
        logger = pystruct_logger(model_path)
        model = logger.load()
        print '=== Model:', model_path
        print '    Prob map:', name_unary
        print '    train GT:', name_traingt
        print '    Combo key:', combo
        print 'predicting...'
        x = subvolume.ssvm_create_x(feats_unary, name_pairwise, bias_u=False, bias_p=False)
        pred = model.predict([x])
        pred = pred[0].reshape(img_x, img_y)
        ax_ = fig.add_subplot(n_rows, n_cols, cnt_plot)
        subvolume.plot_activity_labels(image=pred, ax=ax_, cmap='gray', linewidth=.4)
        if cnt_plot == 1:
            # fig.suptitle('Model prediction for %s visualized with different ground truth (C=%s)'%(sub.name, str(model.C)),
            #              fontweight='bold', fontsize='x-large', y=1.04)
            ax_.set_title('labeled prediction', fontweight='bold', fontsize='medium', y=1.08)
        if cnt_plot%n_cols == 1:
            ax_.set_ylabel(combo, fontweight='bold', fontsize='medium', labelpad=13)
        for gt_i in xrange(len(gts)):
            cnt_plot += 1
            ax_ = fig.add_subplot(n_rows, n_cols, cnt_plot)
            rgb_vis = plot_it_like_ferran(gts[gt_i], pred)
            ax_.imshow(rgb_vis)
            if cnt_plot == 2:
                ax_.set_title('gt++', fontweight='bold', fontsize='medium', y=1.08)
            elif cnt_plot == 3:
                ax_.set_title('gt+++', fontweight='bold', fontsize='medium', y=1.08)
            elif cnt_plot == 4:
                ax_.set_title('gt+++-', fontweight='bold', fontsize='medium', y=1.08)
            elif cnt_plot == 5:
                ax_.set_title('gt_janelia', fontweight='bold', fontsize='medium', y=1.08)
    clear_ticks(fig.get_axes())
    # plt.tight_layout()
    return fig


def plot_loss_and_objective_curves_for_different_models(model_paths):
    colors = ['g', 'r', 'b', 'm', 'y', 'c', 'k', .75]
    from core.visuals import plot_ssvm_learning_on_axis
    fig = plt.figure(figsize=(15, 6))
    ax_obj = fig.add_subplot(1,2,1)
    ax_loss = fig.add_subplot(1,2,2)
    col_cnt = 0
    for model_path in model_paths:
        logger = pystruct_logger(model_path)
        model = logger.load()
        current_c = float(model_path.split('C=')[1].split('/')[0])
        ax_obj, ax_loss = plot_ssvm_learning_on_axis(model, time=False, axes=[ax_obj, ax_loss], label=current_c, color=colors[col_cnt])
        col_cnt += 1
    return fig


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    in_dir = args[0]


    if 0: # test 'crazy' segmentaions (sets AJDE, green AND red channel) ###############################################
        test_C = [5, .3, .0001]
        ssvm_algs = ['oneslack_ogm-gc', 'nslack_ogm-gc']
        if 1:
            jd = JaneliaData(dummy_data=True)
            for ssvm_alg in ssvm_algs:
                for C in test_C:
                    for sub in jd.subvolumes[:-1]:
                        model_paths, _, _ = process_results(in_dir, sub.name, ssvm_alg)
                        model_paths = [path for path in model_paths if float(path.split('_C=')[1].split('/')[0]) == C]
                        model_paths = convert_gtsymbols_to_numbers(model_paths)

                        if 1: # A) plot RGB vizzz predictions of trained models on demo subvolumes
                            _fig = plot_rgb_segmentations_for_demo_sub(model_paths, subvolume=sub)
                            _fig.tight_layout()
                            _fig.savefig('out/plot_crazy_%s_%s_model_pred-prgb_C=%.2e.png'%(sub.name, ssvm_alg, C) , dpi=300)
                            plt.close('all')

                        if 1: # B1) plot model predictions on demo sub with different test gt
                            _fig = plot_rgb_vizzz_demo_vol_with_different_gt(model_paths, subvolume=sub)
                            _fig.tight_layout()
                            _fig.savefig('out/plot_crazy_%s_model_pred-prgb_w_different_test-gts_%s_C=%0.1e.png'%(sub.name, ssvm_alg, C), dpi=300)
                            plt.close('all')

        if 0: # B2) table with test set scores for given model
            import cPickle as pickle
            jd = JaneliaData()
            cherry_subs = jd.ssvm_data_cherry_picked
            for ssvm_alg in ssvm_algs:
                test_scores = []
                table_rows = []
                for C in test_C:
                    model_paths, _, _ = process_results(in_dir, demo_sub_name=None, ssvm_alg=ssvm_alg)
                    model_paths = [path for path in model_paths if float(path.split('_C=')[1].split('/')[0]) == C]
                    model_paths = convert_gtsymbols_to_numbers(model_paths)
                    scores, rows = table_crazy_score_models_with_different_gt(model_paths, cherry_subs)
                    test_scores.append(scores)
                    table_rows.append(rows)
                    pickle.dump(test_scores, open('out/crazy_scores_%s_c=%0.1e.p'%(ssvm_alg, C), 'w'))
                    pickle.dump(table_rows, open('out/crazy_table_rows_%s_c=%0.1e.p'%(ssvm_alg, C), 'w'))
                pickle.dump(test_scores, open('out/crazy_scores_%s_allC.p'%ssvm_alg, 'w'))
                pickle.dump(table_rows, open('out/crazy_table_rows_%s_allC.p'%ssvm_alg, 'w'))
                for i in xrange(len(test_C)):
                    print '======  C:', test_C[i]
                    for row in table_rows[i]:
                        print row
                    print '\n'
            print '\n\n\n======================================================================='

        if 1: # C) loss and objective plots for different C
            for ssvm_alg in ssvm_algs:
                combos = {}
                model_paths, _, _ = process_results(in_dir, None, ssvm_alg)
                model_paths = convert_gtsymbols_to_numbers(model_paths)
                for model_path in model_paths:
                    model_path = convert_gtnumber_to_symbol(model_path)
                    combo = 'AJDE_gr' + convert_gtsymbol_to_number(model_path.split('train-gt=gt_active')[1].split('C=')[0])[-2] # excuse me?
                    try:
                        combos[combo].append(model_path)
                    except:
                        combos[combo] = [model_path]
                for combo in combos.keys():
                    fig = plot_loss_and_objective_curves_for_different_models(combos[combo])
                    plt.savefig('out/plot_crazy_loss_and_objective_curves_differentC_%s_model=%s.png'%(combo, ssvm_alg), dpi=300)



    if 1:# test_cherry_segmentations ###################################################################################
        converged_models = ['A0', 'D0', 'E0', 'E2', 'J0']
        shitty_example_models = ['A1', 'A2', 'E1']
        test_C = [100, 50, 5, .3, .1, .01, .001, .0001]
        ssvm_algs = ['oneslack', 'nslack']
        if 0: # A) plot RGB vizzz predictions of trained models on demo subvolumes
            jd = JaneliaData(dummy_data=True)
            for ssvm_alg in ssvm_algs:
                for C in [5, .3, .0001]: # restrict to some C
                    for sub in jd.subvolumes:
                        model_paths, _, _ = process_results(in_dir, sub.name, ssvm_alg)
                        model_paths = [path for path in model_paths if float(path.split('_C=')[1].split('/')[0]) == C]
                        model_paths = convert_gtsymbols_to_numbers(model_paths)
                        _fig = plot_rgb_segmentations_for_demo_sub(model_paths, subvolume=sub)
                        _fig.savefig('out/plot_cherry_%s_%s_model_predictions_prgb_matrix_C=%.2e.png'%(sub.name, ssvm_alg, C) , dpi=300)
                        plt.close('all')
                        # plt.show()

        if 0: # B1) plot selected model predictions on demo sub with different test gt
            jd=JaneliaData(dummy_data=True)
            for sub in jd.subvolumes:
                for ssvm_alg in ssvm_algs:
                    for C in test_C:
                        example_model_paths = []
                        model_paths, _, _ = process_results(in_dir, None, ssvm_alg)
                        model_paths = [path for path in model_paths if float(path.split('_C=')[1].split('/')[0]) == C]
                        model_paths = convert_gtsymbols_to_numbers(model_paths)
                        for model_path in model_paths:
                            model_path = convert_gtnumber_to_symbol(model_path)
                            name_unary = model_path.split('_u=')[1].split('_p=')[0]
                            label_set = name_unary.split('feats_pm_set')[1][0]
                            combo = label_set + convert_gtsymbol_to_number(model_path.split('train-gt=gt_active')[1].split('C=')[0])[-2] # excuse me?
                            if combo in converged_models:
                                example_model_paths.append(model_path)
                            if combo in shitty_example_models:
                                example_model_paths.append(model_path)
                        fig = plot_rgb_vizzz_demo_vol_with_different_gt(example_model_paths, sub)
                        plt.savefig('out/plot_cherry_%s_example_models_pred-prgb_w_different_test-gts_%s_C=%0.1e.png'%(sub.name, ssvm_alg, C), dpi=300)
                        plt.close('all')

        if 0: # B2) table with test set scores for given model
            import cPickle as pickle
            jd = JaneliaData()
            cherry_subs = jd.ssvm_data_cherry_picked
            for ssvm_alg in ssvm_algs:
                test_scores = []
                table_rows = []
                for C in test_C:
                    scores, rows = table_cherry_score_models_with_different_gt(in_dir, cherry_subs, ssvm_alg, C)
                    test_scores.append(scores)
                    table_rows.append(rows)
                    pickle.dump(test_scores, open('out/cherry_scores_%s_c=%0.1e.p'%(ssvm_alg, C), 'w'))
                    pickle.dump(table_rows, open('out/cherry_table_rows_%s_c=%0.1e.p'%(ssvm_alg, C), 'w'))
                pickle.dump(test_scores, open('out/cherry_scores_%s_allC.p'%ssvm_alg, 'w'))
                pickle.dump(table_rows, open('out/cherry_table_rows_%s_allC.p'%ssvm_alg, 'w'))
                for i in xrange(len(test_C)):
                    print '======  C:', test_C[i]
                    for row in table_rows[i]:
                        print row
                    print '\n'
            print '\n\n\n======================================================================='

        if 1: # C) loss and objective plots for different C
            chosen_C = [5, .3, 0.01,  0.001] # more than 4 C at once quickly turn out to be hideous
            assert chosen_C in test_C
            for ssvm_alg in ssvm_algs:
                combos = {}
                model_paths, _, _ = process_results(in_dir, None, ssvm_alg)
                model_paths = convert_gtsymbols_to_numbers(model_paths)
                for model_path in model_paths:
                    model_path = convert_gtnumber_to_symbol(model_path)
                    name_unary = model_path.split('_u=')[1].split('_p=')[0]
                    label_set = name_unary.split('feats_pm_set')[1][0]
                    combo = label_set + convert_gtsymbol_to_number(model_path.split('train-gt=gt_active')[1].split('C=')[0])[-2] # excuse me?
                    current_c = float(model_path.split('C=')[1].split('/')[0])
                    if (current_c in chosen_C) and ((combo in converged_models) or (combo in shitty_example_models)):
                        try:
                            combos[combo].append(model_path)
                        except:
                            combos[combo] = [model_path]
                for combo in combos.keys():
                    fig = plot_loss_and_objective_curves_for_different_models(combos[combo])
                    plt.savefig('out/plot_cherry_loss_and_objective_curves_differentC_%s_model=%s.png'%(combo, ssvm_alg), dpi=300)



    if 0: # test_final_segmentations (BUUH) ############################################################################
        if 0:
            for sub in [jd.demo_volume]:
                sub_name = sub.name
                for ssvm_alg in ['oneslack']:#, 'nslack']:
                    for C in [5]:#[.1, .01, .001, .3, 5]:
                        # 1) plot RGB vizz from precomputed PNG
                        # _fig = plot_final_segmentations_for_demo_sub_from_PNG(in_dir, ssvm_alg=ssvm_alg, C=C, demo_sub_name=sub_name)
                        # _fig.savefig('out/plot1_%s_%s_rgb_vizzz_matrix_C=%.2e.png'%(sub_name, ssvm_alg, C) , dpi=300)

                        # 2) plot RGB vizzz by predicting with the trained models again (test_final_segmentations)
                        #  _fig = plot_final_segmentations_for_demo_sub(in_dir, subvolume=sub, ssvm_alg=ssvm_alg, C=C)
                        #  _fig.savefig('out/plot_final_%s_%s_rgb_vizzz_matrix_C=%.2e.png'%(sub_name, ssvm_alg, C) , dpi=300)

                        fig = plot_demo_sub_best_pred_with_different_gt(in_dir, 7, sub, C, ssvm_alg)
                        # embed()
                        # plt.close('all')
        if 0:
            fig1 = plot_ssvm_score_vs_test_size(in_dir)
            plt.savefig('out/shitty_ssvm_vs_test_size_plot.png', dpi=300)
    print 'Done.'
    sys.exit(0)