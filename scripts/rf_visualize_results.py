__version__ = '0.1'

import sys
import os
import cPickle as pickle

import matplotlib.pyplot as plt

import seaborn as sns

from core.visuals import plot_roc_curve_on_axis, plot_pr_curve_on_axis, sort_legend
from core.data.calcium_imaging import JaneliaData

# globals
colors = ['g', 'r', 'b', 'm', 'y', .75, 'c', 'k']
#colors = sns.color_palette('YlGnBu_r', n_colors=6)
#colors = sns.diverging_palette(255, 133, l=60, n=6, center="dark")
#colors = sns.diverging_palette(10, 220, sep=80, n=6)
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']

flat = {
    'A' : ('active_very_certain'),
    'B' : ('active_very_certain', 'active_mod_certain'),
    'C' : ('active_very_certain', 'active_mod_certain', 'inactive_very_certain', 'inactive_auto'),
    'D' : ('active_very_certain', 'active_mod_certain', 'uncertain', 'inactive_very_certain', 'inactive_auto'),
    'E' : ('active_very_certain', 'active_mod_certain', 'uncertain', 'inactive_auto', 'inactive_very_certain'),
    'F' : ('active_very_certain', 'active_mod_certain', 'uncertain', 'inactive_auto', 'inactive_very_certain'),
    'G' : ('active_very_certain', 'uncertain'),
    'H' : ('active_very_certain', 'inactive_auto'),
    'I' : ('active_very_certain', 'active_mod_certain'),
    'J' : ('active_very_certain', 'active_mod_certain', 'uncertain', 'inactive_auto', 'inactive_very_certain')
}


def get_pickle_paths(rf_labelings_result_dir):
    # phase 1: collect underpants
    stuff = []
    for root, _, files in os.walk(rf_labelings_result_dir, topdown=False):
        for name in files:
            stuff.append(os.path.join(root, name))
    rf_pickle_paths = [path for path in stuff if path.endswith('random_forest.pickle')]
    result_pickle_paths = [path for path in stuff if path.endswith('results.pickle')]
    return rf_pickle_paths, result_pickle_paths

def print_numbers_for_latex_table(result_pickle_paths):
    results = {}

    for pickle_path in result_pickle_paths:
        sample_size = int(pickle_path.split('_max')[1].split('/')[0])
        results[sample_size] = pickle.load(open(pickle_path, 'r'))
        channel = pickle_path.split('rf_labelings_')[1].split('/')[0] # always the same for given input

    if 0:
        print channel.upper()
        for balancing in ['results_balanced_test', 'results_unbalanced_test']:
            print balancing.upper()
            sample_sizes = results.keys()
            sample_sizes.sort()
            for sample_size in sample_sizes:
                accs = []
                aucs = []
                label_sets = results[sample_size].keys() # A-J
                label_sets.sort()
                for label_set in label_sets:
                    # find best cv run
                    best_auc = 0
                    best_acc = 0
                    for cv_run in results[sample_size][label_set].keys():
                        cv_results = results[sample_size][label_set][cv_run][balancing]
                        if best_auc < cv_results['roc_auc']:
                            best_auc = cv_results['roc_auc']
                        if best_acc < cv_results['acc']:
                            best_acc = cv_results['acc']
                    accs.append(best_acc)
                    aucs.append(best_auc)
                numbers = []
                numbers.append(int(sample_size))
                # for acc in accs:
                #     numbers.append(float(acc)*100)
                for auc in aucs:
                    numbers.append(float(auc)*100)
                # while len(numbers) < 13:
                #     numbers.append(float(42.424242))
                #print '& %d & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f && %.2f & %.2f & %.2f & %.2f & %.2f & %.2f \\\\' % tuple(numbers)
                print '& %d & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f \\\\' % tuple(numbers)
    if 0:
        print channel.upper()
        for balancing in ['results_balanced_test', 'results_unbalanced_test']:
            print balancing.upper()
            sample_sizes = results.keys()
            sample_sizes.sort()
            for sample_size in sample_sizes:
                best_results = []
                label_sets = results[sample_size].keys() # A-J
                label_sets.sort()
                for label_set in label_sets:
                    # find best cv run by maximum AUROC
                    best_auc = 0
                    # best_acc = 0
                    for cv_run in results[sample_size][label_set].keys():
                        cv_results = results[sample_size][label_set][cv_run][balancing]
                        if best_auc < cv_results['roc_auc']:
                            best_auc = cv_results['roc_auc']
                            best_run = cv_results
                    best_results.append(best_run)

                cnt = 0
                for bla in best_results:
                    numbers = [label_sets[cnt], int(sample_size), int(bla['n_pos_train']), int(bla['n_neg_train']),
                               int(bla['n_pos_test']), int(bla['n_neg_test']), float(bla['acc'])*100, float(bla['roc_auc'])*100]
                    print '%s & %d & %d & %d & %d & %d & %.2f & %.2f \\\\' % tuple(numbers)
                    cnt+=1
            print '\n\n'
                    #print '& %d & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f && %.2f & %.2f & %.2f & %.2f & %.2f & %.2f \\\\' % tuple(numbers)

    if 0:
        sample_size = 75000
        print channel.upper()
        print sample_size
        for balancing in ['results_balanced_test', 'results_unbalanced_test']:
            print balancing.upper()
            best_results = []
            label_sets = results[sample_size].keys() # A-J for sample size 75k
            label_sets.sort()
            print label_sets
            for label_set in label_sets:
                # find best cv run by maximum AUROC
                best_auc = 0
                # best_acc = 0
                for cv_run in results[sample_size][label_set].keys():
                    cv_results = results[sample_size][label_set][cv_run][balancing]
                    if best_auc < cv_results['roc_auc']:
                        best_auc = cv_results['roc_auc']
                        best_run = cv_results
                best_results.append(best_run)

            auc = []
            acc = []
            per_class_train = []
            pos_test = []
            neg_test = []
            for bla in best_results:
                auc.append(float(bla['roc_auc'])*100)
                acc.append(float(bla['acc'])*100)
                per_class_train.append(int(bla['n_pos_train']))
                pos_test.append(int(bla['n_pos_test']))
                neg_test.append(int(bla['n_neg_test']))
            pattern_d = '& & %d & %d & %d & %d & %d & %d & %d & %d & %d & %d \\\\'
            pattern_f = '& & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f \\\\'
            print pattern_d % tuple(per_class_train)
            print pattern_d % tuple(pos_test)
            print pattern_d % tuple(neg_test)
            print pattern_f % tuple(acc)
            print pattern_f % tuple(auc)
            print '\n'
            print '\n\n'
                    #print '& %d & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f && %.2f & %.2f & %.2f & %.2f & %.2f & %.2f \\\\' % tuple(numbers)
    if 1:
        sample_size = 75000
        print channel.upper()
        print sample_size
        filling = []

        for balancing in ['results_balanced_test', 'results_unbalanced_test']:
            best_results = []
            label_sets = results[sample_size].keys() # A-J for sample size 75k
            label_sets.sort()
            for label_set in label_sets:
                # find best cv run by maximum AUROC
                best_auc = 0
                # best_acc = 0
                for cv_run in results[sample_size][label_set].keys():
                    cv_results = results[sample_size][label_set][cv_run][balancing]
                    if best_auc < cv_results['roc_auc']:
                        best_auc = cv_results['roc_auc']
                        best_run = cv_results
                best_results.append(best_run)

            if balancing == 'results_balanced_test':
                for i in xrange(len(label_sets)):
                    bla = best_results[i]
                    mystring = '& %s & %d && %d & %.2f & %.2f &&' % (label_sets[i], int(bla['n_pos_train']),
                                                                     int(bla['n_pos_test']), float(bla['acc'])*100,
                                                                     float(bla['roc_auc'])*100 )
                    filling.append(mystring)
            else:
                for i in xrange(len(label_sets)):
                    bla = best_results[i]
                    filling[i] = '%s %d & %d & %.2f & %.2f \\\\' % (filling[i], int(bla['n_pos_test']), int(bla['n_neg_test']), float(bla['acc'])*100,
                                                                     float(bla['roc_auc'])*100 )

        for line in filling:
            print line
        print '\n\n'
                    #print '& %d & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f && %.2f & %.2f & %.2f & %.2f & %.2f & %.2f \\\\' % tuple(numbers)

def find_best_model(rf_labelings_result_dir, max_sample_size=None):
    best_models = []
    rf_pickle_paths, result_pickle_paths = get_pickle_paths(rf_labelings_result_dir)

    results = {}
    for pickle_path in result_pickle_paths:
        sample_size = int(pickle_path.split('_max')[1].split('/')[0])
        results[sample_size] = pickle.load(open(pickle_path, 'r'))

    rf_paths = rf_pickle_paths
    rf_paths.sort()

    if max_sample_size is None:
        max_sample_sizes = [5000, 25000, 50000, 75000]
    else:
        max_sample_sizes = [max_sample_size]

    for label_set in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
        label_rfs = [path for path in rf_paths if path.split('label set ')[1][0] == label_set]
        best_auc = 0
        best_cv = None
        for sample_size in max_sample_sizes:
            tmp = [rf for rf in label_rfs if int(rf.split('_max')[1].split('/')[0]) == sample_size]
            for path in tmp:
                cv_run = int(path.split('/cv_runs/cv')[1][0])
                auc = results[sample_size][label_set][cv_run]['results_unbalanced_test']['roc_auc']
                if best_auc < auc:
                    best_auc = auc
                    best_cv = path
        best_models.append(best_cv)
        print best_cv
        """
        OUTPUT for 75000
        results/rf_labelings_nice_green/rf_test_label_sets_max75000/label set A/cv_runs/cv2_random_forest.pickle
        results/rf_labelings_nice_green/rf_test_label_sets_max75000/label set B/cv_runs/cv2_random_forest.pickle
        results/rf_labelings_nice_green/rf_test_label_sets_max75000/label set C/cv_runs/cv2_random_forest.pickle
        results/rf_labelings_nice_green/rf_test_label_sets_max75000/label set D/cv_runs/cv2_random_forest.pickle
        results/rf_labelings_nice_green/rf_test_label_sets_max75000/label set E/cv_runs/cv0_random_forest.pickle
        results/rf_labelings_nice_green/rf_test_label_sets_max75000/label set F/cv_runs/cv0_random_forest.pickle
        results/rf_labelings_nice_green/rf_test_label_sets_max75000/label set G/cv_runs/cv2_random_forest.pickle
        results/rf_labelings_nice_green/rf_test_label_sets_max75000/label set H/cv_runs/cv1_random_forest.pickle
        results/rf_labelings_nice_green/rf_test_label_sets_max75000/label set I/cv_runs/cv1_random_forest.pickle
        results/rf_labelings_nice_green/rf_test_label_sets_max75000/label set J/cv_runs/cv2_random_forest.pickle

        OUTPUT for overall best model per set
        results/rf_labelings_nice_green/rf_test_label_sets_max75000/label set A/cv_runs/cv2_random_forest.pickle
        results/rf_labelings_nice_green/rf_test_label_sets_max25000/label set B/cv_runs/cv0_random_forest.pickle
        results/rf_labelings_nice_green/rf_test_label_sets_max50000/label set C/cv_runs/cv0_random_forest.pickle
        results/rf_labelings_nice_green/rf_test_label_sets_max75000/label set D/cv_runs/cv2_random_forest.pickle
        results/rf_labelings_nice_green/rf_test_label_sets_max75000/label set E/cv_runs/cv0_random_forest.pickle
        results/rf_labelings_nice_green/rf_test_label_sets_max75000/label set F/cv_runs/cv0_random_forest.pickle
        results/rf_labelings_nice_green/rf_test_label_sets_max50000/label set G/cv_runs/cv2_random_forest.pickle
        results/rf_labelings_nice_green/rf_test_label_sets_max75000/label set H/cv_runs/cv1_random_forest.pickle
        results/rf_labelings_nice_green/rf_test_label_sets_max50000/label set I/cv_runs/cv1_random_forest.pickle
        results/rf_labelings_nice_green/rf_test_label_sets_max75000/label set J/cv_runs/cv2_random_forest.pickle
        """
    return best_models

def save_prob_maps(out_dir, pickled_forest_path, subvolumes, channel='green'):
    cmaps = ['YlGnBu_r', None, 'coolwarm']
    label_set = pickled_forest_path.split('label set ')[1][0]
    print 'Loading random forest from', pickled_forest_path
    rf = pickle.load(open(pickled_forest_path, 'r'))
    for sub in subvolumes:
        print 'Processing', sub.name
        feats = sub.get('feats_ilastik_pseudoRGB_%s' % channel)
        pm = rf.predict_proba(feats.reshape((512 * 512, -1)))
        for cmap in cmaps:
            ax = sub.plot_activity_labels(image=pm[:, 0].reshape(512, 512), cmap=cmap, labels=flat[label_set])
            plt.savefig(out_dir + '/vizzz_prob_maps/%s_set%s_pm0_%s.png' % (sub.name, label_set, cmap))
            plt.close('all')
            ax2 = sub.plot_activity_labels(image=pm[:, 1].reshape(512, 512), cmap=cmap, labels=flat[label_set])
            plt.savefig(out_dir + '/vizzz_prob_maps/%s_set%s_pm1_%s.png' % (sub.name, label_set, cmap))
            plt.close('all')

def plot_prob_map_comparison_for_example_volume(subvolume, best_rf_model_paths, out_dir=None, cmap=None, p_class=0):
    assert len(best_rf_model_paths) == 10 # should be the one best classifier trained on the A to J label sets respectively
    assert p_class in [0, 1] # plot heat maps with positive or negative class probabilites
    channel = best_rf_model_paths[0].split('rf_labelings_nice_')[1].split('/')[0] # always the same for given input
    feats = subvolume.get('feats_ilastik_pseudoRGB_%s' % channel)
    pms = []
    label_sets = []
    print 'Processing', subvolume.name
    for pickled_forest_path in best_rf_model_paths:
        label_set = pickled_forest_path.split('label set ')[1][0]
        label_sets.append(label_set)
        print 'Predicting with random forest:', pickled_forest_path
        rf = pickle.load(open(pickled_forest_path, 'r'))
        pms.append(rf.predict_proba(feats.reshape((512 * 512, -1))))
    # pms = pickle.load(open('out/demo_pms_best.p', 'r'))

    fig =plt.figure(figsize=(8, 8))
    #fig.subplots_adjust(bottom=0.025, left=0.025, top = 0.975, right=0.975)
    layout = [(3,3,1), (3,3,2), (3,3,3), (3,3,4), (3,3,5), (3,3,6), (3,3,7), (3,3,8), (3,3,9)]
    cnt=0
    for nrows, ncols, plot_number in layout:
        if label_sets[cnt] == 'F': cnt +=1 # skip janelia gt label set
        sp = fig.add_subplot(nrows, ncols, plot_number)
        sp.set_xticks([])
        sp.set_yticks([])
        sp = subvolume.plot_activity_labels(image=pms[cnt][:, p_class].reshape(512, 512), cmap=cmap, labels=flat[label_sets[cnt]])
        sp.set_title(label_sets[cnt])
        cnt+=1
    fig.tight_layout()
    if out_dir is not None:
        fig.savefig(os.path.join(out_dir, '%s_pm%d_plot_cmap=%s_channel=%s.png'%(subvolume.name, p_class, cmap, channel)), dpi=300)
    return fig

def plot_roc_and_pr_balanced_vs_unbalanced(result_pickle_path):
    """ Plot ROC and PR curves for a given Calcium channel and the max sample size used for balancing during test.
    Show the difference between balanced and unbalanced test sets on prediction performance.
    :param result_pickle_path: CV result for a given Calcium channel and max sample size used for balancing
    :param channel: Calcium imaging channel
    :param max_sample_size: max sample size used for balancing
    :return: ROC and PR plot for balanced and unbalanced test set respectively.
    """
    style = { 'A' : ('b', '-'),
              'B' : ('m', '-'),
              'C' : ('g', '-'),
              'D' : ('r', '-.'),
              'E' : ('r', '--'),
              'F' : ('k', '--'),
              'G' : ('b', '--'),
              'H' : ('b', '-.'),
              'I' : ('b', ':'),
              'J' : ('r', '-')
    }
    result = pickle.load(open(result_pickle_path, 'r'))
    channel = result_pickle_path.split('rf_labelings_')[1].split('/')[0]
    max_sample_size = int(result_pickle_path.split('_max')[1].split('/')[0])

    ### OUTPUT FIGURES ###
    titles = []
    # ROC curves
    titles.append('ROC curve for %s channel on unbalanced test set (max samples %d)' % (channel, max_sample_size))
    # fig_roc = plt.figure(titles[-1])
    # ax_roc = fig_roc.add_subplot(111)
    titles.append('ROC curve for %s channel on balanced test set (max samples %d)' % (channel, max_sample_size))
    # fig_roc_bal = plt.figure(titles[-1])
    # ax_roc_bal = fig_roc_bal.add_subplot(111)
    # PR curves
    titles.append('PR curve for %s channel on unbalanced test set (max samples %d)' % (channel, max_sample_size))
    # fig_pr = plt.figure(titles[-1])
    # ax_pr = fig_pr.add_subplot(111)
    titles.append('PR curve for %s channel on balanced test set (max samples %d)' % (channel, max_sample_size))
    # fig_pr_bal = plt.figure(titles[-1])
    # ax_pr_bal = fig_pr_bal.add_subplot(111)
    # figs = [fig_roc, fig_roc_bal, fig_pr, fig_pr_bal]
    # axes = [ax_roc, ax_roc_bal, ax_pr, ax_pr_bal]

    fig = plt.figure(figsize=(12, 10))
    ### COLLECT UNDERPANTS ###
    for balancing in ['results_balanced_test', 'results_unbalanced_test']:
        sets = result.keys()
        #sets = ['A', 'G', 'H', 'I', 'J', 'F']
        for label_set in sets:
            # find best cv run
            best_auc = 0
            for cv_run in result[label_set].keys():
                cv_results = result[label_set][cv_run][balancing]
                if best_auc < cv_results['roc_auc']:
                    best_auc = cv_results['roc_auc']
                    best_cv = cv_results
            prec = best_cv['precision_threshs']
            reca = best_cv['recall_threshs']
            fpr = best_cv['fpr']
            tpr = best_cv['tpr']

            if balancing == 'results_balanced_test':
                ax_pr_bal = fig.add_subplot(221)
                ax_pr_bal = plot_pr_curve_on_axis(precision=prec, recall=reca, label=label_set, color=style[label_set][0], linestyle=style[label_set][1], ax=ax_pr_bal)
                # ax_pr_bal.set_title('balanced test set')
                ax_roc_bal = fig.add_subplot(223)
                ax_roc_bal = plot_roc_curve_on_axis(fpr, tpr, best_auc, label=label_set, color=style[label_set][0], linestyle=style[label_set][1], ax=ax_roc_bal)
            else:
                ax_pr = fig.add_subplot(222)
                ax_pr = plot_pr_curve_on_axis(precision=prec, recall=reca, label=label_set, color=style[label_set][0], linestyle=style[label_set][1],ax=ax_pr)
                # ax_pr.set_title('unbalanced test set')
                ax_roc = fig.add_subplot(224)
                ax_roc = plot_roc_curve_on_axis(fpr, tpr, label=label_set, color=style[label_set][0], linestyle=style[label_set][1], ax=ax_roc)

    ax_pr.plot([1, 0], [0, 1], label='random', color='k', linestyle=':', linewidth=.3)
    ax_roc.plot([0, 1], [0, 1], label='random', color='k', linestyle=':', linewidth=.3)
    ax_pr_bal.plot([1, 0], [0, 1], label='random', color='k', linestyle=':', linewidth=.3)
    ax_roc_bal.plot([0, 1], [0, 1], label='random', color='k', linestyle=':', linewidth=.3)

    # for ax in axes[:2]:
    #     ax.plot([0, 1], [0, 1], label='random', color='k', linestyle=':', linewidth=.3)
    # for ax in axes[2:]:
    #     ax.plot([1, 0], [0, 1], label='random', color='k', linestyle=':', linewidth=.3)

    [sort_legend(ax) for ax in fig.get_axes()]
    # return figs, axes, titles
    fig.tight_layout()
    return fig, 'roc_and_pr_bal_vs_unbal_%s_%s.png'%(max_sample_size, channel)

def write_prob_maps_to_HDF5(best_rf_paths):
    cd = JaneliaData()
    subvolumes = cd.ssvm_data_cherry_picked

    for forest_file_path in best_rf_paths:
        print 'Loading random forest from', forest_file_path
        rf = pickle.load(open(forest_file_path, 'r'))
        label_set = forest_file_path.split('label set ')[1][0]
        channel = forest_file_path.split('rf_labelings_nice_')[1].split('/')[0]
        for sub in subvolumes:
            print 'Processing', sub.name
            h5_field='feats_pm_set%s_%s'%(label_set, channel)
            try:
                sub.get(h5_field)
                print '\talready computed', h5_field, 'for', sub.name, '--> skipping'
                continue
            except AssertionError:
                feats = sub.get('feats_ilastik_pseudoRGB_%s' % channel)
                pm = rf.predict_proba(feats.reshape((512 * 512, -1)))
                sub.h5_write(h5_field=h5_field, value=pm)
    print 'Done.'

if __name__=='__main__':
    sns.set(style='ticks', palette='muted', color_codes=True, context='notebook')

    from IPython import embed
    args = sys.argv[1:]
    in_dir = args[0]
    vizzz_out = 'out/eval_pm/'
    rf_pickle_paths, result_pickle_paths = get_pickle_paths(in_dir)

    # 1) tables
    print_numbers_for_latex_table(result_pickle_paths=result_pickle_paths)

    # 2 ) figures
    # for result in result_pickle_paths:
    #     fig, title = plot_roc_and_pr_balanced_vs_unbalanced(result)
    #     # figs, axes, titles = plot_roc_and_pr_balanced_vs_unbalanced(result)
    # plt.show()
    # fig.savefig(os.path.join(vizzz_out, title), dpi=300)

    # for i in xrange(len(figs)):
    #     figs[i].savefig('%s/%s.png'%(vizzz_out, titles[i]), dpi=300)

    # 3) plot prob maps comparison
    # jd = JaneliaData(dummy_data=True)
    # best_rf_paths = find_best_model(in_dir, 75000)
    # plot_prob_map_comparison_for_example_volume(jd.demo_volume, best_rf_paths, out_dir=vizzz_out, cmap='gray', p_class=1)

    # 4) save probmaps to PNG
    # for rf_path in best_rf_paths:
    #     save_prob_maps(vizzz_out, rf_path, jd.subvolumes, 'green')

    # 5) write probmaps back to subvolume HDF5 files
    # best_rf_paths = find_best_model(in_dir, 75000)
    # print best_rf_paths
    # write_prob_maps_to_HDF5(best_rf_paths)

    # embed()
    sys.exit()
