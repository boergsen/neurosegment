__version__ = '0.1'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core.learning import RandomForestPipeline, PixelwiseSSVMPipeline
from core.data.calcium_imaging import NeurofinderData
from core.methods import create_out_path
from core.visuals import plot_it_like_ferran


def test_neurofinder_data(out_path, lab_name='hausser', overwrite_output=True):

    out_path += 'test_neurofinder_%s/' % lab_name
    create_out_path(out_path=out_path, except_on_exist=overwrite_output)

    nd = NeurofinderData(lab_name=lab_name)
    dims = nd.subvolumes[0].dims

    rf_pipe = RandomForestPipeline.init_model_from_scratch(out_dir=out_path, calcium_data=nd, which_gt='gt', balance_train_data=True,
                                                            per_class_sample_size=50000, test_size=.75,
                                                            exclude_inactive_neurons=False, only_nice_volumes=False)
    for sub in nd.subvolumes:
        pm = rf_pipe.predict_on_subvolume(sub, verbosity=1)
        plt.imsave(out_path + '%s_pm0.png' % sub.name, pm[:, :, 0])
        plt.imsave(out_path + '%s_pm1.png' % sub.name, pm[:, :, 1])

        h, v = sub.visualize_xcorr()
        plt.imsave(out_path + '%s_xcorr_h.png' % sub.name, h)
        plt.imsave(out_path + '%s_xcorr_v.png' % sub.name, v)

        plt.imsave(out_path + '%s_mean.png' % sub.name, sub.get('mean_green'))
        plt.imsave(out_path + '%s_max.png' % sub.name, sub.get('max_green'))
        plt.imsave(out_path + '%s_std.png' % sub.name, sub.get('std_green'))

        try:
            sub.h5_write('feats_pm', pm.reshape((dims[0] * dims[1], -1)))
        except:
            print 'prob map for sub %s already written' % sub.name
            if overwrite_output:
                print 'updating entry in HDF5 file of %s' % sub.name
                sub.h5_remove_field('feats_pm')
                sub.h5_write('feats_pm', pm.reshape((dims[0] * dims[1], -1)))

    ssvm_baseline_pipe = PixelwiseSSVMPipeline.init_model_from_scratch(out_dir=out_path, calcium_data=nd, test_size=.75,
                                                              feats_unary='feats_pm', feats_pairwise='feats_xcorr_green',
                                                              which_gt='gt', ssvm_iter=50)
    fig, _ = ssvm_baseline_pipe.plot_learning(time=False)
    fig.savefig(out_path + 'ssvm_plot_learning.png', dpi=300)
    plt.close('all')

    for sub in nd.subvolumes:
        pred = ssvm_baseline_pipe.predict_on_subvolume(sub)
        gt = sub.get('gt').astype('uint8')
        rgb_vis = plot_it_like_ferran(gt, pred)
        plt.imsave(out_path + '%s_ssvm_prediction.png' % sub.name, pred)
        plt.imsave(out_path + '%s_gt.png' % sub.name, gt)
        plt.imsave(out_path + '%s_rgb_vis.png' % sub.name, rgb_vis)
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

    test_peres = test_neurofinder_data(out_path=out_dir, lab_name='svoboda_peres')
    test_sofroniew = test_neurofinder_data(out_path=out_dir, lab_name='svoboda_sofroniew')
    test_hausser = test_neurofinder_data(out_path=out_dir, lab_name='hausser')

