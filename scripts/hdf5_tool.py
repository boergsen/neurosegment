__version__ = '0.6'

# import matplotlib
# matplotlib.use('Agg')
import argparse
import cPickle as pickle
import h5py as h5
import numpy as np
import sys

import matplotlib.pyplot as plt

from core.data.calcium_imaging import JaneliaData
from core.methods import create_2d_edge_graph, normalize_image_to_zero_one, create_out_path
from core.visuals import visualize_edges


def save_active_gts_to_H5(h5_volumes_dir, matlab_session_dir, roi_signals_dir, h5_out_field, activity_tresh):
    print '\n===== Generating active ground truths for subvolumes ======'
    print 'reading Calcium data (H5 volumes: %s, Matlab session files: %s, H5 data out name: %s, Activity threshold: %0.2f)' \
          % (h5_volumes_dir, matlab_session_dir, h5_out_field, activity_tresh)
    cd = JaneliaData(subvolumes_dir=h5_volumes_dir, session_files_dir=matlab_session_dir, load_precomputed_signals=True)
    for sub in cd.subvolumes:
        print 'processing', sub.session_name, sub.vvppp
        try:
            _ = sub.get(h5_out_field)
            print '\tactive ground truth for current subvolume already computed, skipping.'
            continue
        except:
            print '\tcomputing active ground truth'
            gt_active = sub.get_gt_from_active_rois_by_thresh(activity_tresh)
            print '\twriting active ground truth back to H5 file as "%s"' % h5_out_field
            sub.h5_write(h5_out_field, gt_active.reshape(512, 512))
    print 'Done.'


def save_labeled_gts_to_H5(h5_volumes_dir, matlab_session_dir, h5_out_field, labels):
    print '\n===== Generating active ground truths for subvolumes ======'
    print 'reading Calcium data (H5 volumes: %s, Matlab session files: %s, H5 data out name: %s, Activity labels: %s)' \
          % (h5_volumes_dir, matlab_session_dir, h5_out_field, str(labels))
    cd = JaneliaData(subvolumes_dir=h5_volumes_dir, session_files_dir=matlab_session_dir, dummy_data=False,
                     load_precomputed_signals=False)
    for sub in cd.subvolumes:
        print 'processing', sub.name
        try:
            _ = sub.get(h5_out_field)
            print '\tactive ground truth for current subvolume already computed, skipping.'
            continue
        except:
            print '\tbuilding active ground truth from labeling'
            gt_active = sub.get_activeGT_by_labels(labels=labels)
            print '\twriting active ground truth back to H5 file as "%s"' % h5_out_field
            sub.h5_write(h5_out_field, gt_active.reshape(sub.dims[0], sub.dims[1]))
    print 'Done.'


def save_cell_activity_plots(h5_volumes_dir, matlab_session_dir, out_dir, activity_thresh):
    cd = JaneliaData(subvolumes_dir=h5_volumes_dir, session_files_dir=matlab_session_dir)
    for sub in cd.subvolumes:
        print 'processing', sub.session_name, sub.vvppp
        for roi in sub.rois:
            print '\tgetting spike train for ROI %d' % roi.id
            roi.plot_spike_trains(activity_thresh)
            plt.savefig(
                out_dir + '%s.png' % (roi.meta_info['session_name'] + '_' + roi.meta_info['vvppp'] + '_' + str(roi.id)))
            plt.close()
            roi.plot_mad_thresholds_on_delta_f_estimates()
            plt.savefig(out_dir + '%s.png' % (
            roi.meta_info['session_name'] + '_' + roi.meta_info['vvppp'] + '_' + str(roi.id) + '_mad_threshs'))
            plt.close()
            roi.plot_baseline_fluorescence_estimated_quantiles()
            plt.savefig(out_dir + '%s.png' % (
            roi.meta_info['session_name'] + '_' + roi.meta_info['vvppp'] + '_' + str(roi.id) + '_f0_estimates'))
            plt.close()
            roi.plot_baseline_fluorescence()
            plt.savefig(out_dir + '%s.png' % (
            roi.meta_info['session_name'] + '_' + roi.meta_info['vvppp'] + '_' + str(roi.id) + '_f0_interpolates'))
            plt.close()
            roi.plot_f0_interpolation()
            plt.savefig(out_dir + '%s.png' % (
            roi.meta_info['session_name'] + '_' + roi.meta_info['vvppp'] + '_' + str(roi.id) + '_interpolation'))
            plt.close()
    print 'Done.'


def write_spike_trains_to_ROI_HDF5(rois_dir):
    cd = JaneliaData(load_precomputed_signals=True)
    for sub in cd.subvolumes:
        print 'Processing subvolume', sub.name
        h5out = h5.File(rois_dir + sub.name + '_rois.h5')
        for roi_id in h5out.iterkeys():
            print '\tROI', roi_id
            h5_grp = h5out[roi_id]
            try:
                _ = h5_grp['spike_trains']
                continue
            except:
                roi = sub.get_roi_by_id(roi_id)
                st = roi.spike_train
                h5_grp.create_dataset(name='spike_trains', data=st)
        h5out.close()
    print 'Done.'


def write_edges_to_PNG(out, edge_feature, channel):
    cd = JaneliaData()
    edges = create_2d_edge_graph(512, 512)
    for sub in cd.subvolumes:
        xcorr = sub.get('feats_xcorr_%s' % channel)
        img_h, img_v = visualize_edges(edges, xcorr, 0)
        plt.imsave(out + '%s_xcorr_h.png' % sub.name, img_h)
        plt.imsave(out + '%s_xcorr_v.png' % sub.name, img_v)
    print 'Done. All saved to', out


def write_roi_signals_to_HDF5(h5_volumes_dir, matlab_session_dir, out_dir, part):
    """ Create a HDF5 file for each subvolume containing all its ROIs and precomputed stuff for them (cell's raw signal,
    DeltaF/F, F0, ...).

    :param h5_volumes_dir:
    :param matlab_session_dir:
    :param h5_out_field:
    :return:
    """

    print 'reading Calcium data (H5 volumes: %s, Matlab session files: %s, H5 output dir: %s)' \
          % (h5_volumes_dir, matlab_session_dir, out_dir)
    cd = JaneliaData(subvolumes_dir=h5_volumes_dir, session_files_dir=matlab_session_dir)

    if part is not None:
        subvolumes = cd.subvolumes[part::5]
    else:
        subvolumes = cd.subvolumes

    overall_rois = int(np.sum([len(sub.rois) for sub in subvolumes]))
    overall_cnt = 0

    if part is not None:
        print 'PART %d/5: computing %d rois for %d subvolumes' % (part, overall_rois, len([sub for sub in subvolumes]))

    for sub in subvolumes:
        print 'processing volume:', sub.name, '| number of rois:', len(sub.rois)
        cnt = 0
        for roi in sub.rois:
            with h5.File(out_dir + sub.name + '_rois.h5', 'a') as h5out:
                cnt += 1
                overall_cnt += 1
                try:
                    _ = h5out[str(roi.id)]
                    print '\tRoi %d already computed, skipping.' % roi.id
                    continue
                except:
                    print '\tcomputing signals for roi with id:', str(roi.id), \
                        '(current: %d/%d, overall: %d/%d)' % (cnt, len(sub.rois), overall_cnt, overall_rois)
                    # compute ROI properties
                    properties = {
                        'indices': roi.indices,
                        'polygon': roi.polygon,
                        'centroid': roi.centroid,
                        'delta_f': roi.delta_ff,
                        'delta_f_estimates': roi.delta_ff_estimates,
                        'f_corrected': roi.f_corrected,
                        'f_neuropil': roi._f_neuropil,
                        'f_raw': roi.f_raw,
                        'f0_estimated': roi.f0_estimates,
                        'f0_interpolated': roi.f0_interpolates,
                        'prelim_event_series': roi.preliminary_events
                    }
                    # create HDF5 data sets within a ROI group
                    roi_grp = h5out.create_group(str(roi.id))
                    for k, v in properties.iteritems():
                        roi_grp.create_dataset(name=k, data=v)
    if part is not None:
        print 'PART %d/5: Computed signals for %d of %d ROIs. Done.' % (part + 1, overall_cnt, overall_rois)


def write_prob_maps_to_HDF5(forest_file_path, h5_field, channel):
    cd = JaneliaData()
    print 'Loading random forest from', forest_file_path
    rf = pickle.load(open(forest_file_path, 'r'))
    # channel = forest_file_path.split('/')[-1].split('random_forest_')[-1].split('_')[0]
    for sub in cd.subvolumes:
        print 'Processing', sub.name
        feats = sub.get('feats_ilastik_pseudoRGB_%s' % channel)
        print '\tpredicting...'
        pm = rf.predict_proba(feats.reshape((512 * 512, -1)))
        print '\tsaving prob. map to HDF5...'
        sub.h5_write(h5_field=h5_field, value=pm)
    print 'Done.'

def write_fftconv_to_HDF5():
    jd = JaneliaData(dummy_data=False, only_nice_volumes=False)
    edges = create_2d_edge_graph(512 ,512)
    cnt = 0
    for sv in jd.subvolumes:
        cnt += 1
        print 'computing FFT convolution for subvolume', sv.name, '(%d/%d)'%(cnt, len(jd.subvolumes))
        fftconv = sv.compute_pixel_fftconvolution(channel='green')
        norm_fftconv = normalize_image_to_zero_one(fftconv)
        h, v = visualize_edges(edges, norm_fftconv, 0)
        create_out_path('out/fftconv/')
        plt.imsave('out/fftconv/%s_fftconv_h.png'%sv.name, h, cmap='gray')
        plt.imsave('out/fftconv/%s_fftconv_v.png'%sv.name, v, cmap='gray')
        sv.h5_write(h5_field='feats_fftconv_green', value=norm_fftconv)
    print 'Done.'


def process_command_line():
    # create a parser instance
    parser = argparse.ArgumentParser(description="Core scripts for HDF5 maintenance of Calcium imaging data.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # add arguments to parser
    parser.add_argument('--save_active_gts_to_H5', action='store_true', default=False,
                        help='Given a directory with H5 volume files, compute the active ground truth of each volume and save it back to the H5 file under the field given by --h5_out_field.')
    parser.add_argument('--save_active_gt_by_label_to_H5', nargs='+',
                        choices=['active_very_certain', 'active_mod_certain', 'uncertain', 'inactive_very_certain',
                                 'inactive_auto'], default=False,
                        help='Given a directory with H5 volume files, compute the active ground truth of each volume and save it back to the H5 file under the field given by --h5_out_field.')
    parser.add_argument('--h5_volumes_dir', type=str, default=False)
    parser.add_argument('--matlab_session_dir', type=str, default=False)
    parser.add_argument('--roi_signals_dir', type=str, default=False)
    parser.add_argument('--h5_out_field', type=str, default=False)
    parser.add_argument('--active_gts_thresh', type=float, default=0.06,
                        help='Set threshold above which the spike trains have to max for being called "active cell".')

    parser.add_argument('--build_volumes', action='store_true', default=False,
                        help='Build H5 volumes from CRCNS SSC-1 data dump. Afterwards each (sub)volume contains time series Calcium data, projections, features, ground truths, etc.')
    parser.add_argument('--data_path', type=str, default=False)

    parser.add_argument('--save_cell_activity_plots', action='store_true', default=False,
                        help='Save plots as PNG for every ROI in the data to "--out-dir". Plots: spike trains, MAD thresholds, F0 estimates and interpolation.')
    parser.add_argument('--out_dir', type=str, default=False)

    parser.add_argument('--write_roi_signals_to_HDF5', action='store_true', default=False,
                        help="Create a HDF5 file for each subvolume containing all its ROIs and precomputed stuff for them (cell's raw signal, DeltaF/F, F0, ...).")
    parser.add_argument('--part', type=int, default=None, choices=[0, 1, 2, 3, 4])

    parser.add_argument('--channel', type=str, default='green', choices=['green', 'red'])

    parser.add_argument('--write_spike_trains', action='store_true', default=False)

    parser.add_argument('--save_edge_features_as_PNG', action='store_true', default=False)
    parser.add_argument('--edge_feature', type=str, default=False, choices=['xcorr', 'corr', 'fftconv'])

    parser.add_argument('--generate_prob_maps', action='store_true', default=False, help='Generate a probability map for every subvolume and write it back to the underlying HDF5 file. Use random forest model (trained and pickled), path specified by --random_forest_pickle. Create dataset with name specified by --h5_out_field.')
    parser.add_argument('--random_forest_pickle', action='store_true', default=False, help='Path to pickle file containing a trained random forest.')

    parser.add_argument('--write_fftconv_to_HDF5', action='store_true', default=False)

    # parse arguments
    args = parser.parse_args()
    return args


def main():
    args = process_command_line()

    if args.generate_prob_maps:
        if args.random_forest_pickle and args.h5_out_field:
            write_prob_maps_to_HDF5(forest_file_path=args.random_forest_pickle, h5_field=args.h5_out_field, channel=args.channel)
            return 0
    else:
        print 'Call --help to see params for generating prob. maps.'
        return 1

    if args.build_volumes:
        from scripts.build_hdf5_volumes_janelia import build_volumes
        if args.h5_volumes_dir and args.data_path:
            build_volumes(data_path=args.data_path,
                          h5_out_path=args.h5_volumes_dir)
            return 0
        else:
            print '[Missing parameter(s)] Point "--data-path" to (unzipped) raw Janelia Calcium data. Point "--h5_volumes_dir" to path where to save the final volumes.'
            return 1

    if args.save_active_gts_to_H5:
        if args.h5_volumes_dir and args.matlab_session_dir and args.h5_out_field:
            save_active_gts_to_H5(h5_volumes_dir=args.h5_volumes_dir,
                                  matlab_session_dir=args.matlab_session_dir,
                                  roi_signals_dir=args.roi_signals_dir,
                                  h5_out_field=args.h5_out_field,
                                  activity_tresh=args.active_gts_thresh)
            return 0
        else:
            print '[Missing parameter(s)] Point "--h5_volumes_dir" and "--matlab_session_dir" to directories with (preprocessed) volumes and session files respectively. Give H5 dataset name for saving with "--h5_out_field".'
            print 'Example usage: \n\tpython hdf5_tool.py --save_active_gts_to_H5 --h5_volumes_dir data/preprocessed/final/ --matlab_session_dir data/preprocessed/session_files/ --h5_out_field gt_active_test2 --active_gts_thresh 0.06'
            return 1

    if args.save_cell_activity_plots:
        if args.h5_volumes_dir and args.matlab_session_dir and args.out_dir:
            save_cell_activity_plots(h5_volumes_dir=args.h5_volumes_dir,
                                     matlab_session_dir=args.matlab_session_dir,
                                     out_dir=args.out_dir,
                                     activity_thresh=args.active_gts_thresh)
            return 0
        else:
            print '[Missing parameter(s)] Point "--h5_volumes_dir" and "--matlab_session_dir" to directories with (preprocessed) volumes and session files respectively. Point "--out_dir" to output directory.'
            return 1

    if args.write_roi_signals_to_HDF5:
        if args.h5_volumes_dir and args.matlab_session_dir and args.out_dir:
            write_roi_signals_to_HDF5(h5_volumes_dir=args.h5_volumes_dir,
                                      matlab_session_dir=args.matlab_session_dir,
                                      out_dir=args.out_dir,
                                      part=args.part)
            return 0
        else:
            print '[Missing parameter(s)] Point "--h5_volumes_dir" and "--matlab_session_dir" to directories with (preprocessed) volumes and session files respectively. Point "--out_dir" to output directory.'
            return 1

    if args.write_spike_trains and args.roi_signals_dir:
        write_spike_trains_to_ROI_HDF5(args.roi_signals_dir)
        return 0

    if args.save_edge_features_as_PNG:
        if args.out_dir and args.edge_feature:
            write_edges_to_PNG(args.out_dir, args.edge_feature, channel=args.channel)
            return 0
        else:
            print '[Computer says no]'
            return 1

    if args.save_active_gt_by_label_to_H5:
        if args.h5_volumes_dir and args.matlab_session_dir and args.h5_out_field:
            save_labeled_gts_to_H5(h5_volumes_dir=args.h5_volumes_dir,
                                   matlab_session_dir=args.matlab_session_dir,
                                   h5_out_field=args.h5_out_field,
                                   labels=args.save_active_gt_by_label_to_H5)
            return 0
        else:
            print '[Computer says no]'
            return 1

    if args.write_fftconv_to_HDF5:
        write_fftconv_to_HDF5()
        return 0

    return 0


if __name__ == "__main__":
    status = main()
    sys.exit(status)