__version__ = '1.0'

import h5py as h5
import numpy as np
import sys

import matplotlib.pyplot as plt

from core.data.calcium_imaging import JaneliaData


# noinspection PyBroadException
def eval_rois(calcium_data):
    """Evaluate neuron activity.
    """

    def key_down(event):
        # quit the application by pressing escape
        if event.key == "escape":
            sys.exit(0)

        # only continue if 1, 2, 3 or 4 is pressed
        if event.key not in ["1", "2", "3", "4"]:
            return
        k = int(event.key)

        if k == 1:
            label = "active_very_certain"
            print "Neuron marked as being certainly active"
        elif k == 2:
            label = "active_mod_certain"
            print "Moderately certain that neuron is active"
        elif k == 3:
            label = "uncertain"
            print "Uncertain if neuron is active or not"
        else:  # k == 4:
            label = "inactive_very_certain"
            print "Neuron marked as being certainly inactive"

        roi.write_back_to_hdf5(h5_field='activity_label', content=label)

        # close the current figure (and show the next one)
        plt.close()

    for sub in calcium_data.subvolumes:
        print sub.name
        mean = sub.get('mean_green')
        xcorr_h = sub.visualize_xcorr('green')[0]
        xcorr_v = sub.visualize_xcorr('green')[1]
        cnt = 0
        for roi in sub.rois:
            cnt += 1
            print 'ROI ID:', str(roi.id), '(%d/%d)' % (cnt, len(sub.rois))
            try:
                with h5.File(roi.subvolume.h5_roi_signals_path, 'r') as h5_in:
                    roi_grp = h5_in[str(roi.id)]
                    _ = roi_grp['activity_label']
                print 'Activity label already set.'
                continue
            except:
                print "Load precomputed signals"
                roi.load_precomputed_signals()

                raw_max = np.max(roi.f_corrected)
                st = roi.spike_train
                if raw_max < 80 or \
                        (raw_max < 120 and np.max(st[:, 0]) < 0.1 and np.max(st[:, 1]) < 0.1 and np.max(st[:, 2]) < 0.1):
                    print "Automatically annotating with label 'inactive_auto'"
                    roi.write_back_to_hdf5(h5_field='activity_label', content='inactive_auto')
                    continue

                fig = plt.figure()
                # fig.set_size_inches(13,18)
                gs = plt.GridSpec(9, 3)
                fig.suptitle(str(roi.meta_info) + ' | ID: ' + str(roi.id))

                # trigger initialization
                st = roi.spike_train
                t = roi.time_scale
                cx, cy = roi.centroid

                # upper row
                ax00 = plt.subplot(gs[0:4, 0])
                ax01 = plt.subplot(gs[0:4, 1])
                ax02 = plt.subplot(gs[0:4, 2])

                # lower row
                ax10 = plt.subplot(gs[5, :])
                ax20 = plt.subplot(gs[6, :], sharex=ax10)
                ax30 = plt.subplot(gs[7, :], sharex=ax20)
                ax40 = plt.subplot(gs[8, :])

                ax00.imshow(mean)
                ax00.add_patch(plt.Polygon(roi.polygon, color='r', fill=False))
                ax00.set_xticks([])
                ax00.set_yticks([])
                ax00.set_xlim(cx - 50, cx + 50)
                ax00.set_ylim(cy - 50, cy + 50)

                ax01.imshow(xcorr_h)
                ax01.add_patch(plt.Polygon(roi.polygon, color='r', fill=False))
                ax01.set_xticks([])
                ax01.set_yticks([])
                ax01.set_xlim(cx - 50, cx + 50)
                ax01.set_ylim(cy - 50, cy + 50)

                ax02.imshow(xcorr_v)
                ax02.add_patch(plt.Polygon(roi.polygon, color='r', fill=False))
                ax02.set_xticks([])
                ax02.set_yticks([])
                ax02.set_xlim(cx - 50, cx + 50)
                ax02.set_ylim(cy - 50, cy + 50)

                ymax = 0.5
                ax10.plot(t, st[:, 0], '-k')
                ax10.set_ylabel(r'$\hat{n}$', fontsize='large')
                ax10.set_xlim(0, t[-1])
                ax10.set_ylim(0, ymax)
                ax10.set_xticks([])
                ax10.axhline(0.1, color='g', linestyle='--')

                ax20.plot(t, st[:, 1], '-k')
                ax20.set_ylabel(r'$\hat{n}$', fontsize='large')
                ax20.set_xlim(0, t[-1])
                ax20.set_ylim(0, ymax)
                ax20.set_xticks([])
                ax20.axhline(0.1, color='g', linestyle='--')

                ax30.plot(t, st[:, 2], '-k')
                ax30.set_ylabel(r'$\hat{n}$', fontsize='large')
                ax30.set_xlim(0, t[-1])
                ax30.set_ylim(0, ymax)
                ax30.set_xticks([])
                ax30.axhline(0.1, color='g', linestyle='--')

                ax40.hold(True)
                # fig.tight_layout()
                ax40.plot(t, roi.f_corrected, 'g', label='raw signal (corrected)')
                ax40.plot(t, roi.f0_interpolates[:, 0], 'r', label='$F_0$ (50%tile)')
                ax40.plot(t, roi.f0_interpolates[:, 1], 'c', label='$F_0$ (20%tile)')
                ax40.plot(t, roi.f0_interpolates[:, 2], 'b', label='$F_0$ (5%tile)')
                # plt.legend(loc=1, fancybox=True, fontsize='small')

                ax40.set_xlabel('Time (s)')
                ax40.set_ylabel('F')
                ax40.set_xlim(0, t[-1])
                ax40.hold(False)

                fig.canvas.mpl_connect("key_press_event", key_down)
                plt.show()

if __name__ == '__main__':
    cd = JaneliaData()
    eval_rois(cd)
    sys.exit(0)