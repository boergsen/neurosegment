from core.data.calcium_imaging import JaneliaData
from core.visuals import plot_pr_curve_on_axis, plot_roc_curve_on_axis
import matplotlib.pyplot as plt
import h5py as h5
import os
import numpy as np

out_dir='/home/herrgilb/thesis/code/out'
out_name = 'eval_activity_labels_spike-train-threshs-to-10_fmax160'
h5_out_path = os.path.join(out_dir, '%s.h5'%out_name)

try:
    with h5.File(h5_out_path, 'r') as hin:
        threshs = np.asarray(hin['thresholds'])
        thresh_scores = np.asarray(hin['thresh_scores'])
        p = np.asarray(hin['precisions'])
        r = np.asarray(hin['recalls'])
        acc = np.asarray(hin['accuracies'])
        tpr = np.asarray(hin['tpr'])
        fpr = np.asarray(hin['fpr'])
except:
    jd = JaneliaData(dummy_data=False, load_precomputed_signals=True)
    threshs, thresh_scores, p, r, acc, tpr, fpr = jd.eval_activity_labels(h5_out_path=h5_out_path, verbose=0)

fig, (ax0, ax1, ax2) = plt.subplots(1,3)
ax0 = plot_pr_curve_on_axis(p, r, label_random='random', ax=ax0)
ax1 = plot_roc_curve_on_axis(fpr, tpr, label_random='random', ax=ax1)
ax2.plot(threshs[:251], acc[:251], color='g')
ax2.set_xlabel('Spike train threshold')
ax2.set_ylabel('Accuracy')
fig.set_size_inches(13,4)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, '%s_plot.png'%out_name), dpi=300)
plt.show()
import IPython
IPython.embed()
#jd.plot_pie_roi_activity()
#jd.plot_hist_roi_activity()
