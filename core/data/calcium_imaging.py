"""calcium_imaging.py: Wrappers for handling and visualizing Calcium imaging data."""

__version__ = '0.99'

# built-in
import cPickle as pickle
import os

# 3rd party
import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
from scipy.io import loadmat

# own
from core.data.subvolumes import JaneliaSubvolume, NeurofinderSubvolume
from core.data.rois import JaneliaRoi


class CalciumDataAbstract(object):
    pass


class JaneliaData(CalciumDataAbstract):
    """Wrapper for accessing the Janelia Calcium imaging data saved in HDF5 files on disk."""

    _default_h5_subvolume_files_dir = os.getenv("HOME") + '/thesis/code/data/full/final/'
    _default_matlab_session_files_dir = os.getenv("HOME") + '/thesis/code/data/full/session_files/'
    _default_roi_signals_dir = os.getenv("HOME") + '/thesis/code/data/full/rois_labeled/'
    _default_random_forest_features_dir = os.getenv("HOME") + '/thesis/code/data/full/features/'

    def __init__(self, subvolumes_dir=None, session_files_dir=None, roi_signals_dir=None, uncorrupted=False,
                 dummy_data=False, load_precomputed_signals=False, only_nice_volumes=False, only_demo_volume=False):
        """Constructor: Spawn Subvolume objects from Janelia session information and pre-built HDF5 files.

        :param subvolumes_dir: Path to directory with subvolume HDF5 files (as created by build_hdf5_volumes_janelia.py from CCRNS SSC-1 data).
        :param session_files_dir: Path to directory with Janelia session files (in Matlab format)
        :param roi_signals_dir: Path to directory with HDF5 files containing precomputed signals for every ROI.
        :param dummy_data: If True, only load a small subset of subvolumes for development purposes.
        :param load_precomputed_signals: Load all precomputed ROI signals beforehand instead of on demand.
        :return: void
        """

        # load default data directories
        if subvolumes_dir is None:
            subvolumes_dir = self._default_h5_subvolume_files_dir
        if session_files_dir is None:
            session_files_dir = self._default_matlab_session_files_dir
        if roi_signals_dir is None:
            roi_signals_dir = self._default_roi_signals_dir
        print 'Subvolumes: %s \nSession files: %s \nROI signals: %s \nRandom Forest features: %s\n' % \
              (subvolumes_dir, session_files_dir, roi_signals_dir, self._default_random_forest_features_dir)

        # ------------------------------------------------------------
        # session_ROI_dict = { Janelia session name : { Janelia session_ID : Roi object } }
        # ------------------------------------------------------------
        try:
            print 'Trying to load Session-JaneliaID-ROI mapping from disk...'
            session_ROI_dict = pickle.load(open(session_files_dir + 'session_roi_dict.p', 'r'))
        except:
            print 'Not found: extracting Session-JaneliaID-ROI mapping from Matlab dump...'
            session_ROI_dict = self._build_ROI_dict(session_files_dir)  # session_ROI_dict = { sessionName : { ROI_ID : Roi Object } }
            try:
                print 'Dumping mapping for future use to %s' % (session_files_dir + 'session_roi_dict.p')
                pickle.dump(session_ROI_dict, open(session_files_dir + 'session_roi_dict.p', 'w'))
            except:
                raise IOError('[Error] could not save dict to pickle file!')

        # ----------
        # SUBVOLUMES
        # ----------
        if dummy_data:
            h5_subvolume_file_names = [
                                        'an197522_2013_03_10_fov_13002.h5',
                                        'an197522_2013_03_08_fov_06003.h5',
                                        'an229719_2013_12_05_fov_03004.h5',
                                        'an229719_2013_12_05_fov_07003.h5',
                                        'an229719_2013_12_05_fov_08004.h5',
                                      ]
            self.use_dummy_data = True
        elif only_demo_volume:
            h5_subvolume_file_names = [ 'an197522_2013_03_10_fov_13002.h5' ]
        elif uncorrupted:
            h5_subvolume_file_names = [f for f in os.listdir(subvolumes_dir) if f.endswith('.h5') and not f.endswith('_corrupted.h5')]
        else:
            h5_subvolume_file_names = [f for f in os.listdir(subvolumes_dir) if f.endswith('.h5') and not f.endswith('corrupted.h5')]
            self.use_dummy_data = False

        h5_subvolume_file_paths = [subvolumes_dir + h5file for h5file in h5_subvolume_file_names]
        h5_subvolume_file_paths.sort()

        print 'Creating subvolume objects...'
        self.subvolumes = [JaneliaSubvolume(h5_file_path, session_ROI_dict, roi_signals_dir) for h5_file_path in h5_subvolume_file_paths]

        # optional
        if only_nice_volumes:
            print "Using only 'nice' subvolumes..."
            self.subvolumes = [s for s in self.subvolumes if s.is_nice()]
            self.use_only_nice = True
        else:
            self.use_only_nice = False

        if load_precomputed_signals:
            print 'Loading precomputed ROI signals.... '
            for s in self.subvolumes:
                for r in s.rois:
                    r.load_precomputed_signals()

    @property
    def ssvm_data_cherry_picked(self):
        is_cherry = {
            'an197522_2013_03_08_01002': True, 'an197522_2013_03_08_01003': True, 'an197522_2013_03_08_01004': True,
            'an197522_2013_03_08_02002': True, 'an197522_2013_03_08_02003': True, 'an197522_2013_03_08_02004': True,
            'an197522_2013_03_08_03004': True, 'an197522_2013_03_08_05004': True, 'an197522_2013_03_08_06003': True,
            'an197522_2013_03_08_06004': True, 'an197522_2013_03_08_07003': True, 'an197522_2013_03_08_08002': True,
            'an197522_2013_03_08_08003': False, 'an197522_2013_03_08_08004': False, 'an197522_2013_03_08_09002': True,
            'an197522_2013_03_08_09003': False, 'an197522_2013_03_08_09004': False, 'an197522_2013_03_08_10002': False,
            'an197522_2013_03_08_10003': False, 'an197522_2013_03_08_10004': False, 'an197522_2013_03_10_11002': False,
            'an197522_2013_03_10_11003': False, 'an197522_2013_03_10_11004': False, 'an197522_2013_03_10_12002': True,
            'an197522_2013_03_10_13002': True, 'an197522_2013_03_10_14003': True, 'an197522_2013_03_10_14004': True,
            'an197522_2013_03_10_15002': True, 'an197522_2013_03_10_15003': False, 'an197522_2013_03_10_15004': True,
            'an197522_2013_03_10_16002': True, 'an197522_2013_03_10_16003': True, 'an197522_2013_03_10_16004': False,
            'an197522_2013_03_10_17002': False, 'an197522_2013_03_10_17003': False, 'an197522_2013_03_10_17004': False,
            'an197522_2013_03_10_18002': False, 'an197522_2013_03_10_18003': False, 'an197522_2013_03_10_18004': False,
            'an197522_2013_03_10_19002': False, 'an197522_2013_03_10_19003': False, 'an229717_2013_12_01_01003': False,
            'an229717_2013_12_01_01004': True, 'an229717_2013_12_01_02002': False, 'an229717_2013_12_01_02003': False,
            'an229717_2013_12_01_02004': False, 'an229717_2013_12_01_04002': True, 'an229717_2013_12_01_04003': True,
            'an229717_2013_12_01_07002': True, 'an229717_2013_12_01_07003': True, 'an229717_2013_12_01_07004': True,
            'an229719_2013_12_02_06002': True, 'an229719_2013_12_02_09002': False, 'an229719_2013_12_02_09003': True,
            'an229719_2013_12_02_09004': False, 'an229719_2013_12_02_10002': True, 'an229719_2013_12_02_12002': False,
            'an229719_2013_12_02_12003': False, 'an229719_2013_12_02_12004': True, 'an229719_2013_12_02_15002': True,
            'an229719_2013_12_02_15003': False, 'an229719_2013_12_02_15004': False, 'an229719_2013_12_05_02002': False,
            'an229719_2013_12_05_02003': True, 'an229719_2013_12_05_02004': True, 'an229719_2013_12_05_03002': True,
            'an229719_2013_12_05_03003': False, 'an229719_2013_12_05_03004': True, 'an229719_2013_12_05_05003': True,
            'an229719_2013_12_05_05004': True, 'an229719_2013_12_05_06002': True, 'an229719_2013_12_05_06003': True,
            'an229719_2013_12_05_06004': True, 'an229719_2013_12_05_07002': True, 'an229719_2013_12_05_07003': True,
            'an229719_2013_12_05_07004': True, 'an229719_2013_12_05_08002': True, 'an229719_2013_12_05_08003': True,
            'an229719_2013_12_05_08004': False
        }
        return [sub for sub in self.subvolumes if (is_cherry[sub.name] and sub in self.nice_subs)]

    @property
    def ssvm_test_set(self):
        test_volumes = [
            'an197522_2013_03_08_01002', 'an197522_2013_03_08_01004', 'an197522_2013_03_08_02002', 'an197522_2013_03_08_02004',
            'an197522_2013_03_08_03004', 'an197522_2013_03_08_06003', 'an197522_2013_03_08_06004', 'an197522_2013_03_08_08002',
            'an197522_2013_03_08_09003', 'an197522_2013_03_10_12002', 'an197522_2013_03_10_14004', 'an197522_2013_03_10_15002',
            'an197522_2013_03_10_16002', 'an197522_2013_03_10_16003', 'an197522_2013_03_10_17003', 'an229717_2013_12_01_01003',
            'an229717_2013_12_01_01004', 'an229717_2013_12_01_02003', 'an229717_2013_12_01_02004', 'an229717_2013_12_01_04003',
            'an229717_2013_12_01_07002', 'an229717_2013_12_01_07004', 'an229719_2013_12_02_06002', 'an229719_2013_12_05_02002',
            'an229719_2013_12_05_02003', 'an229719_2013_12_05_03002', 'an229719_2013_12_05_03003', 'an229719_2013_12_05_05003',
            'an229719_2013_12_05_05004', 'an229719_2013_12_05_06003', 'an229719_2013_12_05_06004', 'an229719_2013_12_05_07003',
            'an229719_2013_12_05_07004', 'an229719_2013_12_05_08003', 'an229719_2013_12_05_08004', 'an229719_2013_12_05_07002',
            'an197522_2013_03_10_13002', 'an229719_2013_12_05_03004', 'an197522_2013_03_08_09004', 'an229719_2013_12_05_08002'
        ]
        return [s for s in self.subvolumes if s.name in test_volumes]

    @property
    def ssvm_train_set(self):
        train_volumes = [
            'an197522_2013_03_08_01003', 'an197522_2013_03_08_02003', 'an197522_2013_03_08_05004', 'an197522_2013_03_08_07003',
            'an197522_2013_03_10_14003', 'an197522_2013_03_10_15004', 'an229717_2013_12_01_02002', 'an229717_2013_12_01_04002',
            'an229717_2013_12_01_07003', 'an229719_2013_12_02_10002', 'an229719_2013_12_05_02004', 'an229719_2013_12_05_06002',
        ]
        return [s for s in self.subvolumes if s.name in train_volumes]

    @property
    def nice_subs(self):
        nice_volumes = [
            'an197522_2013_03_08_01002', 'an197522_2013_03_08_01003', 'an197522_2013_03_08_01004',
            'an197522_2013_03_08_02002', 'an197522_2013_03_08_02003', 'an197522_2013_03_08_02004',
            'an197522_2013_03_08_03004', 'an197522_2013_03_08_05004', 'an197522_2013_03_08_06003',
            'an197522_2013_03_08_06004', 'an197522_2013_03_08_07003', 'an197522_2013_03_08_08002',
            'an197522_2013_03_08_09003', 'an197522_2013_03_08_09004', 'an197522_2013_03_10_12002',
            'an197522_2013_03_10_13002', 'an197522_2013_03_10_14003', 'an197522_2013_03_10_14004',
            'an197522_2013_03_10_15002', 'an197522_2013_03_10_15004', 'an197522_2013_03_10_16002',
            'an197522_2013_03_10_16003', 'an197522_2013_03_10_17003', 'an229717_2013_12_01_01003',
            'an229717_2013_12_01_01004', 'an229717_2013_12_01_02002', 'an229717_2013_12_01_02003',
            'an229717_2013_12_01_02004', 'an229717_2013_12_01_04002', 'an229717_2013_12_01_04003',
            'an229717_2013_12_01_07002', 'an229717_2013_12_01_07003', 'an229717_2013_12_01_07004',
            'an229719_2013_12_02_06002', 'an229719_2013_12_02_10002', 'an229719_2013_12_05_02002',
            'an229719_2013_12_05_02003', 'an229719_2013_12_05_02004', 'an229719_2013_12_05_03002',
            'an229719_2013_12_05_03003', 'an229719_2013_12_05_03004', 'an229719_2013_12_05_05003',
            'an229719_2013_12_05_05004', 'an229719_2013_12_05_06002', 'an229719_2013_12_05_06003',
            'an229719_2013_12_05_06004', 'an229719_2013_12_05_07002', 'an229719_2013_12_05_07003',
            'an229719_2013_12_05_07004', 'an229719_2013_12_05_08002', 'an229719_2013_12_05_08003',
            'an229719_2013_12_05_08004'
        ]
        return [s for s in self.subvolumes if s.name in nice_volumes]

    @property
    def dummy_subs(self):
        demo_volumes = [
            'an197522_2013_03_10_13002',
            'an197522_2013_03_08_06003',
            'an229719_2013_12_05_03004',
            'an229719_2013_12_05_07003',
            'an229719_2013_12_05_08004'
        ]
        return [s for s in self.subvolumes if s.name in demo_volumes]

    @property
    def demo_volume(self):
        return self.get_subvolume_by_name('an197522_2013_03_10_13002')

    @property
    def session_names(self):
        return list(set([s.session_name for s in self.subvolumes]))

    @property
    def n_subvolumes(self):
        return len(self.subvolumes)

    @property
    def n_samples(self):
        return self.n_subvolumes * self.subvolumes[0].dims[0] * self.subvolumes[0].dims[1]

    @property
    def average_and_min_max_n_frames(self, verbose=0):
        n_frames = 0
        min_frames = np.inf
        max_frames = 0
        for s in self.subvolumes:
            sr = s.rois[0]
            sr.load_precomputed_signals()
            sr_frames = sr.f_raw.shape[0]
            n_frames += sr_frames
            if sr_frames < min_frames:
                min_frames = sr_frames
                min_vol_name = s.name
            if sr_frames > max_frames:
                max_frames = sr_frames
                max_vol_name = s.name
        average_frames = n_frames/self.n_subvolumes
        if verbose > 0:
            print 'Subvolume with the most frames (%d): %s' % (max_frames, max_vol_name)
            print 'Subvolume with the least frames (%d): %s' % (min_frames, min_vol_name)
            print 'Average number of frames:',
        return average_frames, min_frames, max_frames


    def _build_ROI_dict(self, session_files_dir):
        """Create a dict of dicts where keys are session file names and values are separate ROI dicts extracted from the
        respective session file (containing a Janelia ID to Roi object mapping).

        :return: Dict with Janelia session tag as key and a ROI-session-ID to ROI-python-object mapping as value.
        """
        print 'Loading Janelia session files...'
        session_files = os.listdir(session_files_dir)
        session_file_paths = [session_files_dir + s for s in session_files]
        matlab_files = [loadmat(path) for path in session_file_paths]
        session_names = [s.split('_session')[0] for s in session_files]
        print 'Building Session-to-JaneliaID-to-Roi mapping'
        session_ROI_dict = {}
        for s in xrange(len(matlab_files)):
            # NOTE: Missing parameters (subvolume, meta_info) will be set when adding
            # Roi objects to a Subvolume instance (see Subvolume._create_rois_for_subvolume())
            session_ROIs = self._extract_ROIs_dict_from_session_object(matlab_files[s], subvolume=None, meta_info=None)
            session_ROI_dict[session_names[s]] = session_ROIs
        return session_ROI_dict

    @staticmethod
    def _extract_ROIs_dict_from_session_object(matlab_session_object, subvolume, meta_info):
        """Takes a session file in .mat format (MATLAB) and extracts all ROIs (=marked neurons) from it.
        Returns a dict with Janelia ID as key (valid within a session file) and Roi object as value.

        NOTE: For a Roi object, at this point, only its pixel indices, polygon shape and Janelia ID are known.
        Missing parameters (subvolume, meta_info) will be set when adding Roi objects to a Subvolume instance
        (see Subvolume._create_rois_for_subvolume())

        # ---------------------------------------------------------------------------
        # Note to future me: If you are about to load .mat objects into Python again,
        # for God's sake, use squeeze_me=True and struct_as_record=False in loadmat().
        # ---------------------------------------------------------------------------

        :param matlab_session_object:
        :param subvolume:
        :param meta_info:
        :return: rois dict
        """
        # convenience
        descrHashField = matlab_session_object['s']['timeSeriesArrayHash'][0, 0][0, 0][3][0]
        rois = {}
        for i in xrange(1, len(descrHashField)):  # for every fov
            current_fov_value = descrHashField[i][0, 0][2][0, 0][0]
            no_of_fields_in_fov = len(current_fov_value)
            for j in xrange(no_of_fields_in_fov):
                rois_field = current_fov_value[j][2]  # every ROI has a set of pixels marking a neuron
                for k in xrange(rois_field.shape[1]):  # every roi is a field of fields
                    roi = rois_field[0, k]
                    roi_id = int(roi[0].squeeze())
                    roi_pixel = roi[2].squeeze()
                    roi_xy = roi[1]
                    rois[roi_id] = JaneliaRoi(indices=roi_pixel, xy=roi_xy, janelia_id=roi_id,
                                              subvolume=subvolume, meta_info=meta_info) # see note above: gets filled out later
        return rois

    def get_subvolume_by_name(self, name):
        """Return a subvolume by its name (Janelia session tag, e.g. 'an229719_2013_12_02_06003').

        :param name: Name of the subvolume to be returned.
        :return: Subvolume object if name matches, KeyError otherwise.
        """
        for s in self.subvolumes:
            if s.name == name:
                return s
        raise KeyError('[Error] subvolume with name %s not found. Typo?' % name)

    def plot_hist_roi_activity(self, kind='area'):
        assert kind in ['area', 'bar', 'barh'], ValueError("valid arguments for 'kind' are 'area', 'bar' or 'barh'")
        import pandas as pd
        pd.options.display.mpl_style = 'default'
        stats = []
        for sub in self.subvolumes:
            stats.append(sub.get_roi_activity_label_counts()[0])
        df = pd.DataFrame(stats, columns=self.subvolumes[0].activity_labels_short)
        colors = [self.subvolumes[0].mpl_activity_label_colors[label] for label in self.subvolumes[0].activity_labels]
        ax = df.plot(kind=kind, stacked=True, colors=colors)
        ax.set_xlabel('Subvolume')
        ax.set_ylabel('Number of labeled ROI')
        return ax

    def plot_pie_roi_activity(self, ax=None, mpl_style='default'):
        import pandas as pd
        pd.options.display.mpl_style = mpl_style
        stats = []
        for sub in self.subvolumes:
            stats.append(sub.get_roi_activity_label_counts()[0])
        stats = np.asarray(stats)
        sums = [np.sum(stats[:, i]) for i in xrange(5)]
        labels = self.subvolumes[0].activity_labels_short
        colors = [self.subvolumes[0].mpl_activity_label_colors[label] for label in self.subvolumes[0].activity_labels]
        if ax is None:
            ax = plt.gca()
        ax.pie(sums, labels=['','','','',''], colors=colors, autopct='%1.1f%%')
        ax.legend(loc=0, labels=labels)
        return ax

    def eval_activity_labels(self, h5_out_path=None, verbose=0):
        from core.methods import precision, recall, accuracy
        from core.methods import fpr, tpr
        thresh_scores = []
        stats = []
        threshs = np.arange(0, 10.01, .01)
        n_rois = 0
        for sub in self.subvolumes:
            for roi in sub.rois:
                n_rois += 1
                #roi.load_precomputed_signals()
                res = roi.is_active_by_thresh(return_stats=threshs, verbose=verbose)
                stats.append(res)
        # shape: (rois, threshs, scores=(tp, fp, tn, fn, unsure))'
        stats = np.asarray(stats)
        # res.shape:
        for i in xrange(len(threshs)):
            roi_scores_for_thresh = stats[:, i]
            thresh_scores.append(np.sum(roi_scores_for_thresh, axis=0))
        # shape: (thresh, confusion_val)
        thresh_scores = np.asarray(thresh_scores)
        tps = thresh_scores[:, 0]
        fps = thresh_scores[:, 1]
        tns = thresh_scores[:, 2]
        fns = thresh_scores[:, 3]
        p = precision(tp=tps, fp=fps)
        r = recall(tp=tps , fn=fns)
        acc = accuracy(tps, fps, tns, fns)
        tprs = tpr(tps, fns)
        fprs = fpr(fps, tns)

        print '------------------------------------------------------------\n'
        print 'Comparison of manually labeled rois with activity labels \n' \
              'derived by thresholding spike trains.'
        print 'overall rois:', n_rois
        print 'overall rois (without unsure):', n_rois - thresh_scores[0, 4]
        print 'tp, fp, tn, fn, unsure:', thresh_scores[0, :]
        for i in xrange(len(threshs)):
            print '-----------------'
            print 'threshold:', threshs[i]
            print '   precision:', p[i]
            print '   recall:', r[i]
            print '   accuracy:', acc[i]
            print '   tpr:', tprs[i]
            print '   fpr:', fprs[i]
        print '------------------------------------------------------------\n'

        stats2 = []
        for sub in self.subvolumes:
            stats2.append(sub.get_roi_activity_label_counts()[0])
        stats2 = np.asarray(stats2)
        sums = np.asarray([np.sum(stats2[:, i]) for i in xrange(5)])
        labels = self.subvolumes[0].activity_labels_short
        for i in xrange(len(labels)):
            print 'label "%s": %d' % (labels[i], int(sums[i]))
        if h5_out_path is not None:
            print 'write results to', h5_out_path
            with h5.File(h5_out_path) as h5out:
                h5out.create_dataset(name='thresh_scores', data=thresh_scores)
                h5out.create_dataset(name='precisions', data=p)
                h5out.create_dataset(name='recalls', data=r)
                h5out.create_dataset(name='accuracies', data=acc)
                h5out.create_dataset(name='n_rois', data=n_rois)
                h5out.create_dataset(name='roi_confusion_vals_per_thresh', data=stats)
                h5out.create_dataset(name='label_counts', data=sums)
                h5out.create_dataset(name='thresholds', data=threshs)
                h5out.create_dataset(name='tpr', data=tprs)
                h5out.create_dataset(name='fpr', data=fprs)
        return threshs, thresh_scores, p, r, acc, tprs, fprs

    def get_average_session_image(self, session_name, image='pseudorgb', channel='green'):
        from core.methods import normalize_image_to_zero_one as normalize
        assert session_name in self.session_names
        assert channel in self.subvolumes[0].channels
        res = []
        if image == 'pseudorgb':
            for s in self.subvolumes:
                if s.session_name == session_name:
                    mean = s.get('mean_%s'%channel)
                    std = s.get('std_%s'%channel)
                    max = s.get('max_%s'%channel)
                    rgb = np.dstack([normalize(std),
                                     normalize(mean),
                                     normalize(max)])
                    res.append(rgb)
        else:
            for s in self.subvolumes:
                if s.session_name == session_name:
                    res.append(s.get(image))
        res = np.asarray(res)
        return np.mean(res, axis=0)



class NeurofinderData(CalciumDataAbstract):
    lab_names = ['hausser', 'losonczy', 'svoboda_peres', 'svoboda_sofroniew']
    _default_neurofinder_dir = os.getenv("HOME") + '/thesis/code/data/neurofinder/'

    def __init__(self, neurofinder_dir=None, lab_name='hausser'):
        assert lab_name in self.lab_names, ValueError('not a valid lab name')

        # data paths
        if neurofinder_dir is None:
            self.neurofinder_dir = self._default_neurofinder_dir
        self.h5_subvolume_dir = self.neurofinder_dir + 'neurofinder_%s/H5/' % lab_name
        self.h5_subvolume_file_names = os.listdir(self.h5_subvolume_dir)
        self.h5_subvolume_file_paths = [self.h5_subvolume_dir + h5file for h5file in self.h5_subvolume_file_names]
        self.subvolumes = [NeurofinderSubvolume(h5_file_path) for h5_file_path in self.h5_subvolume_file_paths]
