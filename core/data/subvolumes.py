"""subvolumes.py: Wrappers data structure for Calcium imaging subvolumes."""

__version__ = '0.8'

# built-in
import os
from copy import deepcopy

# 3rd party
import matplotlib.pyplot as plt
import numpy as np
import h5py as h5

from scipy.signal import fftconvolve

# own
from core.visuals import visualize_edges
from core.methods import create_2d_edge_graph, compute_ilastik_features_RGB, normalize_image_to_zero_one, \
    create_out_path, norm_cross_correlation



class SubvolumeAbstract(object):
    """ 'Abstract' class representing a subvolume, implements common functionality. """

    rois = []

    def __init__(self, h5_file_path, channels):
        self.h5_file_path = h5_file_path
        data = h5.File(h5_file_path, 'r')
        self.h5_fields = [data.values()[i].name.encode('utf_8').split('/')[1] for i in xrange(len(data.values()))]
        data.close()

        self.channels = channels
        self.dims = self.get('mean_green').shape
        self.name = self._set_name()

        self._visualized_features_cache = {}
        self._vis_xcorr_h = None
        self._vis_xcorr_v = None
        self._vis_corr_h = None
        self._vis_corr_v = None

    def _set_name(self):
        raise NotImplementedError('To be implemented by subclasses.')

    def _create_rois_for_subvolume(self):
        raise NotImplementedError('To be implemented by subclasses.')

    def is_nice(self):
        raise NotImplementedError('To be implemented by subclasses.')

    def get(self, h5_field):
        """ Access underlying HDF5 file and return value of given field.

        :param h5_field: valid field in underlying HDF5 file
        :return: value of h5_field
        """
        assert h5_field in self.h5_fields, KeyError('Not a valid dataset name in the underlying HDF5 file: %s'%h5_field)
        h5_file = h5.File(self.h5_file_path, 'r')
        value = np.asarray(h5_file[h5_field])
        h5_file.close()
        return value

    def h5_write(self, h5_field, value, dtype=None):
        """ Write a value to the given field in the subvolume's HDF5 file.

        :param h5_field: New field name.
        :param value: Value of new field.
        :return: void
        """
        assert h5_field not in self.h5_fields, KeyError('[Computer says NO] field already exists in %s.' % self.name)
        if dtype is None:
            dtype = value.dtype
        with h5.File(self.h5_file_path, 'a') as h5_file:
            h5_file.create_dataset(name=h5_field, data=value, dtype=dtype)
        self.h5_fields.append(h5_field)

    def h5_remove_field(self, h5_field):
        assert h5_field in self.h5_fields, KeyError('[Computer says NO] cannot delete field, does not exist %s.' % self.name)
        with h5.File(self.h5_file_path) as h:
            del h[h5_field]
        self.h5_fields.remove(h5_field)

    def h5_show_image(self, h5_field):
        """ Show an image from the subvolume's HDF5 file as given by h5_field.

        :param h5_field: the image to show, must be of shape (512,512)
        :return: figures and axes with the requested image
        """
        assert h5_field in self.h5_fields, KeyError('H5 field %s not in list of h5_fields.' % h5_field)
        img = self.get(h5_field)
        assert img.shape == self.dims, ValueError('The returned value is not an image or contains more than one.')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.imshow(img)
        return fig, ax

    def get_gt_from_active_rois_by_thresh(self, spike_thresh=0.2, fmax_thresh=160):
        """ Create a ground truth where only active neurons are included based on a threshold, that is, 'active' neurons
        have spikes above the specified threshold (in all three quantiles).

        :param thresh: Spike train threshold above which neurons are considered 'active'.
        :return: An active ground truth for this subvolume.
        """
        subvol_gt_active = np.zeros(self.dims[0] * self.dims[1], dtype=np.uint8)
        for roi in self.rois:
            if roi.is_active_by_thresh(spike_thresh=spike_thresh, fmax_thresh=fmax_thresh):
                subvol_gt_active[roi.indices] = 1
        return subvol_gt_active

    def get_roi_by_id(self, roi_id):
        """ Return ROI with given Janelia session ID.

        :param roi_id: Janelia session ID.
        :return: ROI with given Janelia session ID.
        """
        assert isinstance(roi_id, int)
        for r in self.rois:
            if r.id == roi_id:
                return r
        # lazy checking if an roi with specified id was found for this volume
        raise ValueError('ROI ID %d not part this subvolume (%s)' % (roi_id, self.name))

    def compute_edge_feature(self, method='xcorr', channel='green'):
        methods = {
            'corr' : np.correlate,
            'xcorr' : norm_cross_correlation,
            'fftconv' : fftconvolve
        }
        params = {
            'corr' : {'mode':'same'},
            'xcorr' : {},
            'fftconv' : {'mode':'valid'}
        }
        assert method in methods.keys()
        assert channel in self.channels

        f = methods[method]
        param = params[method]

        edge_graph = create_2d_edge_graph(self.dims[0], self.dims[1])
        edge_features = np.ndarray((edge_graph.shape[0], 1))
        vol_rshp = self.get('volume_%s' % channel).reshape((-1, self.dims[0]*self.dims[1]))

        for edge_idx in xrange(edge_graph.shape[0]): # every edge
            i,j = edge_graph[edge_idx] # get pixel indices for current edge
            pairwise_feature_ij = f(vol_rshp[:, i], vol_rshp[:, j], **param)
            edge_features[edge_idx] = pairwise_feature_ij.squeeze()
        return edge_features

    def ssvm_create_x(self, unary_feat_names, pairwise_feat_names, bias_u=False, bias_p=False, normalize_u=False, verbose=0):
        # An instance ``x_i`` is represented as a tuple ``(node_features, edges, edge_features)``
        edges = create_2d_edge_graph(self.dims[0], self.dims[1])
        unaries = self.ssvm_pack_unary_features(unary_feat_names, bias=bias_u, normalize=normalize_u, verbose=verbose)
        pairies = self.ssvm_pack_pairwise_features(pairwise_feat_names, bias=bias_p, verbose=verbose)
        return (unaries, edges, pairies)

    def ssvm_create_y(self, gt_name):
        # Labels ``y_i`` have to be an array of shape (n_nodes,)
        gt = self.get(gt_name).astype('uint8')
        return gt.reshape(self.dims[0] * self.dims[1])

    def ssvm_pack_unary_features(self, h5_feature_names, add_features=None, bias=False, verbose=0, normalize=True):
        unaries = []

        if bias:
            unaries.append(np.ones(self.dims[0] * self.dims[1]))
            if verbose > 0: print 'added bias'

        if not isinstance(h5_feature_names, list):
            h5_feature_names = [h5_feature_names]

        for feat_name in h5_feature_names:
            assert feat_name in self.h5_fields, KeyError('Unary features: H5 volume has no entry %s' % feat_name)
            feat = self.get(feat_name)
            if len(feat.shape) == 2:
                if feat.shape == self.dims: # like (512, 512)
                    feat = feat.reshape((self.dims[0] * self.dims[1]))
                    if normalize: feat = normalize_image_to_zero_one(feat)
                    unaries.append(feat)
                    if verbose > 0: print 'added feature', feat_name
                elif (feat.shape[0] == (self.dims[0] * self.dims[1])) and (feat.shape[1] >= 2): # like (262144, 2)
                    for feat_i in xrange(feat.shape[1]):
                        if normalize:
                            unaries.append(normalize_image_to_zero_one(feat[:, feat_i]))
                        else:
                            unaries.append(feat[:, feat_i])
                        if verbose > 0: print 'added feature', feat_name, 'with index', feat_i
            elif len(feat.shape) == 3: # like (512, 512, 147)
                if ((feat.shape[0], feat.shape[1]) == self.dims) and (feat.shape[2] >= 2):
                    feat = feat.reshape((self.dims[0] * self.dims[1], -1))
                    for feat_i in xrange(feat.shape[1]):
                        if normalize:
                            unaries.append(normalize_image_to_zero_one(feat[:, feat_i]))
                        else:
                            unaries.append(feat[:, feat_i])
                        if verbose > 0: print 'added feature', feat_name, 'with index', feat_i
            elif feat.shape == (self.dims[0] * self.dims[1]):
                unaries.append(feat)

        if add_features is not None:
            assert add_features.shape[0] == (self.dims[0] * self.dims[1]), ValueError('Additional features must be of shape (n_samples, n_features)')
            if len(add_features.shape) == 2:
                for feat_i in xrange(add_features.shape[1]):
                    if normalize:
                        unaries.append(normalize_image_to_zero_one(add_features[:, feat_i]))
                    else:
                        unaries.append(add_features[:, feat_i])
                    if verbose > 0: print 'added additional feature #', feat_i+1
            else:
                if normalize:
                    unaries.append(normalize_image_to_zero_one(add_features))
                else:
                    unaries.append(add_features)
                if verbose > 0: print 'added additional feature'
        return np.hstack([unaries]).transpose()

    def ssvm_pack_pairwise_features(self, h5_feature_names, bias=False, verbose=0):
        if not isinstance(h5_feature_names, list):
            h5_feature_names = [h5_feature_names]
        pairies = []
        x, y = self.dims
        n_edges = (x*(y-1)) + ((x-1)*y)
        if bias:
            pairies.append(np.ones([n_edges, 1]))
            if verbose > 0: print 'added bias'
        for feat_name in h5_feature_names:
            assert feat_name in self.h5_fields, KeyError('Pairwise features: H5 volume has no entry %s' % feat_name)
            feat = self.get(feat_name)
            assert feat.shape == (n_edges, 1), ValueError('Shape mismatch, does not seem to be an edge feature: %s' % feat_name)
            if feat_name == 'feats_xcorr_green': feat = 1 - feat # to retain submodularity
            pairies.append(feat)
            if verbose > 0: print 'added feature: %s' % feat_name
        return np.hstack(pairies)

    def visualize_edge_feature(self, h5_feature_name):
        """ Visualize pairwise features between neighbouring pixels as two images of
        shape (m, n-1) and (m-1, n) corresponding, respectively, to only the
        horizontal or vertical edges in a graph with m x n vertices.

        :param h5_feature_name: HDF5 field name of the edge feature, e.g. "feats_xcorr_green".
        :return: Images showing the pairwise features between pixels in horizontal and vertical direction resp.
        """
        assert h5_feature_name in self.h5_fields
        if not h5_feature_name in self._visualized_features_cache:
            edges = create_2d_edge_graph(self.dims[0], self.dims[1])
            self._visualized_features_cache[h5_feature_name] = \
            visualize_edges(edges, self.get(h5_feature_name), 0, self.dims)
        return self._visualized_features_cache[h5_feature_name]

    #deprecated
    def visualize_xcorr(self, channel='green'):
        """
        Visualize normalized cross-correlation (xcorr) between pixels.

        :param channel: Visualize computed xcorr for red or green channel.
        :return: Images showing the normalized cross-correlation between pixels in horizontal and vertical direction resp.
        """
        assert channel in self.channels
        if (self._vis_xcorr_h is None) or (self._vis_xcorr_v is None):
            edges = create_2d_edge_graph(self.dims[0], self.dims[1])
            xcorr = self.get('feats_xcorr_%s' % channel)
            self._vis_xcorr_h, self._vis_xcorr_v = visualize_edges(edges, xcorr, 0, self.dims)
        return self._vis_xcorr_h, self._vis_xcorr_v

    #deprecated
    def visualize_corr(self, channel='green'):
        """
        Visualize correlation (corr) between pixels.

        :param channel: Visualize computed corr for red or green channel.
        :return: Images showing correlation between pixels in horizontal and vertical direction resp.
        """
        assert channel in self.channels
        if (self._vis_corr_h is None) or (self._vis_corr_v is None):
            edges = create_2d_edge_graph(self.dims[0], self.dims[1])
            corr = self.get('feats_corr_%s' % channel)
            self._vis_corr_h, self._vis_corr_v = visualize_edges(edges, corr, 0, self.dims)
        return self._vis_corr_h, self._vis_corr_v

    def visualize_pseudo_rgb(self, channel='green'):
        """
        Visualize average mean, maximum value and standard deviation of the current volume as pseudo RGB (values are
        normalized from 0 to 1), that is, put each of those three projections in a different RGB channel.
        :return: pseudo RGB image
        """
        assert channel in self.channels
        mean = self.get('mean_green')
        std = self.get('std_green')
        max = self.get('max_green')
        rgb = np.dstack([normalize_image_to_zero_one(std),
                         normalize_image_to_zero_one(mean),
                         normalize_image_to_zero_one(max)])
        return rgb

    def save_subvolume_stuff_as_png(self, out_dir, cmap=None, dpi=None):
        pngs = dict()
        out_dir = os.path.join(out_dir, self.name)
        try:
            create_out_path(out_dir, except_on_exist=True)
            create_out_path(os.path.join(out_dir, 'ilastik_feats'), except_on_exist=True)
        except IOError:
            print 'Out dir exists, aborting.'
            return 1

        print 'Reading contents from HDF5'
        for field in self.h5_fields:
            pngs[field] = self.get(field)
        print '--> OK'

        print 'Saving HDF5 contents as PNG to', out_dir
        # save everything that is just a png and check for reshaped ones
        for name, img in pngs.iteritems():
            png_out = os.path.join(out_dir, name)
            if img.shape == self.dims:
                plt.imsave(png_out + '.png', img, cmap=cmap, dpi=dpi)
            if img.shape[0] == (self.dims[0] * self.dims[1]):
                for dim in xrange(img.shape[1]):
                    plt.imsave(png_out + '_%d.png'%dim, img[:, dim].reshape(self.dims[0], self.dims[1]), cmap=cmap, dpi=dpi)

        # for every channel:
        #   a: save pairwise potentials as images in vertical and horizontal direction
        #   b: save pseudo RGB
        #   c: save ilastik features
        for channel in ['green', 'red']:
            xcorr_h, xcorr_v = self.visualize_xcorr(channel=channel)
            plt.imsave(os.path.join(out_dir, 'xcorrh_%s.png'%channel), xcorr_h, cmap=cmap, dpi=dpi)
            plt.imsave(os.path.join(out_dir, 'xcorrv_%s.png'%channel), xcorr_v, cmap=cmap, dpi=dpi)

            corr_h, corr_v = self.visualize_corr(channel=channel)
            plt.imsave(os.path.join(out_dir, 'corrh_%s.png'%channel), corr_h, cmap=cmap, dpi=dpi)
            plt.imsave(os.path.join(out_dir, 'corrv_%s.png'%channel), corr_v, cmap=cmap, dpi=dpi)

            pseudo_rgb = self.visualize_pseudo_rgb(channel=channel)
            plt.imsave(os.path.join(out_dir, 'pseudoRGB_%s.png'%channel), pseudo_rgb, cmap=cmap, dpi=dpi)

            ilastik_feats = self.get('feats_ilastik_pseudoRGB_%s'%channel)
            for feat in xrange(ilastik_feats.shape[2]):
                plt.imsave(os.path.join(out_dir, 'ilastik_feats', 'feat_%s_%d.png'%(channel, feat)), ilastik_feats[:, :, feat], cmap=cmap, dpi=dpi)
        print '--> OK'

    def save_ilastik_features_as_pseudoRGB(self, out_dir, channel='green'):
        print 'Saving pseudo RGB image for subvolume', self.name

        # these are the features that are computed using VIGRA
        feature_names = {
            'gaussian_smoothing': [0, 1, 9, 17, 25, 33, 41],
            'laplacian_of_gaussian': [2, 10, 18, 26, 34, 42],
            'gaussian_gradient_magnitude': [3, 11, 19, 27, 35, 43],
            'difference_of_gaussians': [4, 12, 20, 28, 36, 44],
            'structure_tensor_eigenvalues': [5, 6, 13, 14, 21, 22, 29, 30, 37, 38, 45, 46],
            'hessian_of_gaussian_eigenvalues': [7, 8, 15, 16, 23, 24, 31, 32, 39, 40, 47, 48]
        }

        rgb = self.get('pseudo_rgb_%s' % channel)
        f = compute_ilastik_features_RGB(rgb)
        n_feats = f.shape[2]/3
        feats_rgb = []
        # build an RGB image from a specific feature setting applied to mean, max and std
        for i in xrange(n_feats):
            r, g, b = f[:,:,i], f[:,:,i+n_feats], f[:,:,i+n_feats+n_feats]
            f_rgb = np.dstack([ normalize_image_to_zero_one(r),
                                normalize_image_to_zero_one(g),
                                normalize_image_to_zero_one(b)])
            feats_rgb.append(f_rgb)
        # save the features
        for img_i in xrange(len(feats_rgb)):
            for k, v in feature_names.iteritems():
                if img_i in v:
                    feat_name = k
                    break
            plt.imsave('%s/%s_feats_%s_as_pseudoRGB_%d.png' % (out_dir, self.name, feat_name, img_i), feats_rgb[img_i])
        print '--> OK'

    def plot_rois(self, background='mean_green', cmap=None, roi_color='m'):
        """
        Plot ROIs of this subvolume.

        :param background: Specify image to plot ROIs into. Must be in subvolume.values and of dimension (512,512).
        :return: figure and axis
        """
        assert background in self.h5_fields, KeyError('background %s not part of h5_fields.' % background)
        background = self.get(background)
        assert background.shape == self.dims
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(background, cmap=cmap)
        for roi in self.rois:
            ax.add_patch(plt.Polygon(roi.polygon, color=roi_color, fill=False))
        return fig, ax

    def plot_active_rois_by_thresh(self, thresh, background='mean_green'):
        """
        Plot ROIs of this subvolume.

        :param background: Specify image to plot ROIs into. Must be in subvolume.values and of dimension (512,512).
        :return: figure and axis
        """
        assert background in self.h5_fields, KeyError('[Computer says NO] background %s not part of h5_fields.' % background)
        background = self.get(background)
        assert background.shape == self.dims

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(background)
        for roi in self.rois:
            if roi.is_active_by_thresh(thresh):
                ax.add_patch(plt.Polygon(roi.polygon, color='g', fill=False))
            else:
                ax.add_patch(plt.Polygon(roi.polygon, color='r', fill=False))
        return fig, ax

    def mpl_show_pixel_intensity_hist(self, channel='green'):
        print 'histogram...'
        plt.figure('Histogram of pixel intensities (%s)'%self.name)
        data = self.get('volume_%s'%channel)
        bins=np.bincount(np.ravel(data.astype(np.int64)))
        plt.bar(range(len(bins)), bins)
        plt.show()

    def mpl_show_pixel_intensity_scatter_plot(self, channel='green'):
        print 'scatter plot...'
        plt.figure('Scatter plot of pixel intensities (volume)')
        data = self.get('volume_%s'%channel)
        plt.scatter(data[:,-1,:].ravel(),data[:,:,-1].ravel())
        plt.show()

    def myv_show_contour3d(self, slice_step=3, pixel_step=2, channel='green'):
        from mayavi import mlab
        print 'Calculating 3d contour using Mayavi...'
        mlab.contour3d(self.get('volume_%s'%channel)[::slice_step, ::pixel_step, ::pixel_step])
        mlab.show()




class JaneliaSubvolume(SubvolumeAbstract):
    """Class representing a subvolume."""
    channels = ['green', 'red']
    activity_labels = ['active_very_certain', 'active_mod_certain', 'uncertain', 'inactive_auto', 'inactive_very_certain']
    activity_labels_short = ['++', '+', '-', 'x', 'xx']
    mpl_activity_label_colors = {
        'active_very_certain' : '#F12416',
        'active_mod_certain' : '#F17116',
        'uncertain' : '#FFEE00',
        'inactive_auto' : '#0DB0F5',
        'inactive_very_certain' : '#0D35F5'
    }

    def __init__(self, h5_file_path, session_ROI_dict, roi_signals_dir):
        SubvolumeAbstract.__init__(self, h5_file_path, self.channels)
        print h5_file_path
        self.roi_signals_dir = roi_signals_dir
        self.rois = self._create_rois_for_subvolume(session_rois=session_ROI_dict[self.session_name].values())
        self.h5_roi_signals_path = self.roi_signals_dir + self.name + '_rois.h5'

    #override
    def _set_name(self):
        self.session_name = self.h5_file_path.split('_fov')[0].split('/')[-1]
        self.vvppp = self.h5_file_path.split('_fov_')[1].split('.h5')[0]
        return self.session_name + '_' + self.vvppp

    #override
    def is_nice(self):
        """
        Simply checks if a subvolume contains at least one very active neuron.

        :return: True if subvolume contains at least one very active neuron, False otherwise.
        """
        stats, _ = self.get_roi_activity_label_counts()
        if stats[0] == 0:   # no very active neuron at all, no nice volume
            return False
        else:
            return True

    #override
    # noinspection PyMethodOverriding
    def _create_rois_for_subvolume(self, session_rois):
        """
        Given the values from a subvolume key in a Janelia session file, extract ROI information for the current
        subvolume and spawn ROI objects.

        :param session_rois: ROIs for the current subvolume (extracted from the Janelia session file)
        :return: List of ROI objects.
        """
        subvol_gt = self.get('gt').reshape(self.dims[0] * self.dims[1])
        subvol_rois = []
        for r in session_rois:
            # if a roi matches the ground truth completely, it will return array of 1's only
            tmp = np.unique(subvol_gt[r.indices])
            if (len(tmp) == 1) and (tmp[0] == 1):
                r = deepcopy(r)
                r.subvolume = self
                r.meta_info = {'session_name': self.session_name, 'vvppp': self.vvppp}
                subvol_rois.append(r)
        return subvol_rois

    def init_roi_signals(self):
        [r.load_precomputed_signals() for r in self.rois]

    def mpl_get_polygon_patches(self, labels=None, linewidth=None, color=None, fill=False):
        """ Get MPL polygon patches for all ROI based on the given activity label(s).
        :param labels: Only include polygons for the specified activity labels.
        :return: MPL polygons to be used as patches.
        """
        polygons = []
        if labels is None: # get patches for all ROI
            for roi in self.rois:
                if color is None:
                    roi_color = roi.mpl_color
                else:
                    roi_color = color
                polygons.append(plt.Polygon(roi.polygon, color=roi_color, linewidth=linewidth, fill=fill))
        else: # get patches only for specified labels
            if isinstance(labels, list):
                for label in labels:
                    assert label in self.activity_labels, KeyError('%s is not a valid label name.' % label)
            elif isinstance(labels, str):
                assert labels in self.activity_labels, KeyError('%s is not a valid label name.' % labels)
                labels = [labels]
            for roi in self.rois:
                roi_label = roi.activity_label
                if roi_label in labels:
                    if color is None:
                        roi_color = roi.mpl_color
                    else:
                        roi_color = color
                    polygons.append(plt.Polygon(roi.polygon, color=roi_color, linewidth=linewidth, fill=fill))
        return polygons

    def get_pixel_indices_for_background(self):
        gt = self.get('gt').reshape(self.dims[0]*self.dims[1])
        return np.where(gt==0)[0]

    def get_pixel_indices_by_activity_labels(self, activity_labels = ['inactive_very_certain', 'inactive_auto']):
        if isinstance(activity_labels, list):
            for label in activity_labels:
                assert (label in self.activity_labels) or (label in self.activity_labels_short) or (label in ['bg']), \
                    ValueError('%s is not a valid label name.' % label)
        elif isinstance(activity_labels, str):
            assert (activity_labels in self.activity_labels) or (activity_labels in self.activity_labels_short) \
                   or (activity_labels is 'bg'), ValueError('%s is not a valid label name.' % activity_labels)
            activity_labels = [activity_labels]
        pixel_indices = []
        for roi in self.rois:
            for label in activity_labels:
                if label == 'bg':
                    continue    # skip background label for now, add all at once below
                elif roi.compare_activity_label(label):
                    for pixel in roi.indices:
                        pixel_indices.append(pixel)
                    break # label found, no need to check the rest (if any)
        # also add background pixels if specified in negative labels
        if 'bg' in activity_labels:
            for pixel in self.get_pixel_indices_for_background():
                pixel_indices.append(pixel)
        return np.asarray(pixel_indices, dtype=np.uint32)

    def get_activeGT_by_labels(self, labels):
        """ Create a ground truth containing only those ROI that have the specified activity label(s).

        :param labels: Valid activity label according to subvolume.activity_labels.
        :return: Active ground truth containing only neurons with the specified label(s).
        """
        if isinstance(labels, list):
            for label in labels: assert label in self.activity_labels, KeyError(
                '[Error] %s is not a valid label name.' % label)
        elif isinstance(labels, str):
            assert labels in self.activity_labels, KeyError('%s is not a valid label name.' % labels)
            labels = [labels]

        subvol_gt_active = np.zeros(self.dims[0] * self.dims[1], dtype=np.uint8)
        for roi in self.rois:
            label = roi.activity_label
            if label in labels:
                subvol_gt_active[roi.indices] = 1
        return subvol_gt_active

    def get_roi_activity_label_counts(self):
        """ Return distribution of activity labels for subvolume.
        :return: ROI activity label counts.
        """
        cnt_active_very = 0
        cnt_active_mod = 0
        cnt_unsure = 0
        cnt_inactive = 0
        cnt_inactive_auto = 0
        cnt_all = len(self.rois)
        # collect activity stats
        for r in self.rois:
            label = r.activity_label
            if label == 'active_very_certain':
                cnt_active_very += 1
            elif label == 'active_mod_certain':
                cnt_active_mod += 1
            elif label == 'uncertain':
                cnt_unsure += 1
            elif label == 'inactive_very_certain':
                cnt_inactive += 1
            else:  # label == 'inactive_auto'
                cnt_inactive_auto += 1
        return np.asarray([cnt_active_very, cnt_active_mod, cnt_unsure, cnt_inactive_auto, cnt_inactive]), cnt_all

    def plot_activity_labels(self, image=None, labels=None, ax=None, cmap=None, color=None, linewidth=.75):
        """ Plot ROI with given activity label(s) for this subvolume into the given image.

        :param labels: Specify activity label(s) to be plotted. If None, all labels are used.
        :param image: Specify image to plot ROIs into. Must be of dimension (512,512). If None, mean image is used.
        :return: axis showing an image with ROI plotted in
        """
        if labels is not None:
            if isinstance(labels, list):
                for label in labels:
                    assert label in self.activity_labels, KeyError('%s is not a valid label name.' % label)
            elif isinstance(labels, str):
                assert labels in self.activity_labels, KeyError('%s is not a valid label name.' % labels)
                labels = [labels]
        else:
            labels = self.activity_labels
        if image is None:
            image = self.get('mean_green')
        else:
            assert image.shape[0] == self.dims[0]
            assert image.shape[1] == self.dims[1]
        if ax is None:
            ax = plt.gca()
        ax.imshow(image, cmap=cmap)
        roi_polygon_patches = self.mpl_get_polygon_patches(labels=labels, color=color, linewidth=linewidth)
        for roi_patch in roi_polygon_patches:
            ax.add_patch(roi_patch)
        return ax



class NeurofinderSubvolume(SubvolumeAbstract):

    channels = ['green']

    def __init__(self, h5_file_path):
        SubvolumeAbstract.__init__(self, h5_file_path=h5_file_path, channels=self.channels)
    #override

    def _set_name(self):
        return self.h5_file_path.split('/')[-1].split('.h5')[0]
    #override

    def _create_rois_for_subvolume(self):
        raise NotImplementedError()
