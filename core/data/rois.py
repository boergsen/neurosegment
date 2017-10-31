"""Roi.py: Wrapper for handling and visualizing Neurons."""

__version__ = '0.8'

# built-in
import os

# 3rd party
import matplotlib.pyplot as plt
import numpy as np
import h5py as h5

# own
from core.methods import median_absolute_deviation as mad
from core.methods import create_out_path, get_optimal_number_of_bins


class JaneliaRoi(object):
    """A class representing a neuron as a Region-Of-Interest (ROI)."""

    # constants
    _mad_factor = 4             # 3, as proposed in Huber2012, thresholds not enough
    _pixel_size = 0.88          # 1 pixel corresponds to 0.88 um (Huber2012)
    _sample_rate_hz = 4         # huber says 4Hz, crncs and neurofinder docs say 7Hz ?!?
    _sliding_window_width = 60

    def __init__(self, indices, xy, janelia_id, subvolume, meta_info):
        """Constructor for the Roi class.
        :param indices: Pixel indices determining this ROI in the reshaped volume dimensions (512*512)
        :param xy: x and y coordinates determining a Polygon around the current ROI
        :param janelia_id: Janelia ROI ID (valid across a session file)
        :param subvolume: Reference to the subvolume this ROI belongs to.
        :param meta_info: Contains subvolume info, ROI info.
        :return: void
        """
        self.indices = indices
        self.xy = xy
        self.polygon = np.asarray(zip(self.xy[1], self.xy[0]))
        self.id = janelia_id
        self.centroid = self._compute_cell_centroid()
        self.subvolume = subvolume
        self.meta_info = meta_info
        self.init_empty_signals()

    def init_empty_signals(self):
        # Activity properties
        self._spike_train = None  # !
        self._is_active = None
        self._activity_label = None
        # Signals from Calcium imaging and when deriving the spike train
        self._f_raw = None
        self._f_neuropil = None
        self._f_corrected = None  # !
        self._f0_estimates = None
        self._delta_ff_estimates = None
        self._preliminary_events = None
        self._f0_interpolates = None  # !
        self._delta_ff = None  # !
        # Non-negative deconvolution results (from pyfnnd)
        self._conv_n_best = None
        self._conv_c_best = None
        self._conv_theta_best = None
        self._conv_LL = None

    @property
    def mpl_color(self):
        return self.subvolume.mpl_activity_label_colors[self.activity_label]

    @property
    def dims(self):
        return self.subvolume.dims

    @property
    def time_scale(self):
        return np.arange(self.f_raw.shape[0]) * 1.0 / self._sample_rate_hz

    @property
    def name(self):
        return '%s_JID%s(%s)' % (self.subvolume.name, str(self.id), self.activity_label)

    @staticmethod
    def _compute_delta_ff(f, f0):
        """Correct a signal $F$ by the amount of baseline fluorescence $F_0$ that contributes to it.
        :param f: Raw signal.
        :param f0: Baseline fluorescence of raw signal.
        :return: $\Delta F/F$.
        """
        return (f - f0) / float(f0)

    def _compute_cell_centroid(self):
        """Compute the centroid of the ROI.
        :return: centroid coordinates (x, y)
        """
        polygon_points = zip(self.xy[1], self.xy[0])
        x = [p[0] for p in polygon_points]
        y = [p[1] for p in polygon_points]
        cx, cy = np.sum(x) / len(polygon_points), np.sum(y) / len(polygon_points)
        return cx, cy

    def _compute_temporal_signal(self, mask, channel='green'):
        """Average the pixels within a given mask (typically a ROI) in the current subvolume over time.
        :param mask: List with pixel indices the signal is extracted for.
        :param channel: Set green or red channel.
        :return: Temporal signal within the given mask (averaged pixel intensities over time).
        """
        assert channel in ['green', 'red']
        subvolume = self.subvolume.get('volume_%s' % channel).reshape((-1, self.dims[0] * self.dims[1]))
        temporal_signal = []
        for slice in subvolume:
            mean_roi_intensity = np.mean(slice[mask])
            temporal_signal.append(mean_roi_intensity)
        return np.asarray(temporal_signal)

    def _compute_neuropil_signal(self, corona_radius_mum, pixel_size_mum):
        """Extract the neuropil signal that surrounds each cell by averaging the intensities of all pixels within
        a 20 micrometer circular region from the cell center (excluding ROI pixels beforehand, so only a corona
        remains) (Akerboom2012, Kerlin2010).

        :param int corona_radius_mum: radius of the corona from ROI center
        :param float pixel_size_mum: actual size of a pixel in micrometer
        :return: the raw signal of the ROI corrected by the signal of the neuropil corona
        """
        if self._f_neuropil is None:
            # centroid coordinates
            cx, cy = self.centroid
            # get all pixels in 20 micrometer radius around center (20 * .88 = 17.6 pixels ~ 18)
            radius = np.ceil(corona_radius_mum * pixel_size_mum)
            # create neuropil mask (relative to the ogrid!)
            y, x = np.ogrid[-radius: radius, -radius: radius]
            mask_np = x ** 2 + y ** 2 <= radius ** 2
            # crate roi mask
            mask_roi = np.zeros(self.dims[0] * self.dims[1], dtype=np.bool)
            mask_roi[self.indices] = True
            # create corona mask by removing roi mask from neuropil mask
            mask_corona = np.zeros(self.dims, dtype=np.bool)
            # have to check that in case corona goes over image borders
            mx, my = mask_corona[cy - radius:cy + radius, cx - radius:cx + radius].shape
            mask_corona[cy-radius : cy+radius, cx-radius : cx+radius][mask_np[:mx, :my]] = True
            mask_corona = mask_corona.reshape(self.dims[0] * self.dims[1]) - mask_roi
            # average the intensities in the corona over time to get the neuropil signal
            self._f_neuropil = self._compute_temporal_signal(mask_corona)
        return self._f_neuropil

    def _deconvolve_spike_train_for_quantile(self, quantile, spikes_tol=1E-10, params_tol=1E-6, verbosity=0, interpolate_f0=True):
        """Compute the spike train on $\Delta F/F$ with the neuropil corrected raw signal for $F$ and and different estimates
        for $F_0$. Three $F_0$ estimates are derived by a 60 sec sliding window over the neuropil corrected raw signal
        where each window contributes the 50th, 20th and 5th percentile to the current time step of the respective estimate
        (corresponding to quantile in [0,1,2]). If 'interpolated' is given the $F_0$ estimate is further refined by
        interpolation during events (events are estimated using different thresholds on the median average deviation of the
        neuropil corrected raw signal).

        :param quantile: int in [0,1,2] for the 50th, 20th or 5th percentile to be used as base fluo estimate in $\Delta F/F$ computation.
        :param spikes_tol: Spikes tolerance used for non-negative deconvolution in pyfnnd.
        :param params_tol: Parameter tolerance used for non-negative deconvolution in pyfnnd.
        :param verbosity: Verbosity level of pyfnnd.
        :param interpolate_f0: Use interpolated or unchanged baseline fluorescence $F_0$ estimates for calculating $\Delta F/F$.
        :return: Spike train computed on the $\Delta F/F$ for the given quantile.
        """
        from pyfnnd import deconvolve

        if interpolate_f0:
            delta_f = self.delta_ff
        else:
            delta_f = self.delta_ff_estimates
        dt = 0.02
        # use 'fast non-negative deconvolution' to extract the spike train from the fluorescence estimates
        n_best, c_best, LL, theta_best = deconvolve(delta_f[:, quantile], dt=dt,
                                                    verbosity=verbosity,
                                                    learn_theta=(0, 1, 1, 1, 0), spikes_tol=spikes_tol,
                                                    params_tol=params_tol)
        self._conv_n_best = n_best
        self._conv_c_best = c_best
        self._conv_LL = LL
        self._conv_theta_best = theta_best
        return n_best

    def is_active_by_thresh(self, spike_thresh=.2, fmax_thresh=None, return_stats=None, verbose=0):
        """ Mark neuron as active if max value in spike trains in all quantiles is above given threshold.
        :param spike_thresh: Threshold on spike train activity above which a neuron is considered active or not.
        :return: True if active, False otherwise.
        """
        n_best50 = self.spike_train[:, 0]
        n_best20 = self.spike_train[:, 1]
        n_best5 = self.spike_train[:, 2]
        max50 = np.max(n_best50)
        max20 = np.max(n_best20)
        max5 = np.max(n_best5)

        if fmax_thresh is None:
            # neuron considered active if any of the two quantiles is above thresh
            if (int(max50 > spike_thresh) + int(max20 > spike_thresh) + int(max5 > spike_thresh)) > 1:
                self._is_active = True
            else:
                self._is_active = False
        else:
            f_true = self.f_corrected
            fmax = np.max(f_true)
            # neuron considered active if any of the two quantiles is above thresh and raw fluorescence is above given fmax
            if (int(max50 > spike_thresh) + int(max20 > spike_thresh) + int(max5 > spike_thresh)) > 1 and fmax > fmax_thresh:
                self._is_active = True
            else:
                self._is_active = False

        if verbose > 0:
            print ''
            print (max50 > spike_thresh), (max20 > spike_thresh), (max5 > spike_thresh)
            print max50, max20, max5
            print self.activity_label
            print "ON" if self._is_active else "OFF"
        if verbose > 2:
            self.plot_summary()
            plt.show()
        if return_stats is not None:
            f_true = self.f_corrected
            fmax = np.max(f_true)
            threshs = return_stats  # jep, one should not do that
            res = []
            for t in threshs:
                # if any of the two quantiles is above thresh and and raw fluorescence is above 160
                if (int(max50 > t) + int(max20 > t) + int(max5 > t)) > 1 and fmax > 160:
                    is_active = True
                else:
                    is_active = False
                tp = 0
                fp = 0
                tn = 0
                fn = 0
                unsure = 0
                if (self.activity_label in ['active_very_certain', 'active_mod_certain']) and is_active:
                    if verbose > 1: print 'correct'
                    tp += 1
                elif (self.activity_label not in ['active_very_certain', 'active_mod_certain']) and is_active:
                    if verbose > 1: print "wrong"
                    fp += 1
                elif (self.activity_label in ['inactive_auto', 'inactive_very_certain']) and not is_active:
                    if verbose > 1: print 'correct'
                    tn += 1
                elif (self.activity_label not in ['inactive_auto', 'inactive_very_certain']) and not is_active:
                    if verbose > 1: print "wrong"
                    fn += 1
                else:
                    if verbose > 1: print "unsure"
                    unsure += 1
                res.append(np.asarray([tp, fp, tn, fn, unsure]))
            return res
        return self._is_active

    def load_precomputed_signals(self):
        """Load precomputed signals and properties of the current neuron from the signal HDF5 file for the current subvolume.
        Within the HDF5 file every ROI is a data group with ROI ID as key and the actual signals as members.
        :return: void
        """
        try:
            with h5.File(self.subvolume.h5_roi_signals_path, 'r') as h5_in:
                try:
                    roi = h5_in[str(self.id)]
                except:
                    raise LookupError('[Uups] ROI ID not present in HDF5 data set. This should not happen.')
                try:
                    self.centroid = np.asarray(roi['centroid'])
                    self._delta_ff = np.asarray(roi['delta_f'])
                    self._delta_ff_estimates = np.asarray(roi['delta_f_estimates'])
                    self._f0_estimates = np.asarray(roi['f0_estimated'])
                    self._f0_interpolates = np.asarray(roi['f0_interpolated'])
                    self._f_corrected = np.asarray(roi['f_corrected'])
                    self._f_neuropil = np.asarray(roi['f_neuropil'])
                    self._f_raw = np.asarray(roi['f_raw'])
                    self.indices = np.asarray(roi['indices'])
                    self.polygon = np.asarray(roi['polygon'])
                    self._preliminary_events = np.asarray(roi['prelim_event_series'])
                    self._spike_train = np.asarray(roi['spike_trains'])
                    self._activity_label = str(roi['activity_label'].value)
                except:
                    raise LookupError('[Uups] Somethings wrong with the key/values for ROI', self.id)
        except:
            raise IOError('[Error] Precomputed signals: cannot load ROI HDF5 file for subvolume', self.subvolume.name)

    def write_back_to_hdf5(self, h5_field, content):
        """ Create a new entry in the signal HDF5 file of the current subvolume. Raise error if field exists.
        :param h5_field: The new field of the entry.
        :param content: The new content.
        :return: void
        """
        h5_path = self.subvolume.h5_roi_signals_path
        try:
            with h5.File(h5_path, 'a') as h5_in:
                roi = h5_in[str(self.id)]
                try:
                    _ = roi[str(h5_field)]
                    print '[Warning] value %s already present. Aborting.' % str(h5_field)
                    return
                except:
                    print '[Out] writing %s to %s' % (str(h5_field), h5_path)
                    roi.create_dataset(name=str(h5_field), data=content)
        except:
            raise IOError('[Error] Writing back to ROI HDF5: Cannot load file for subvolume', self.subvolume.name)

    @property
    def activity_label(self):
        """ Return the activity label for the current ROI (read it from signals HDF5 on first access).
        :return: The neuron's activity label.
        """
        if self._activity_label is None:
            try:
                with h5.File(self.subvolume.h5_roi_signals_path, 'r') as h5_in:
                    h5_grp = h5_in[str(self.id)]
                    label = h5_grp['activity_label'].value
                    self._activity_label = str(label)
            except:
                raise LookupError('[Warning] activity label for ROI %d not set yet.' % self.id)
        return self._activity_label

    def compare_activity_label(self, label):
        """ Check if ROI activity label is the same as 'label'.
        :param label: Label to compare.
        :return: True if label matches, False otherwise.
        """
        try:
            assert label in self.subvolume.activity_labels, ValueError()
            if self.activity_label == label:
                return True
            else:
                return False
        except AssertionError:
            assert label in self.subvolume.activity_labels_short
            if self.subvolume.activity_labels_short[self.subvolume.activity_labels.index(self.activity_label)] == label:
                return True
            else:
                return False

    @property
    def f_raw(self):
        """ Extract raw pixel intensities over time for current ROI.
        :return: Raw pixel intensities over time for current ROI.
        """
        if self._f_raw is None:
            self._f_raw = self._compute_temporal_signal(self.indices)
        return self._f_raw

    @property
    def f_corrected(self, neuropil_weight=.7, corona_radius_mum=20):
        """ Extract the neuropil signal F_neuropil that surrounds each cell by averaging the signal of all pixels within
        a 20 micrometer circular region from the cell center (excluding all selected cells) (Akerboom2012, Kerlin2010).

        The true fluorescence signal of a cell body is then estimated as:
            F_cell_true(t) = F_cell_measured(t) - ( r * F_neuropil(t) )

        :param neuropil_weight: Amount of neuropil reduction.
        :param corona_radius_mum: Distance of outer corona to ROI centroid.
        :return: The neuropil corrected raw signal.
        """
        if self._f_corrected is None:
            # trigger initialization
            _ = self._compute_neuropil_signal(corona_radius_mum=corona_radius_mum, pixel_size_mum=self._pixel_size)
            self._f_corrected = self.f_raw - (neuropil_weight * self._f_neuropil)
        return self._f_corrected

    @property
    def f0_estimates(self):
        """ Use a sliding window of 60 seconds width on the raw signal to get a $F_0$ estimate as the
        averaged 50th, 20th and 5th percentile of the ROI intensity distribution.

        :return: $F_0$ estimates for the 50th, 20th and 5th percentile taken from a sliding window over the raw signal.
        """
        if self._f0_estimates is None:
            window_size = self._sample_rate_hz * self._sliding_window_width  # 60-sec-wide sliding window
            window_half = window_size / 2
            cell_signal = self.f_corrected
            f0_estimate = []
            for i in xrange(0, cell_signal.shape[0]):
                if (i - window_half) < 0:  # window starts moving in from the left, only use its right side
                    window_signal = cell_signal[0: (i + window_half)]
                elif (i + window_half) > cell_signal.shape[0]:  # window starts leaving to the right, only use left side
                    window_signal = cell_signal[(i - window_half): cell_signal.shape[0]]
                else:  # use full window
                    window_signal = cell_signal[(i - window_half): (i + window_half)]
                f0_estimate.append(np.percentile(window_signal, [50, 20, 5]))
            f0_estimate = np.vstack(f0_estimate)
            self._f0_estimates = f0_estimate
        return self._f0_estimates

    @property
    def delta_ff_estimates(self):
        """ Calculate a preliminary $\Delta F/F$ estimate using the neuropil corrected raw signal and a preliminary $F_0$
        for every quantile.

        :return: Preliminary $\Delta F/F$ estimates for every quantile.
        """
        if self._delta_ff_estimates is None:
            delta_ff_estimates = [[self._compute_delta_ff(self.f_corrected[i], self.f0_estimates[i, q])
                                  for i in xrange(self.f0_estimates.shape[0])] for q in
                                 xrange(self.f0_estimates.shape[1])]
            self._delta_ff_estimates = np.asarray(delta_ff_estimates).transpose()
        return self._delta_ff_estimates

    @property
    def preliminary_events(self, start_peak_sec=2, end_peak_sec=5):
        """ Preliminary event series are needed for interpolating the final $F_0$. To this end the deltaF estimates
        are MAD thresholded and peaks are extracted for periods of activity (a period starting 2 seconds before and
        ending 5 seconds after the peak).

        :param start_peak_sec: Starting point of a period with detected peak.
        :param end_peak_sec: End of period for a detected peak.
        :return: A series of preliminary events for every quantile.
        """
        if self._preliminary_events is None:
            start_peak = start_peak_sec * self._sample_rate_hz  # sec*frame_rate
            end_peak = end_peak_sec * self._sample_rate_hz  # sec*frame_rate
            events = []
            signal_frames = self.delta_ff_estimates.shape[0]

            # extract MAD thresholds on the delta_f estimates for f0 for every quantile
            mad_threshs = np.asarray([self._mad_factor * mad(self.delta_ff_estimates[:, i])
                                      for i in xrange(self.delta_ff_estimates.shape[1])])

            # extract preliminary event series
            for q in xrange(self.delta_ff_estimates.shape[1]):  # for every quantile
                events_tmp = np.zeros_like(self.delta_ff_estimates[:, 0])
                for f in xrange(start_peak, signal_frames - end_peak):
                    if self.delta_ff_estimates[f, q] > mad_threshs[q]:  # delta(f)/f is above thresh --> possible peak
                        events_tmp[f - start_peak:f + end_peak] = self.delta_ff_estimates[f - start_peak:f + end_peak, q]
                events.append(events_tmp)
            self._preliminary_events = np.asarray(events).transpose()  # other signals are the other way around
        return self._preliminary_events

    @property
    def f0_interpolates(self):
        """ Interpolate $F_0$ for periods during events.
        :return: $F_0$, interpolated for periods during events.
        """

        def interpolate_x(xi, x0, x1, y0, y1):
            """ Interpolate the value $y_i$ for a point $x_i$ on a line between two points $x_0$ and $x_1$.

            :param xi: Point on a line to interpolate $y_i$ for.
            :param x0: x coordinate for point $x_0$
            :param x1: x coordinate for point $x_1$
            :param y0: y coordinate for point $x_0$
            :param y1: y coordinate for point $x_1$
            :return: Interpolated $y_i$ for given $x_i$.
            """
            return y0 + ((y1 - y0) * (xi - x0) / float((x1 - x0)))

        def start_end_indxs(signal_with_gaps):
            """ Go through a signal and get start and end indices of gaps, that is, periods of zeros only.

            :param signal_with_gaps: Signal that contains gaps.
            :return: Start and end indices of gaps within the signal.
            """
            indxs = np.where(signal_with_gaps == 0)[0]
            start_end_indxs = []
            start = indxs[0]
            for i in xrange(len(indxs)):
                current = indxs[i]
                if (i + 1) < len(indxs):
                    next = indxs[i + 1]
                    if (next - current) == 1:
                        continue
                    else:
                        end = current
                        start_end_indxs.append((start, end))
                        start = next
                else:
                    end = current
                    start_end_indxs.append((start, end))
            return start_end_indxs

        if self._f0_interpolates is None:
            frame_length = self.f0_estimates.shape[0]
            quantiles = self.f0_estimates.shape[1]
            f0_interpolates = []

            for q in xrange(quantiles):  # for every quantile
                f0_interpolate = np.zeros(frame_length)
                quantile_events = self.preliminary_events[:, q]

                # go over the signal once
                for i in xrange(frame_length):
                    # if no preliminary event was detected, use baseline F0 as is
                    if quantile_events[i] == 0:  # FIXME? what if 0 occurs during an event?
                        f0_interpolate[i] = self.f0_estimates[i, q]

                if len(np.where(f0_interpolate == 0)[0]) is not 0:  # signal has gaps
                    # go over the gaps in the signal and interpolate
                    gaps = start_end_indxs(f0_interpolate)
                    for gap in gaps:
                        gap_range = np.arange(gap[0], gap[1] + 1)  # as always, range is an index short it is ;-)
                        # FIXME?: +1/-1 could be problematic if a gap is at the very end/beginning of the signal
                        x0 = gap[0] - 1
                        x1 = gap[1] + 1
                        y0 = f0_interpolate[x0]
                        y1 = f0_interpolate[x1]
                        for xi in gap_range:
                            f0_interpolate[xi] = interpolate_x(xi, x0, x1, y0, y1)
                f0_interpolates.append(f0_interpolate)
            self._f0_interpolates = np.vstack(f0_interpolates).transpose()
        return self._f0_interpolates

    @property
    def delta_ff(self, bias=200):
        """ Compute the final $\Delta F/F$ using the neuropil corrected raw signal for $F$ and an $F_0$ that was
        interpolated during events.
        :param bias: Add a constant bias to the raw signal (does not change the result, but removes values below zero)
        :return: Fluorescence intensity signal for the given ROI.
        """
        if self._delta_ff is None:
            delta_ff = [[self._compute_delta_ff(self.f_corrected[i] + bias, self.f0_interpolates[i, q] + bias)
                        for i in xrange(self.f0_interpolates.shape[0])] for q in
                       xrange(self.f0_interpolates.shape[1])]
            self._delta_ff = np.asarray(delta_ff).transpose()
        return self._delta_ff

    @property
    def spike_train(self):
        """ Return the spike trains for the current neuron, one for each of the three quantiles used as baseline fluorescence
        in the $\Delta F/F$ computation (50th, 20th and 5th percentile correspond to index [0,1,2]).
        :return: Spike train for every quantile.
        """
        if self._spike_train is None:
            spike_trains = []
            for i in xrange(3):  # for every quantile
                n_best = self._deconvolve_spike_train_for_quantile(i)
                spike_trains.append(n_best)
            self._spike_train = np.asarray(spike_trains).transpose()  # we want (n_frames, n_quantiles)
        return self._spike_train







#################################

####    PLOTTING METHODS    ####

################################


    def plot_roi_marked_in_pRGB_and_xcorr(self):
        """ Plot ROI location marked in subvolume's pseudoRGB and XCORR images.

        :return: figure, ax
        """
        cx, cy = self.centroid
        prgb = self.subvolume.visualize_pseudo_rgb()
        xcorrh, xcorrv = self.subvolume.visualize_xcorr()
        fig, ax = plt.subplots(1,3)
        ax[0].imshow(prgb)
        ax[0].add_patch(plt.Polygon(self.polygon, color='m', fill=False))
        ax[0].add_patch(plt.Circle((cx, cy), 18, color='g', fill=False))
        ax[0].set_title('pseudoRGB')
        ax[1].imshow(xcorrh, cmap='gray')
        ax[1].add_patch(plt.Polygon(self.polygon, color='m', fill=False))
        ax[1].set_title('xcorr_h')
        ax[2].imshow(xcorrv, cmap='gray')
        ax[2].add_patch(plt.Polygon(self.polygon, color='m', fill=False))
        ax[2].set_title('xcorr_v')
        for a in ax:
            a.set_xlim(cx - 50, cx + 50)
            a.set_ylim(cy - 50, cy + 50)
            a.set_xticks([])
            a.set_yticks([])
        plt.tight_layout()
        return fig, ax

    def plot_fit(self, quantile_i, spikes_tol=1e-12):
        _ = self._deconvolve_spike_train_for_quantile(quantile_i, spikes_tol=spikes_tol)

        F = self._delta_ff[:, quantile_i]
        quantiles = ['50', '20', '5']
        colors = ['r', 'c', 'b']
        n_hat = self._conv_n_best
        C_hat = self._conv_c_best
        theta_hat = self._conv_theta_best

        sigma, alpha, beta, lamb, gamma = theta_hat

        fig = plt.figure()

        fig.suptitle('Non-neg deconvolution on $\Delta F/F$ and spike train $\hat{n}$ (%s-th percentile)' % quantiles[
            quantile_i])
        gs = plt.GridSpec(3, 1)
        ax1 = fig.add_subplot(gs[0:2])
        ax2 = fig.add_subplot(gs[2:], sharex=ax1)
        axes = np.array([ax1, ax2])

        F_hat = alpha[:, None] * C_hat[None, :] + beta[:, None]

        axes[0].hold(True)
        axes[0].plot(self.time_scale, F, '-%s'%colors[quantile_i], label=r'$\Delta F/F$')
        axes[0].plot(self.time_scale, F_hat.sum(0), '-k', lw=1,
                     label=r'$\hat{\alpha}\hat{C}+\hat{\beta}$')
        axes[0].legend(loc=1, fancybox=True, fontsize='small')
        axes[0].tick_params(labelbottom=False)

        axes[1].plot(self.time_scale, n_hat, '-k')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel(r'$\hat{n}$', fontsize='large')
        axes[1].set_xlim(0, self.time_scale[-1])

        return fig, axes

    def plot_neuropil_correction(self):
        cx, cy = self.centroid

        # get all pixels in 20 micrometer radius around center (20 * .88 = 17.6 pixels ~ 18)
        corona_radius_mum = 20
        radius = np.ceil(corona_radius_mum * self._pixel_size)

        fig = plt.figure('Raw averaged pixel intensity versus neuropil corrected signal.')
        ax0 = plt.subplot2grid((1,3), (0,0))
        ax1 = plt.subplot2grid((1,3), (0,1), colspan=2)
        fig.set_size_inches(17, 5)

        fig.hold(True)
        ax1.plot(self.time_scale, self.f_raw, 'r', label='$F_{raw}(t)$')
        ax1.plot(self.time_scale, self._f_neuropil, 'b', label='$F_{neuropil}(t)$')
        ax1.plot(self.time_scale, self.f_corrected, 'g', label='$F_{true}(t)$')
        plt.legend(loc=1, fancybox=True, fontsize='large')
        ax1.set_xlabel('Time $t$ [s]')
        ax1.set_ylabel('Pixel intensity')
        fig.hold(False)

        ax0.imshow(self.subvolume.visualize_pseudo_rgb())
        ax0.set_xlim(cx - 50, cx + 50)
        ax0.set_ylim(cy - 50, cy + 50)
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax0.add_patch(plt.Polygon(self.polygon, color='y', fill=False))
        ax0.add_patch(plt.Circle((cx, cy), radius, color='y', fill=False))
        plt.tight_layout()
        return fig, [ax0, ax1]

    def plot_baseline_fluorescence(self):
        fig = plt.figure()
        fig.set_size_inches(12, 6)
        fig.suptitle('Baseline $F_0$ (interpolated for regions with events) on the neuropil corrected signal.\n')
        ax0 = fig.add_subplot(111)
        fig.hold(True)
        ax0.plot(self.time_scale, self.f_corrected, 'g', label='$F_{true}$')
        ax0.plot(self.time_scale, self.f0_interpolates[:, 0], 'r', label='$F_0$ (50%tile)')
        ax0.plot(self.time_scale, self.f0_estimates[:, 0], ':k')
        ax0.plot(self.time_scale, self.f0_interpolates[:, 1], 'c', label='$F_0$ (20%tile)')
        ax0.plot(self.time_scale, self.f0_estimates[:, 1], ':k')
        ax0.plot(self.time_scale, self.f0_interpolates[:, 2], 'b', label='$F_0$ (5%tile)')
        ax0.plot(self.time_scale, self.f0_estimates[:, 2], ':k', label='$F_0$ estimate')
        plt.legend(loc=2, fancybox=True, fontsize='small')
        ax0.set_xlabel('Time [s]')
        ax0.set_ylabel('Pixel intensity (ROI average)')
        return fig, ax0

    def plot_mad_thresholds_on_delta_f_estimates(self):
        """ Plot estimated $\Delta F/F$ plus median average deviation (MAD) thresholds for 3x and 4x MAD (for different $F_0$).
        :return: figure, axes
        """
        t = self.time_scale
        # extract MAD thresholds on the delta_f estimates for f0 for every quantile
        mad_threshs = np.asarray([mad(self.delta_ff_estimates[:, i]) for i in xrange(self.delta_ff_estimates.shape[1])])

        fig, (ax0, ax1, ax2) = plt.subplots(3, 1)

        plt.hold(True)
        ax0.plot(t, self.delta_ff_estimates[:, 0], '-r', label='$\Delta F/F$ estimate')
        #ax0.axhline(3 * mad_threshs[0], color='r', label='3x MAD')
        ax0.axhline(4 * mad_threshs[0], color='k', label='4x MAD')
        ax0.axhline(mad_threshs[0], color='k', linestyle='--', label='MAD')
        ax0.set_xlim(0, t[-1])
        ax0.set_ylabel('50%ile', fontsize='large')
        ax0.set_xticks([])
        ax0.set_yticks([])

        ax1.plot(t, self.delta_ff_estimates[:, 1], '-c')
        #ax1.axhline(3 * mad_threshs[1], color='r')
        ax1.axhline(4 * mad_threshs[1], color='k')
        ax1.axhline(mad_threshs[1], color='k', linestyle='--')
        ax1.set_xlim(0, t[-1])
        ax1.set_ylabel('20%ile', fontsize='large')
        ax1.set_xticks([])
        ax1.set_yticks([])

        ax2.plot(t, self.delta_ff_estimates[:, 2], '-b')
        #ax2.axhline(3 * mad_threshs[2], color='r')
        ax2.axhline(4 * mad_threshs[2], color='k')
        ax2.axhline(mad_threshs[2], color='k', linestyle='--')
        ax2.set_xlim(0, t[-1])
        ax2.set_ylabel('5%ile', fontsize='large')
        ax2.set_yticks([])
        ax2.set_xlabel('Time [s]')
        ax0.legend(loc=1, fancybox=True, fontsize='small', bbox_to_anchor = (1.1, 1.4))

        return fig, (ax0, ax1, ax2)

    def plot_preliminary_event_series(self):
        fig, axes = plt.subplots(3, 1)
        plt.hold(True)
        axes[0].plot(self.time_scale, self.preliminary_events[:, 0], '-r', label='Preliminary ev-\nents (50%tile)')
        axes[1].plot(self.time_scale, self.preliminary_events[:, 1], '-c', label='Preliminary ev-\nents (20%tile)')
        axes[2].plot(self.time_scale, self.preliminary_events[:, 2], '-b', label='Preliminary ev-\nents (5%tile)')
        axes[2].set_xlabel('Time [s]')
        plt.hold(False)
        for ax in axes:
            ax.set_yticks([])
            ax.legend(loc=1, fancybox=True, fontsize='small')
        axes[0].set_xticks([])
        axes[1].set_xticks([])
        return fig, axes

    def plot_summary(self):
        df = self.delta_ff
        st = self.spike_train
        cx, cy = self.centroid
        prgb = self.subvolume.visualize_pseudo_rgb()
        xcorrh, xcorrv = self.subvolume.visualize_xcorr()

        fig = plt.figure(self.name)
        fig.set_size_inches(13,8)
        ax1 = plt.subplot2grid((4,3), (0,0))
        ax2 = plt.subplot2grid((4,3), (0,1))
        ax3 = plt.subplot2grid((4,3), (0,2))
        ax4 = plt.subplot2grid((4,3), (1,0), colspan=3)
        ax5 = plt.subplot2grid((4,3), (2,0), colspan=3)
        ax6 = plt.subplot2grid((4,3), (3,0), colspan=3)

        ax1.imshow(prgb)
        ax1.add_patch(plt.Polygon(self.polygon, color='m', fill=False))
        ax1.add_patch(plt.Circle((cx, cy), 18, color='g', fill=False))
        ax1.set_title('pseudoRGB')
        ax2.imshow(xcorrh, cmap='gray')
        ax2.add_patch(plt.Polygon(self.polygon, color='m', fill=False))
        ax2.set_title('XCORR (horiz.)')
        ax3.imshow(xcorrv, cmap='gray')
        ax3.add_patch(plt.Polygon(self.polygon, color='m', fill=False))
        ax3.set_title('XCORR (vert.)')
        for ax in [ax1, ax2, ax3]:
            ax.set_xlim(cx - 50, cx + 50)
            ax.set_ylim(cy - 50, cy + 50)
            ax.set_xticks([])
            ax.set_yticks([])
        #ax4.plot(self.time_scale, self._f_raw, 'm', label='$F_{raw}(t)$')
        #ax4.plot(self.time_scale, self._f_neuropil, 'y', label='$F_{neuropil}(t)$')
        ax4.plot(self.time_scale, self.f_corrected, 'g', label='$F_{true}(t)$')
        ax4.plot(self.time_scale, self.f0_interpolates[:, 0], 'r', label='$F_0$ (50%tile)')
        ax4.plot(self.time_scale, self.f0_estimates[:, 0], ':k')
        ax4.plot(self.time_scale, self.f0_interpolates[:, 1], 'c', label='$F_0$ (20%tile)')
        ax4.plot(self.time_scale, self.f0_estimates[:, 1], ':k')
        ax4.plot(self.time_scale, self.f0_interpolates[:, 2], 'b', label='$F_0$ (5%tile)')
        ax4.plot(self.time_scale, self.f0_estimates[:, 2], ':k', label='$F_0$ estimate')
        #ax4.legend(loc=1, fancybox=True, fontsize='small', bbox_to_anchor = (1.1, 1.12))
        ax4.legend(loc=2, fancybox=True, fontsize='small')
        ax4.set_ylabel('Pixel intensity ($\overline{ROI}$)')
        ax4.set_xticklabels([])
        ax5.plot(self.time_scale, df[:,0], '-r', label='50%ile')
        ax5.plot(self.time_scale, df[:,1], '-c', label='20%ile')
        ax5.plot(self.time_scale, df[:,2], '-b', label='5%ile')
        ax5.set_ylabel('$\Delta F/F$')
        ax5.set_xticklabels([])
        #ax5.legend(loc=1, fancybox=True, fontsize='small', bbox_to_anchor = (1.1, 1.08))
        ax5.legend(loc=2, fancybox=True, fontsize='small')
        ax6.plot(self.time_scale, st[:,0], 'r', label='50%ile')
        ax6.plot(self.time_scale, st[:,1], 'c', label='20%ile')
        ax6.plot(self.time_scale, st[:,2], 'b', label='5%ile')
        ax6.set_xlabel('Time [s]')
        ax6.set_ylabel('$\hat{n}$')
        #ax6.legend(loc=1, fancybox=True, fontsize='small', bbox_to_anchor = (1.1, 1.08))
        ax6.legend(loc=2, fancybox=True, fontsize='small')
        return (ax1, ax2, ax3, ax5, ax6)

    def plot_spike_trains(self, cmap=None):
        # trigger initialization
        cx, cy = self.centroid

        fig = plt.figure()
        fig.set_size_inches(13, 9)
        gs = plt.GridSpec(9, 4)

        # left column
        ax00 = plt.subplot(gs[:4, :1])
        ax01 = plt.subplot(gs[5:9, :1])

        # right column
        ax10 = plt.subplot(gs[0:3, 1:4])
        ax11 = plt.subplot(gs[3:6, 1:4])
        ax12 = plt.subplot(gs[6:9, 1:4])

        # ax00.title('Test title')
        ax00.imshow(self.subvolume.get('mean_green'), cmap=cmap)
        ax00.add_patch(plt.Polygon(self.polygon, color='r', fill=False))
        ax00.set_xticks([])
        ax00.set_yticks([])
        ax00.set_xlim(cx - 50, cx + 50)
        ax00.set_ylim(cy - 50, cy + 50)
        ax00.set_title('mean')

        ax01.imshow(self.subvolume.visualize_xcorr('green')[0], cmap=cmap)
        ax01.add_patch(plt.Polygon(self.polygon, color='r', fill=False))
        ax01.set_xticks([])
        ax01.set_yticks([])
        ax01.set_xlim(cx - 50, cx + 50)
        ax01.set_ylim(cy - 50, cy + 50)
        ax01.set_title('xcorr (horizontal)')

        ax10.plot(self.time_scale, self.spike_train[:, 0], '-r')
        ax10.set_ylabel(r'$\hat{n}$ (50%ile)')
        ax10.set_xlim(0, self.time_scale[-1])

        ax11.plot(self.time_scale, self.spike_train[:, 1], '-c')
        ax11.set_ylabel(r'$\hat{n}$ (20%ile)')
        ax11.set_xlim(0, self.time_scale[-1])

        ax12.plot(self.time_scale, self.spike_train[:, 2], '-b')
        ax12.set_xlabel('Time [s]')
        ax12.set_ylabel(r'$\hat{n}$ (5%ile)')
        ax12.set_xlim(0, self.time_scale[-1])

        plt.tight_layout()
        return fig, (ax00, ax01, ax10, ax11, ax12)

    def plot_f0_interpolation(self):
        """ Plot interpolated vs. estimated $F_0$ for different quantiles (50,20,5 percentiles) for the current ROI.

        :return: figure, axes
        """
        events = self.preliminary_events()
        fig, ax = plt.subplots(6)
        fig.set_size_inches(16, 12)
        fig.suptitle('Interpolated vs. estimated $F_0$ for different quantiles (50,20,5 from top to bottom) (%s).'
                     % (self.meta_info['session_name'] + '_' + self.meta_info['vvppp'] + '_' + str(self.id)))
        ax[0].plot(self.f0_interpolates[:, 0])
        ax[0].plot(self.f0_estimates[:, 0])
        ax[1].plot(events[:, 0])
        ax[2].plot(self.f0_interpolates[:, 1])
        ax[2].plot(self.f0_estimates[:, 1])
        ax[3].plot(events[:, 1])
        ax[4].plot(self.f0_interpolates[:, 2])
        ax[4].plot(self.f0_estimates[:, 2])
        ax[5].plot(events[:, 2])
        return fig, ax

    def plot_intensity_hist(self, ax=None):
        """ Plot intensity distribution of raw fluorescence signal."""
        if ax is None:
            ax = plt.gca()
        return ax.hist(self.f_raw, bins=get_optimal_number_of_bins(self.f_raw))

    def plot_like_annotation_labeler(self, prediction_image=None, label_polygons=None):
        fig = plt.figure()
        fig.set_size_inches(12,10)
        gs = plt.GridSpec(18, 4)
        fig.suptitle(str(self.meta_info) + ' | ID: ' + str(self.id))

        # trigger initialization ##########################################################
        st = self.spike_train
        t = np.arange(st.shape[0]) * self._pixel_size
        cx, cy = self.centroid
        f_corrected = self.f_corrected
        f0_interpolates = self.f0_interpolates


        pm1_img = self.subvolume.get('feats_test-size10_pm_activeGT_very_certain')[:,1].reshape(self.dims[0], self.dims[1])

        mean = self.subvolume.get('mean_green')

        xcorr_h, xcorr_v = self.subvolume.visualize_xcorr('green')

        # upper row #######################################################################
        ax00 = plt.subplot(gs[0:4, 0])
        ax01 = plt.subplot(gs[0:4, 1])
        ax02 = plt.subplot(gs[0:4, 2])
        ax03 = plt.subplot(gs[0:4, 3])

        # middle row ######################################################################
        ax11 = plt.subplot(gs[5:9, 0])
        ax12 = plt.subplot(gs[5:9, 1])
        ax13 = plt.subplot(gs[5:9, 2])
        ax14 = plt.subplot(gs[5:9, 3])

        # lower row #######################################################################
        ax10 = plt.subplot(gs[10:12, :])
        ax20 = plt.subplot(gs[12:14, :], sharex=ax10)
        ax30 = plt.subplot(gs[14:16, :], sharex=ax20)
        ax40 = plt.subplot(gs[16:18, :])

        # upper row #######################################################################
        ax00.imshow(mean)
        #ax00.add_patch(plt.Polygon(self.polygon, color='r', fill=True))
        ax00.set_xticks([])
        ax00.set_yticks([])
        ax00.set_xlim(cx - 50, cx + 50)
        ax00.set_ylim(cy - 50, cy + 50)
        ax00.set_title('mean (zoomed)')

        ax01.imshow(pm1_img)
        ax01.set_xticks([])
        ax01.set_yticks([])
        ax01.set_xlim(cx - 50, cx + 50)
        ax01.set_ylim(cy - 50, cy + 50)
        ax01.set_title("prob. map")

        ax02.imshow(xcorr_h)
        #ax01.add_patch(plt.Polygon(self.polygon, color='r', fill=False))
        ax02.set_xticks([])
        ax02.set_yticks([])
        ax02.set_xlim(cx - 50, cx + 50)
        ax02.set_ylim(cy - 50, cy + 50)
        ax02.set_title('xcorr (horiz.)')

        ax03.imshow(xcorr_v)
        #ax02.add_patch(plt.Polygon(self.polygon, color='r', fill=False))
        ax03.set_xticks([])
        ax03.set_yticks([])
        ax03.set_xlim(cx - 50, cx + 50)
        ax03.set_ylim(cy - 50, cy + 50)
        ax03.set_title('xcorr (vert.)')

        # middle row #######################################################################
        ax11.imshow(mean)
        ax11.add_patch(plt.Polygon(self.polygon, color='r', fill=True))
        ax11.set_xticks([])
        ax11.set_yticks([])
        ax11.set_title('mean')

        ax12.imshow(prediction_image, cmap='gray')
        ax12.add_patch(plt.Polygon(self.polygon, color='r', fill=True))
        ax12.set_xticks([])
        ax12.set_yticks([])
        for patch in label_polygons[0]:
            ax12.add_patch(patch)
        ax12.set_title('labeled prediction (++)', fontsize='small')

        ax13.imshow(prediction_image, cmap='gray')
        ax13.set_xticks([])
        ax13.set_yticks([])
        for patch in label_polygons[1]:
            ax13.add_patch(patch)
        ax13.set_title('labeled prediction (+++)', fontsize='small')

        ax14.imshow(prediction_image, cmap='gray')
        ax14.set_xticks([])
        ax14.set_yticks([])
        for patch in label_polygons[2]:
            ax14.add_patch(patch)
        ax14.set_title('labeled prediction (+++x)', fontsize='small')

        # lower row #######################################################################
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

        ax40.plot(t, f_corrected, 'g', label='raw signal (corrected)')
        ax40.plot(t, f0_interpolates[:, 0], 'r', label='$F_0$ (50%tile)')
        ax40.plot(t, f0_interpolates[:, 1], 'c', label='$F_0$ (20%tile)')
        ax40.plot(t, f0_interpolates[:, 2], 'b', label='$F_0$ (5%tile)')
        # plt.legend(loc=1, fancybox=True, fontsize='small')

        ax40.set_xlabel('Time (s)')
        ax40.set_ylabel('F')
        ax40.set_xlim(0, t[-1])
        ax40.hold(False)

        return fig

    def save_vizzz(self, out_dir, load_precomputed_signals=True, overwrite_plots=False, dpi=300):
        create_out_path(out_dir, except_on_exist = not overwrite_plots) # don't except on overwrite
        if load_precomputed_signals:
            self.load_precomputed_signals()

        self.plot_neuropil_correction()
        plt.savefig(os.path.join(out_dir, '%s-neuropil_correction.png'%self.name), dpi=dpi)

        self.plot_preliminary_event_series()
        plt.savefig(os.path.join(out_dir, '%s-preliminary_event_series.png'%self.name), dpi=dpi)

        self.plot_mad_thresholds_on_delta_f_estimates()
        plt.savefig(os.path.join(out_dir, '%s-MAD_thresholded_DeltaF_estimates.png'%self.name), dpi=dpi)

        self.plot_baseline_fluorescence()
        plt.savefig(os.path.join(out_dir, '%s-fluorescence.png'%self.name), dpi=dpi)
        for i in xrange(3):
            self.plot_fit(i)
            plt.savefig(os.path.join(out_dir, '%s-FNND_fit_quantile%d.png'%(self.name, i)), dpi=dpi)

        self.plot_spike_trains()
        plt.savefig(os.path.join(out_dir, '%s-spike_trains.png'%self.name), dpi=dpi)

        self.plot_summary()
        plt.savefig(os.path.join(out_dir, '%s-summary.png'%self.name), dpi=dpi)
        plt.close('all')