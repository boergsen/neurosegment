__version__ = '0.85'

import numpy as np
import vigra
import matplotlib.pyplot as plt

def median_absolute_deviation(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    arr = np.ma.array(arr).compressed() # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.abs(arr - med))

def get_optimal_number_of_bins(values, method='freedman'):
    assert method in ['freedman', 'schill']
    num_instances = len(values)
    # Get the number of bins.
    v_min, v_25, v_75, v_max = np.percentile(values, [0, 25, 75, 100])
    if method=='freedman':
        width = 2 * (v_75 - v_25) / (len(values) ** (1/3.0)) # Freedman Diaconis width
    if method=='schill':
        width = (v_max - v_min) / (len(values) ** (1/3.0))   # optimized Schill width
    num_bins = int(round((v_max - v_min) / width))
    assert num_bins > 0
    return num_bins

def create_2d_edge_graph(x, y):
    """ Creates the edge set defining a 2D image grid (hacky but works).
    :param x: width
    :param y: height
    :return: edges array of shape((2xy-x-y, 2))
    """
    n_pixels = x * y
    n_edges = 2 * x * y - x - y
    edges = []
    for pixel in xrange(1, n_pixels + 1):
        # check for horizontal boundary
        if (pixel % x) != 0:
            edges.append((pixel - 1, pixel))
        # check for last vertical line
        if pixel <= (n_pixels - x):
            edges.append((pixel - 1, pixel - 1 + x))
    assert len(edges) == n_edges
    return np.asarray(edges)

def norm_cross_correlation(x, y):
    def l2_norm(x, mu_x):
        return np.sqrt(np.sum(np.power((x - mu_x), 2)))
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    mean_x = (x - mu_x) / l2_norm(x, mu_x)
    mean_y = (y - mu_y) / l2_norm(y, mu_y)
    return np.dot(mean_x, mean_y)

def compute_class_weights(Y):
    n_classes = len(np.bincount(np.hstack(Y)))
    class_weights = 1. / np.bincount(np.hstack(Y))
    class_weights *= float(n_classes) / np.sum(class_weights)
    return class_weights

def construct_RGB_projection_from_volume(volume):
    """ Computes standard deviation, mean, and max value for each pixel
    of a volume and puts each resulting projection in a different RGB channel.
    """
    pstd = np.std(volume, axis=0)
    pmax = np.max(volume, axis=0)
    pmean = np.mean(volume, axis=0)
    rgb = np.dstack( [pstd, pmean, pmax] )
    return vigra.RGBImage(rgb)

def compute_ilastik_features(img):
    assert len(img.shape) == 2
    def add_features(res, f):
        if len(f.shape) == 2:
            f.shape += (1,)
        if len(res) == 0:
            return np.asarray(f)
        return np.append(res, np.asarray(f), axis = 2)

    img = vigra.ScalarImage(img)
    res = []
    cnt=0
    print 'adding Gaussian smooothing filter with sigma = 0.3'
    res = add_features(res, vigra.filters.gaussianSmoothing(img, 0.3))
    cnt+=1
    print res.shape
    for sigma in [0.7, 1.0, 1.6, 3.5, 5, 10]:
        print "adding other filters with sigma =", sigma
        print 'count:', cnt
        # color
        res = add_features(res, vigra.filters.gaussianSmoothing(img, sigma))
        cnt+=1
        print res.shape
        # edge
        res = add_features(res, vigra.filters.laplacianOfGaussian(img, sigma))
        cnt+=1
        print res.shape
        res = add_features(res, vigra.filters.gaussianGradientMagnitude(img, sigma))
        cnt+=1
        print res.shape
        res = add_features(res, (vigra.filters.gaussianSmoothing(img,sigma)-vigra.filters.gaussianSmoothing(img,0.66 * sigma)))
        cnt+=1
        print res.shape
        # texture
        res = add_features(res, vigra.filters.structureTensorEigenvalues(img, sigma, sigma/2.0))
        cnt+=1
        print res.shape
        res = add_features(res, vigra.filters.hessianOfGaussianEigenvalues(img, sigma))
        cnt+=1
        print res.shape
    print 'count:', cnt
    print 'result shape:', res.shape
    return np.asarray(res)

def compute_ilastik_features_RGB(RGBimg):
    assert (len(RGBimg.shape) == 3) and (RGBimg.shape[2] == 3)
    feats = np.ndarray((RGBimg.shape[0],RGBimg.shape[1],0), dtype=np.float32)
    for ch_idx in xrange(RGBimg.shape[2]):
      feats = np.concatenate((feats, compute_ilastik_features(RGBimg[:,:,ch_idx])), axis=2)
    return feats

def shrink_image(image, factor=10):
    return image[::factor, ::factor]

def cutoff(arr, thresh):
    from copy import deepcopy
    arr = deepcopy(arr)
    arr[arr>thresh] = thresh
    return arr

def lowpass_mean(arr):
    arr[arr>np.mean(arr)] = np.mean(arr)
    return arr

def highpass_mean(arr):
    arr[arr<np.mean(arr)] = np.mean(arr)
    return arr
    
def normalize_image_to_zero_one(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def clear_ticks(axes):
    if isinstance(axes, list):
        for el in xrange(len(axes)):
            axes[el].set_xticks([])
            axes[el].set_yticks([])
    elif isinstance(axes, np.ndarray):
        for axel in axes.ravel():
            axel.set_xticks([])
            axel.set_yticks([])
    else: # just a single axis
        axes.set_xticks([])
        axes.set_yticks([])
    return axes

def savefig(file_out, img, title, cmap='gray'):
    fig=plt.figure()
    plt.imshow(img, cmap)
    plt.title(title)
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    fig.savefig(file_out)
    plt.close()

def convert_index_to_binary_mask(new_shape, index_mask):
    binary_mask = np.ones((new_shape), dtype=np.bool)
    binary_mask[index_mask.all()] = False
    return binary_mask

import os
def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

def create_out_path(out_path, except_on_exist=False):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    else:
        if except_on_exist:
            raise IOError('path %s already exists' % out_path)


def balance_data(instances, labels, per_class_sample_size=None, verbose=0):
    """ Balance data such that it has the same amount of positive and negative pixel samples.
    NOTE: Majority class is downsampled to per_class_sample_size. If this value is larger than the maximum
    number of samples in the minority class, the latter is used instead.

    :param instances: features of shape (n_pixels, n_features)
    :param labels: 0-1 pixel labels of shape (n_pixels,)
    :param per_class_sample_size: set maximum number of samples per class to be returned
    :return: balanced instances, labels
    """
    assert instances.shape[0] == labels.shape[0]

    pos_idxs_shuffled = np.random.permutation(np.where(labels==1)[0])
    neg_idxs_shuffled = np.random.permutation(np.where(labels==0)[0])

    # check wich class has to be downsampled (i.e. which is the majority class)
    if pos_idxs_shuffled.shape[0] < neg_idxs_shuffled.shape[0]:
        max_sample_size = len(pos_idxs_shuffled)
        majority_class = "negative class"
    else:
        max_sample_size = len(neg_idxs_shuffled)
        majority_class = 'positive class'
    if verbose > 0: print 'downsampling', majority_class

    # check maximum number of possible samples per class
    if per_class_sample_size is None:
        per_class_sample_size = max_sample_size
    elif per_class_sample_size > max_sample_size:
        print '  WARNING: Trying to take more samples than data points available for the minority class.\n' \
              '  Reducing balanced sample size to', max_sample_size
        per_class_sample_size = max_sample_size
    #else: class sample size ok

    if verbose > 0: print "balance data..."
    balanced_idxs = np.concatenate((pos_idxs_shuffled[:per_class_sample_size], neg_idxs_shuffled[:per_class_sample_size]))
    balanced_idxs = np.random.permutation(balanced_idxs)
    return instances[balanced_idxs], labels[balanced_idxs]

def precision(tp, fp):
    return tp*1.0/(tp+fp)

def recall(tp, fn):
    return tp*1.0/(tp+fn)

def accuracy(tp, fp, tn, fn):
    return 1.0*(tp+tn)/(tp+fp+tn+fn)

def fpr(fp, tn):
    return 1.0*fp/(fp+tn)

def tpr(tp, fn):
    return 1.0*tp/(tp+fn)

def shuffle(arr, n_times=5):
    print 'shuffling %d times' % n_times
    for i in xrange(n_times):
        arr = np.random.permutation(arr)
    return arr


# import h5py as h5
# def h5_save_to_disk(path, field, data): # appends by default if file exists
#     with h5.File(path) as h5out:
#         h5out.create_dataset(name=field, data=data)
#
# def h5_load_from_disk(path, field):
#     with h5.File(path, 'r') as h5in:
#         return np.asarray(h5in[field])
#
# def h5_delete_dataset(path, field):
#     with h5.File(path, 'w') as h5file:
#         try:
#             h5file.delete(field)
#         except:
#             raise NotImplementedError("REALLY?")