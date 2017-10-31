__version__ = '0.1'


import argparse
import h5py as h5
import json
import numpy as np
import os
import sys
from glob import glob
from numpy import array, zeros, frombuffer
from vigra import RGBImage

from core.methods import compute_ilastik_features_RGB, create_2d_edge_graph, norm_cross_correlation


def load_volume_and_gt(neurofinder_volume_dir):

    # load the images
    with open(neurofinder_volume_dir + 'images/conf.json') as f:
        dims = json.load(f)['dims']

    with open(neurofinder_volume_dir + 'info.json') as f:
        vol_name = json.load(f)['id']

    files = glob(neurofinder_volume_dir + 'images/*/*.bin')
    if len(files) == 0: # only simon perons data is subdivided in volumes under images
        files = glob(neurofinder_volume_dir + 'images/*.bin')

    def toarray(f):
        with open(f) as fid:
            return frombuffer(fid.read(),'uint16').reshape(dims, order='F')

    volume = array([toarray(f) for f in files])

    # load the sources
    with open(neurofinder_volume_dir + 'sources/sources.json') as f:
        sources = json.load(f)

    def tomask(coords):
        mask = zeros(dims)
        mask[zip(*coords)] = 1
        return mask

    masks = array([tomask(s['coordinates']) for s in sources])

    gt = masks.sum(axis=0)

    return volume, gt , vol_name, dims


def build_volumes(neurofinder_data_dir='data/neurofinder_svoboda/raw/', h5_out_path='data/neurofinder_svoboda/H5/'):

    print 'Building volumes, packing everything in H5 files'
    nf_data_folders = os.listdir(neurofinder_data_dir)

    for nf_folder in nf_data_folders:
        in_path = neurofinder_data_dir + nf_folder + '/'

        print 'Loading neurofinder volume data and ground truth'
        volume, gt, vol_name, dims = load_volume_and_gt(in_path)

        h5_out = h5_out_path + vol_name + '.h5'

        print 'Creating out file', h5_out
        h5file = h5.File(h5_out)

        print 'Adding volume data'
        h5file.create_dataset('volume_green', data=volume, chunks=(1, 32, 32), dtype=np.uint16)

        print "Adding ground truth"
        h5file.create_dataset('gt', data=gt)

        print "Calculating mean, max and std for green channel"
        mean = np.mean(volume, axis=0)
        max = np.max(volume, axis=0)
        std = np.std(volume, axis=0)

        print "Adding mean, max, std"
        h5file.create_dataset('mean_green', data=mean, dtype=np.float64)
        h5file.create_dataset('max_green', data=max, dtype=np.float64)
        h5file.create_dataset('std_green', data=std, dtype=np.float64)

        print "Adding pseudoRGB image"
        pseudo_rgb_green = RGBImage( np.dstack( [std, mean, max] ) )
        h5file.create_dataset('pseudo_rgb_green', data=pseudo_rgb_green, dtype=np.float64)

        print "Computing ilastik features on pseudo RGB image"
        feats_RGB_green = compute_ilastik_features_RGB(pseudo_rgb_green)
        h5file.create_dataset('feats_ilastik_pseudoRGB_green', data=feats_RGB_green, dtype=np.float64)

        print 'Preparing computation of temporal signals'
        edges = create_2d_edge_graph(dims[0], dims[1])
        n_edges = edges.shape[0]
        n_edge_features = 1
        edge_features = np.ndarray((n_edges, n_edge_features))
        vol_rshp = volume.reshape((-1, dims[0] * dims[1]))

        print 'Computing cross-correlation'
        for edge_idx in xrange(n_edges):
            i,j = edges[edge_idx] # get pixel indices for current edge
            pairwise_feature_ij = norm_cross_correlation(vol_rshp[:, i], vol_rshp[:, j])
            edge_features[edge_idx] = pairwise_feature_ij

        print 'Adding xcorr'
        h5file.create_dataset(data=edge_features, name='feats_xcorr_green')

        print 'Computing correlation'
        for edge_idx in xrange(n_edges):
            i,j = edges[edge_idx] # get pixel indices for current edge
            pairwise_feature_ij = np.correlate(vol_rshp[:, i], vol_rshp[:, j])
            edge_features[edge_idx] = pairwise_feature_ij

        print 'Adding corr'
        h5file.create_dataset(data=edge_features, name='feats_corr_green')

        print 'Adding subvol info'
        h5file.create_dataset(data=dims, name='dims', dtype=np.uint16)
        h5file.create_dataset(data=vol_name, name='id')

        h5file.close()
    print 'Done.'


def process_command_line():
    # create a parser instance
    parser = argparse.ArgumentParser(description = "Convert data from Neurofinder competition to HDF5.", formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    # add arguments to parser
    parser.add_argument("--in_dir", type=str, default=False, help="Directory with Neurofinder data.")
    parser.add_argument("--out_dir", type=str, default=False, help="Output directory to save HDF5 files.")
    args = parser.parse_args()
    return args

def main():
    args = process_command_line()

    if args.in_dir and args.out_dir:
        build_volumes(neurofinder_data_dir=args.in_dir, h5_out_path=args.out_dir)

    return 0

if __name__ == "__main__":
    status = main()
    sys.exit(status)