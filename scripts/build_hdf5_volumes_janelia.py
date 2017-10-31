__version__ = '0.7'


import h5py as h5
import numpy as np
from scipy.io import loadmat
from vigra import RGBImage

import matplotlib.pyplot as plt
import vigra.impex

from core.methods import compute_ilastik_features_RGB, create_2d_edge_graph, norm_cross_correlation


def extract_ROIs_dict_for_session(matlab_session_object):
    '''
    Takes a session file in .mat format (MATLAB) and extracts all ROIs (=marked neurons) from it.
    Returns a dict with ROI ID as key (valid within a session file) and the neuron pixel indices of the ROI as value.
    '''
    # convenience
    descrHashField=matlab_session_object['s']['timeSeriesArrayHash'][0,0][0,0][3][0]

    rois = {}
    for i in xrange(1, len(descrHashField)): # for every fov
        current_fov_value = descrHashField[i][0,0][2][0,0][0]
        no_of_fields_in_fov = len(current_fov_value)
        for j in xrange(no_of_fields_in_fov):
            rois_field = current_fov_value[j][2] # every ROI is a set of pixels defining an actual neuron
            for k in xrange(rois_field.shape[1]): # for every field in rois
                roi = rois_field[0,k]
                roi_index = int(roi[0].squeeze())
                roi_pixel = roi[2].squeeze()
                rois[roi_index] = roi_pixel
    return rois

def extract_volumes_with_gt_from_session(matlab_session_object, gt_save_dir=None):
    # result
    volumes={}

    # convenience
    valueField=matlab_session_object['s']['timeSeriesArrayHash'][0,0][0,0][2][0]

    subvolumes = [valueField[i][0,0][7] for i in xrange(1,len(valueField))]
    roiID_to_pixelID_dict = extract_ROIs_dict_for_session(matlab_session_object)

    '''
    The follwing gives the TIFF file list of some imaging plane ppp, living inside a subvolume vv in Matlab
    (file title goes like 'fov_vvppp'): s.timeSeriesArrayHash.value{1,subvolume_vv}.imagingPlane{1, imaging_plane_ppp}

    E.g., in Python: for subvolume numero 10 ([9]), this shows some imaging plane ([2]; a 1x3 struct, strangely max 3!):
    subvolumes[9][0][2][0][0][1]

    Set the last index to 0 to get the IDs of the ROIs for that imaging plane.

    Note: to get the info what the actual neuron pixels are, use the ROI IDs in the extracted dict.
    '''
    for subvolume in subvolumes:
        for imaging_plane in subvolume[0]:
            roi_ids = imaging_plane[0][0][0][0]
            tif_files = imaging_plane[0][0][1][0]

            # clean-up
            roi_ids = [int(roi_ids[i]) for i in xrange(len(roi_ids))]
            tif_files = [str(tif_files[i][0]).split('imaging_data/')[1] for i in xrange(len(tif_files))]

            # workaround due to errors in original data ("fluo_batch_out" folder missing in some file paths)
            if tif_files[0].find('fluo_batch_out') == -1:
                tmp=[tif_files[i].split('/') for i in xrange(len(tif_files))]
                tif_files=[tmp[i][0] + '/fluo_batch_out/' + tmp[i][1] for i in xrange(len(tmp))]

            fov_name = tif_files[0].split('/')[0]
            #fov_vv, fov_ppp = fov_name[:2], fov_name[2:]

            print 'Extracting ground truth for %s' % fov_name
            fov_gt = np.zeros(512*512, dtype=np.uint8)
            for i in xrange(len(roi_ids)):
                fov_gt[roiID_to_pixelID_dict[int(roi_ids[i])]] = 1

            volumes[fov_name] = (fov_gt, tif_files)

    if gt_save_dir != None:
        gts=[volumes[volumes.keys()[i]][0] for i in xrange(len(volumes.keys()))]
        names=[volumes[volumes.keys()[i]][1][0].split('/')[0] for i in xrange(len(volumes.keys()))]
        session_id=str(matlab_session_object['s']['metaDataHash'][0,0][0][0][1][0][8][0])
        session_date=str(matlab_session_object['s']['metaDataHash'][0,0][0][0][1][0][10][0])
        session_file_name=session_id + '_' + session_date[0:4] + '_' + session_date[4:6] + '_' + session_date[6:]
        for i in xrange(len(gts)):
            plt.imsave(gt_save_dir + session_file_name + '_' + names[i], gts[i].reshape(512,512), cmap='gray')

    return volumes

def create_volume_from_tiff_file_names(tiff_file_names, data_path, out_dir, channel):

    assert channel in ['green', 'red']

    width, height = 512, 512

    # file name stuff
    tmp=tiff_file_names[0].split('/')[-1].split('_')
    session_name = tmp[3]+ '_' + tmp[4] + '_' + tmp[5] + '_'+ tmp[6]
    fov_name = tiff_file_names[0].split('/')[0]
    h5_file_name = session_name + '_' + fov_name + '.h5'
    h5_dataset_name = 'volume_%s' % channel
    h5_output_path = data_path + out_dir + h5_file_name
    tiff_file_paths = [data_path + session_name + '/' + tiff_file_names[i] for i in xrange(len(tiff_file_names))]

    print "Get overall number of frames from all tiffs for current volume..."
    max_frames = 0
    for tiff_path in tiff_file_paths:
        max_frames += vigra.impex.numberImages(tiff_path)
    assert (max_frames % 2) == 0
    max_frames /= 2

    print "Creating output file %s ..." % h5_file_name
    h5file = h5.File(h5_output_path, 'a')
    vol = h5file.create_dataset(h5_dataset_name, (max_frames, width, height), chunks=(1, 32, 32), dtype=np.uint16)

    # add all frames to a single H5 file
    frame_count = 0
    for tiff_path in tiff_file_paths:
        tif = vigra.impex.readVolume(tiff_path, dtype=np.uint16)
        tif = tif.squeeze()

        if channel == 'green':
            print "Extracting green channel..."
            tif = tif[:,:,::2]
        if channel == 'red':
            print "Extracting red channel..."
            tif = tif[:,:,1::2]

        width, height, n_frames = tif.shape

        print "Processing file: %s, Number of frames: %d" % (tiff_path, n_frames)
        for i in xrange(frame_count, n_frames+frame_count):
            vol[i] = tif[:,:,i-frame_count]
            h5file.flush()
        frame_count += n_frames
    h5file.close()
    print "%d frames written to output file %s , data set name: \"%s\". Done." % (max_frames, h5_file_name, h5_dataset_name)

    return h5_output_path

def build_volumes(data_path, h5_out_path='data/new_stripped/final/', gt_save_dir=None, print_info=True):

    print 'Getting session objects from matlab files...'
    session_file_names = ['an197522_2013_03_08_session.mat',
                          'an197522_2013_03_10_session.mat',
                          'an229717_2013_12_01_session.mat',
                          'an229719_2013_12_02_session.mat',
                          'an229719_2013_12_05_session.mat']
    matlab_session_objects = [loadmat(data_path + s) for s in session_file_names]
    session_volumes_with_gt = [extract_volumes_with_gt_from_session(s, gt_save_dir=gt_save_dir) for s in matlab_session_objects]

    print 'Building volumes, packing everything in H5 files...'
    session_cnt = 0
    for session in session_volumes_with_gt:
        for fov in session.keys():
            fov_name = fov
            fov_gt = session[fov][0].reshape(512, 512)
            fov_tiffs = session[fov][1]
            fov_path=data_path+session_file_names[session_cnt].split("_session.mat")[0] +'/'+fov_tiffs[0].split('Image_Registration')[0]

            # FIXME: an197522_2013_03_08_fov_04001 is 12GB and too big for memory so skip it
            if (fov_name == 'fov_04001') and (fov_path.split('/')[-4] == 'an197522_2013_03_08'): continue

            print "Creating H5 dataset for %s" % fov_name
            _ = create_volume_from_tiff_file_names(fov_tiffs, data_path, h5_out_path, channel='green')
            volume_path = create_volume_from_tiff_file_names(fov_tiffs, data_path, h5_out_path, channel='red')

            print "Adding ground truth to dataset..."
            h5file = h5.File(volume_path)
            h5file.create_dataset('gt', data=fov_gt)

            print "Calculating mean, max and std for green channel..."
            fov_mean_green = np.mean(h5file['volume_green'], axis=0)
            fov_max_green = np.max(h5file['volume_green'], axis=0)
            fov_std_green = np.std(h5file['volume_green'], axis=0)

            print "Calculating mean, max and std for red channel..."
            fov_mean_red = np.mean(h5file['volume_red'], axis=0)
            fov_max_red = np.max(h5file['volume_red'], axis=0)
            fov_std_red = np.std(h5file['volume_red'], axis=0)

            print "Adding mean, max, std for both channels to dataset..."
            h5file.create_dataset('mean_green', data=fov_mean_green, dtype=np.float64)
            h5file.create_dataset('max_green', data=fov_max_green, dtype=np.float64)
            h5file.create_dataset('std_green', data=fov_std_green, dtype=np.float64)
            h5file.create_dataset('mean_red', data=fov_mean_red, dtype=np.float64)
            h5file.create_dataset('max_red', data=fov_max_red, dtype=np.float64)
            h5file.create_dataset('std_red', data=fov_std_red, dtype=np.float64)

            print "Adding pseudoRGB images for both channels..."
            pseudo_rgb_green = RGBImage( np.dstack( [fov_std_green, fov_mean_green, fov_max_green] ) )
            pseudo_rgb_red = RGBImage( np.dstack( [fov_std_red, fov_mean_red, fov_max_red] ) )
            h5file.create_dataset('pseudo_rgb_green', data=pseudo_rgb_green, dtype=np.float64)
            h5file.create_dataset('pseudo_rgb_red', data=pseudo_rgb_red, dtype=np.float64)

            print "Computing ilastik features on RGB images for both channels..."
            feats_RGB_green = compute_ilastik_features_RGB(pseudo_rgb_green)
            feats_RGB_red = compute_ilastik_features_RGB(pseudo_rgb_red)
            print "\tAdding features for each channel..."
            h5file.create_dataset('feats_ilastik_pseudoRGB_green', data=feats_RGB_green, dtype=np.float64)
            h5file.create_dataset('feats_ilastik_pseudoRGB_red', data=feats_RGB_red, dtype=np.float64)

            for channel in ['green','red']:
                print 'Computing cross-correlation for volume, %s channel...'%channel
                edges = create_2d_edge_graph(512,512)
                n_edges = edges.shape[0]
                n_edge_features = 1
                edge_features = np.ndarray((n_edges, n_edge_features))
                vol_rshp = np.asarray(h5file['volume_%s'%channel]).reshape((-1, 512*512))

                for edge_idx in xrange(n_edges):
                    i,j = edges[edge_idx] # get pixel indices for current edge
                    pairwise_feature_ij = norm_cross_correlation(vol_rshp[:, i], vol_rshp[:, j])
                    edge_features[edge_idx] = pairwise_feature_ij

                h5file.create_dataset(data=edge_features, name='feats_xcorr_%s'%channel)

            for channel in ['green','red']:
                print 'Computing correlation for volume, %s channel...'%channel
                edges = create_2d_edge_graph(512,512)
                n_edges = edges.shape[0]
                n_edge_features = 1
                edge_features = np.ndarray((n_edges, n_edge_features))
                vol_rshp = np.asarray(h5file['volume_%s'%channel]).reshape((-1, 512*512))

                for edge_idx in xrange(n_edges):
                    i,j = edges[edge_idx] # get pixel indices for current edge
                    pairwise_feature_ij = np.correlate(vol_rshp[:, i], vol_rshp[:, j])
                    edge_features[edge_idx] = pairwise_feature_ij[0]

                h5file.create_dataset(data=edge_features, name='feats_corr_%s'%channel)


            # print "Adding Janelia projections for volume from data dump..."
            # h5file.create_dataset('janelia_master_imreg_image_chan_01',
            #                       data=vigra.impex.readVolume(fov_path + 'master_imreg_image_chan_01.tif').squeeze())
            # h5file.create_dataset('janelia_master_imreg_image_chan_02',
            #                       data=vigra.impex.readVolume(fov_path + 'master_imreg_image_chan_02.tif').squeeze())
            #
            # h5file.create_dataset('janelia_session_maxproj_chan_01',
            #                       data=vigra.impex.readVolume(fov_path + 'session_maxproj_chan_01.tif').squeeze())
            # h5file.create_dataset('janelia_session_maxproj_chan_02',
            #                       data=vigra.impex.readVolume(fov_path + 'session_maxproj_chan_02.tif').squeeze())
            #
            # h5file.create_dataset('janelia_session_mean_chan_01',
            #                       data=vigra.impex.readVolume(fov_path + 'session_mean_chan_01.tif').squeeze())
            # h5file.create_dataset('janelia_session_mean_chan_02',
            #                       data=vigra.impex.readVolume(fov_path + 'session_mean_chan_02.tif').squeeze())
            #
            # h5file.create_dataset('janelia_session_pertrial_mean_chan_01',
            #                       data=vigra.impex.readVolume(fov_path + 'session_pertrial_mean_chan_01.tif').squeeze())
            # h5file.create_dataset('janelia_session_pertrial_mean_chan_02',
            #                       data=vigra.impex.readVolume(fov_path + 'session_pertrial_mean_chan_02.tif').squeeze())
            #
            # h5file.create_dataset('janelia_session_pertrial_sd_chan_01',
            #                       data=vigra.impex.readVolume(fov_path + 'session_pertrial_sd_chan_01.tif').squeeze())
            #
            # h5file.create_dataset('janelia_session_pertrial_sd_chan_02',
            #                       data=vigra.impex.readVolume(fov_path + 'session_pertrial_sd_chan_02.tif').squeeze())
            #
            # h5file.create_dataset('janelia_sdmax_chan_01',
            #                       data=vigra.impex.readVolume(fov_path + 'session_sdmax_chan_01.tif').squeeze())
            # h5file.create_dataset('janelia_sdmax_chan_02',
            #                       data=vigra.impex.readVolume(fov_path + 'session_sdmax_chan_02.tif').squeeze())
            #
            # h5file.create_dataset('janelia_sdmean_chan_01',
            #                       data=vigra.impex.readVolume(fov_path + 'session_sdmean_chan_01.tif').squeeze())
            # h5file.create_dataset('janelia_sdmean_chan_02',
            #                       data=vigra.impex.readVolume(fov_path + 'session_sdmean_chan_02.tif').squeeze())
            #
            # h5file.close()

        session_cnt += 1

# data_path=os.getenv("HOME") + '/thesis/code/data/new_stripped/'
# build_volumes(data_path, h5_out_path='final/')
#
# print 'Done.'
# sys.exit(0)