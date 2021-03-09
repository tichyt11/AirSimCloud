import matplotlib.pyplot as plt
import numpy as np
import os
from tools.dem_handling import DemHandler
import math
from matplotlib.ticker import PercentFormatter
import matplotlib.cm as cm
from matplotlib.colors import Normalize


GTcloud = 'C:/Users/tomtc/OneDrive/Desktop/MazeResults/NoWater/LargeCV/Clouds/GT.ply'

eval = os.path.join(os.getcwd(), 'evaluation_data')
GTdem = os.path.join(eval, 'GTdem.tif')

ideal_recon_dir = 'C:/Users/tomtc/OneDrive/Desktop/MazeResults/NoWater/LargeSurvey/Ideal'
colmap_cloud = os.path.join(ideal_recon_dir, 'Colmap_295min.ply')
odm_cloud = os.path.join(ideal_recon_dir, 'ODM_60min.las')
omvgs_cloud = os.path.join(ideal_recon_dir, 'omvgs_res1_41min.las')

colmap_dem_ideal = os.path.join(eval, 'ideal_colmap.tif')
odm_dem_ideal = os.path.join(eval, 'ideal_odm.tif')
omvgs_dem_ideal = os.path.join(eval, 'ideal_omvgs.tif')


def prepare_for_hist(dem, gt):
    nodata = 1000
    # diff = (np.abs(dem - gt))
    diff = dem - gt
    diff = np.ma.masked_array(diff, mask=(dem == nodata) | (gt == nodata))
    diff = diff[~diff.mask].flatten()
    return diff


def dem_diff(dem, gt):
    nodata = 1000
    diff = dem - gt
    diff = np.ma.masked_array(diff, mask=(dem == nodata) | (gt == nodata))
    return diff


def main():
    grid_origin = np.array([-16, -80])  # bottom left corner of dsm world coordinates
    h, w = 500, 1220
    res = 0.3
    nodata = 1000

    handler = DemHandler(grid_origin, h, w, res, nodata)
    # handler.create_heightmap(omvgs_cloud, omvgs_dem_ideal)

    # load maps
    dem_colmap = handler.load_heightmap(colmap_dem_ideal)
    dem_colmap = np.ma.masked_array(dem_colmap, mask=dem_colmap==nodata)
    dem_odm = handler.load_heightmap(odm_dem_ideal)
    dem_odm = np.ma.masked_array(dem_odm, mask=dem_odm==nodata)
    dem_omvgs = handler.load_heightmap(omvgs_dem_ideal)
    dem_omvgs = np.ma.masked_array(dem_omvgs, mask=dem_omvgs==nodata)
    gt = handler.load_heightmap(GTdem)
    gt = np.ma.masked_array(gt, mask=gt==nodata)

    # mask and difference
    colmap_h = prepare_for_hist(dem_colmap, gt)
    odm_h = prepare_for_hist(dem_odm, gt)
    omvgs_h = prepare_for_hist(dem_omvgs, gt)

    # histogram
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    n, bins, _ = plt.hist([colmap_h, odm_h, omvgs_h], bins=500, range=(-2, 2), cumulative=False, histtype='step',
             weights=[np.ones(len(colmap_h)) / len(colmap_h), np.ones(len(odm_h)) / len(odm_h), np.ones(len(omvgs_h)) / len(omvgs_h)])
    plt.xticks(np.linspace(0,5,21))
    plt.grid()
    plt.legend(['colmap', 'odm', 'omvgs'])
    plt.show()

    # plot histogram with lines instead of bins, looks better
    plt.plot(bins[0:-1], n[0])
    plt.plot(bins[0:-1], n[1])
    plt.plot(bins[0:-1], n[2])
    plt.legend(['colmap', 'odm', 'omvgs'])
    plt.grid()
    plt.show()

    # show heightmaps with common colorbar
    # cmap = cm.get_cmap('CMRmap')
    # normalizer = Normalize(np.ma.masked_array.min(np.ma.concatenate([dem_omvgs, dem_odm, dem_colmap, gt])),
    #                        np.ma.masked_array.max(np.ma.concatenate([dem_omvgs, dem_odm, dem_colmap, gt])))
    # im = cm.ScalarMappable(norm=normalizer, cmap=cmap)
    #
    # for i in [dem_colmap, dem_odm, dem_omvgs, gt]:
    #     plt.imshow(i, cmap=cmap, norm=normalizer)
    #     plt.colorbar(im, shrink=0.6, label='[m]')
    #     plt.show()

    # fig, axes = plt.subplots(nrows=2, ncols=2)
    # ax = axes.flat
    # ax[0].imshow(dem_colmap, cmap=cmap, norm=normalizer)
    # ax[0].set_title('colmap')
    # ax[1].imshow(dem_odm, cmap=cmap, norm=normalizer)
    # ax[1].set_title('odm')
    # ax[2].imshow(dem_omvgs, cmap=cmap, norm=normalizer)
    # ax[2].set_title('omvgs')
    # ax[3].imshow(GT, cmap=cmap, norm=normalizer)
    # ax[3].set_title('ground truth')
    # fig.colorbar(im, ax=axes.ravel().tolist())
    # plt.show()

    # show diff heightmaps with common colorbar
    colmap_d = dem_diff(dem_colmap, gt)
    odm_d = dem_diff(dem_odm, gt)
    omvgs_d = dem_diff(dem_omvgs, gt)

    cmap = cm.get_cmap('CMRmap')
    normalizer = Normalize(np.ma.masked_array.min(np.ma.concatenate([colmap_d, odm_d, omvgs_d])),
                           np.ma.masked_array.max(np.ma.concatenate([colmap_d, odm_d, omvgs_d])))
    im = cm.ScalarMappable(norm=normalizer, cmap=cmap)

    # for i in [colmap_d, odm_d, omvgs_d]:
    #     plt.imshow(i, cmap=cmap, norm=normalizer)
    #     plt.colorbar(im, shrink=0.6, label='[m]')
    #     plt.show()

    fig, axes = plt.subplots(nrows=2, ncols=2)
    ax = axes.flat
    ax[0].imshow(colmap_d, cmap=cmap, norm=normalizer)
    ax[0].set_title('colmap')
    ax[1].imshow(odm_d, cmap=cmap, norm=normalizer)
    ax[1].set_title('odm')
    ax[2].imshow(omvgs_d, cmap=cmap, norm=normalizer)
    ax[2].set_title('omvgs')
    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.show()


if __name__ =='__main__':
    main()