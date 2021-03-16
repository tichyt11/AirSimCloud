import matplotlib.pyplot as plt
import numpy as np
import os, time
from tools.dem_handling import DemHandler
import math
from matplotlib.ticker import PercentFormatter
import matplotlib.cm as cm
from matplotlib.colors import Normalize

eval = os.path.join(os.getcwd(), 'evaluation_data')
ideal_recon_dir = 'C:/Users/tomtc/OneDrive/Desktop/MazeResults/NoWater/LargeCV/Clouds/'

GTcloud = os.path.join(ideal_recon_dir, 'GT.ply')
colmap_cloud = os.path.join(ideal_recon_dir, 'Colmap_95min.ply')
odm_cloud = os.path.join(ideal_recon_dir, 'odm_55min.las')
omvgs_cloud = os.path.join(ideal_recon_dir, 'openmvgs_36min.las')

GTdem = os.path.join(eval, 'GTdem.tif')
colmap_dem_ideal = os.path.join(eval, 'ideal_colmap.tif')
odm_dem_ideal = os.path.join(eval, 'ideal_odm.tif')
omvgs_dem_ideal = os.path.join(eval, 'ideal_omvgs.tif')


def prepare_for_hist(dem, gt, absolute=False):
    nodata = 1000
    diff = dem - gt
    if absolute:
        diff = (np.abs(diff))
    diff = np.ma.masked_array(diff, mask=(dem == nodata) | (gt == nodata))
    diff = diff[~diff.mask].flatten()
    return diff


def dem_diff(dem, gt, absolute=False):
    nodata = 1000
    diff = dem - gt
    if absolute:
        diff = np.abs(diff)
    diff = np.ma.masked_array(diff, mask=(dem == nodata) | (gt == nodata))
    return diff


def plot_hist(data, labels, cumulative=False, rng=(-5, 5)):
    # histogram
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    n, bins, _ = plt.hist(data, bins=300, range=rng, cumulative=cumulative, histtype='step',
                          weights=[np.ones(len(i)) / len(i) for i in data], label=labels, lw=2)
    plt.xticks(np.linspace(rng[0], rng[1], 11))
    plt.grid()
    plt.legend(prop={'size': 10})
    plt.show()
    return n, bins


def plot_hist_lines(n, bins, labels):
    for data in n:
        plt.plot(bins[0:-1], data)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.legend(labels)
    plt.grid()
    plt.show()


def print_metrics(method, dem, gt):
    nodata = 1000
    num_invalid = np.count_nonzero(dem.mask)
    total = dem.shape[0] * dem.shape[1] - np.count_nonzero(gt.mask)
    diff = dem_diff(dem, gt)
    diff = np.ma.abs(diff)
    std = np.ma.masked_array.std(diff)
    mean = np.ma.masked_array.mean(diff)
    rng = np.max(diff) - np.min(diff)
    print(method)
    print('\ttotal: ', total)
    print('\tinvalid: ', num_invalid)
    print('\tmean: %.2f standard deviation: %.2f'% (mean, std))
    print('\tmin - max: %.2f - %.2f' % (np.min(diff), np.max(diff)))
    print('\tinvalid_ratio: %.2f %%' % (100 * num_invalid / total))
    print('\tcompleteness: %.2f %%' % (100 - 100 * num_invalid / total))


def main():
    nodata = 1000
    grid_origin = np.array([-16, -80])  # bottom left corner of dsm world coordinates
    h_meters = 150
    w_meters = 366
    res = 0.3
    hist_range = (-1.5, 1.5)
    cumulative = False
    grow = True
    absolute = False

    h, w = h_meters/res, w_meters/res

    handler = DemHandler(grid_origin, h, w, res, nodata)
    # handler.create_heightmap(GTcloud, GTdem)
    # handler.create_heightmap(colmap_cloud, colmap_dem_ideal)
    # handler.create_heightmap(odm_cloud, odm_dem_ideal)
    # handler.create_heightmap(omvgs_cloud, omvgs_dem_ideal)

    # load
    gt = handler.load_heightmap(GTdem)
    dem_colmap = handler.load_heightmap(colmap_dem_ideal)
    dem_odm = handler.load_heightmap(odm_dem_ideal)
    dem_omvgs = handler.load_heightmap(omvgs_dem_ideal)

    # grow
    if grow:
        grow_size = 3*math.ceil(math.sqrt(0.5)/res)
        tick = time.time()
        print('growing heightmaps with grow size %d' % grow_size)
        # gt = handler.grow_heightmap(gt, grow_size)
        dem_colmap = handler.grow_heightmap(dem_colmap, grow_size)
        dem_odm = handler.grow_heightmap(dem_odm, grow_size)
        dem_omvgs = handler.grow_heightmap(dem_omvgs, grow_size)
        print('growing done after %.2f seconds' % (time.time() - tick))

    invalid_gt = gt == nodata
    exclude_mask = np.zeros(gt.shape, dtype=np.uint8)
    exclude_mask[348:402, 830:880] = 1  # mask out the hollow tree
    # exclude_mask[103:121, 247:264] = 1  # mask out the hollow tree


    # masks
    gt = np.ma.masked_array(gt, mask=invalid_gt)
    dem_colmap = np.ma.masked_array(dem_colmap, mask=invalid_gt | (dem_colmap == nodata) | exclude_mask)
    dem_odm = np.ma.masked_array(dem_odm, mask=invalid_gt | (dem_odm == nodata) | exclude_mask)
    dem_omvgs = np.ma.masked_array(dem_omvgs, mask=invalid_gt | (dem_omvgs == nodata) | exclude_mask)

    # metrics
    print_metrics('colmap', dem_colmap, gt)
    print_metrics('odm', dem_odm, gt)
    print_metrics('omvgs', dem_omvgs, gt)

    # mask and difference
    colmap_h = prepare_for_hist(dem_colmap, gt, absolute)
    odm_h = prepare_for_hist(dem_odm, gt, absolute)
    omvgs_h = prepare_for_hist(dem_omvgs, gt, absolute)

    # plot histograms
    labels = ['Colmap', 'ODM', 'oMVG + oMVS']
    n, bins = plot_hist([colmap_h, odm_h, omvgs_h], labels, cumulative, rng=hist_range)
    plot_hist_lines(n, bins, labels)

    # show heightmaps with common colorbar
    cmap = cm.get_cmap('CMRmap')
    normalizer = Normalize(np.ma.masked_array.min(np.ma.concatenate([dem_omvgs, dem_odm, dem_colmap, gt])),
                           np.ma.masked_array.max(np.ma.concatenate([dem_omvgs, dem_odm, dem_colmap, gt])))
    im = cm.ScalarMappable(norm=normalizer, cmap=cmap)

    for i in [dem_colmap, dem_odm, dem_omvgs, gt]:
        plt.imshow(i, cmap=cmap, norm=normalizer)
        plt.colorbar(im, shrink=0.6, label='[m]')
        plt.show()

    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True)
    ax = axes.flat
    ax[0].imshow(dem_colmap, cmap=cmap, norm=normalizer)
    ax[0].set_title('Colmap')
    ax[1].imshow(dem_odm, cmap=cmap, norm=normalizer)
    ax[1].set_title('ODM')
    ax[2].imshow(dem_omvgs, cmap=cmap, norm=normalizer)
    ax[2].set_title('oMVG + oMVS')
    ax[3].imshow(gt, cmap=cmap, norm=normalizer)
    ax[3].set_title('Ground truth')
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label='[m]')
    plt.show()

    # show diff heightmaps with common colorbar
    colmap_d = dem_diff(dem_colmap, gt, False)
    odm_d = dem_diff(dem_odm, gt, False)
    omvgs_d = dem_diff(dem_omvgs, gt, False)

    cmap = cm.get_cmap('plasma')
    normalizer = Normalize(np.ma.masked_array.min(np.ma.concatenate([colmap_d, odm_d, omvgs_d])),
                           np.ma.masked_array.max(np.ma.concatenate([colmap_d, odm_d, omvgs_d])))
    im = cm.ScalarMappable(norm=normalizer, cmap=cmap)

    # for i in [colmap_d, odm_d, omvgs_d]:
    #     plt.imshow(i, cmap=cmap, norm=normalizer)
    #     plt.colorbar(im, shrink=0.6, label='[m]')
    #     plt.show()

    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    ax = axes.flat
    ax[0].imshow(colmap_d, cmap=cmap, norm=normalizer)
    ax[0].set_title('Colmap')
    ax[1].imshow(odm_d, cmap=cmap, norm=normalizer)
    ax[1].set_title('ODM')
    ax[2].imshow(omvgs_d, cmap=cmap, norm=normalizer)
    ax[2].set_title('oMVG + oMVS')
    fig.colorbar(im, ax=axes.ravel().tolist(), label='[m]')
    plt.show()


if __name__ == '__main__':
    main()
