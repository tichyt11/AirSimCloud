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
dist_recon_dir = 'C:/Users/tomtc/OneDrive/Desktop/MazeResults/NoWater/LargeSurvey/Distorted'

GTcloud = os.path.join(ideal_recon_dir, 'GT.ply')
colmap_cloud = os.path.join(ideal_recon_dir, 'Colmap_95min.ply')
odm_cloud = os.path.join(ideal_recon_dir, 'odm_55min.las')
omvgs_cloud = os.path.join(ideal_recon_dir, 'openmvgs_36min.las')

odm_cloud_dist = os.path.join(dist_recon_dir, 'odm_42mins.las')
omvgs_cloud_dist = os.path.join(dist_recon_dir, 'omvgs_29mins.las')

GTdem = os.path.join(eval, 'GTdem.tif')
colmap_dem_ideal = os.path.join(eval, 'ideal_colmap.tif')
odm_dem_ideal = os.path.join(eval, 'ideal_odm.tif')
omvgs_dem_ideal = os.path.join(eval, 'ideal_omvgs.tif')

odm_dem_dist = os.path.join(eval, 'dist_odm.tif')
omvgs_dem_dist = os.path.join(eval, 'dist_omvgs.tif')


def prepare_for_hist(dem, gt, absolute=False):
    nodata = 1000
    diff = dem_diff(dem, gt, absolute)
    diff = np.ma.masked_array(diff, mask=(dem == nodata) | (gt == nodata))
    diff = diff[~diff.mask].flatten()
    return diff


def dem_diff(dem, gt, absolute=False):
    nodata = 1000
    diff = gt - dem
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
    plt.xlabel('error [m]')
    plt.ylabel('percentage of tiles')
    plt.grid()
    plt.legend(prop={'size': 10})
    plt.show()
    return n, bins


def plot_hist_lines(n, bins, labels):
    for data in n:
        plt.plot(bins[0:-1], data)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.xlabel('error [m]')
    plt.ylabel('percentage of tiles')
    plt.legend(labels)
    plt.grid()
    plt.show()


def print_metrics(method, dem, gt):
    num_invalid = np.count_nonzero(dem.mask)
    total = dem.shape[0] * dem.shape[1] - np.count_nonzero(gt.mask)
    diff = dem_diff(dem, gt)
    diff = np.ma.abs(diff)
    std = np.std(diff.compressed())
    mean = np.mean(diff.compressed())
    perct = np.percentile(diff.compressed(), 90)
    med = np.percentile(diff.compressed(), 50)
    print(method)
    print('\tinvalid: ', num_invalid)
    print('\tinvalid_ratio: %.2f %%' % (100 * num_invalid / total))
    print('\tcompleteness: %.2f %%' % (100 - 100 * num_invalid / total))
    print('\tmean: %.2f standard deviation: %.2f'% (mean, std))
    print('\t90th percentile %.2f, median %.2f'% (perct, med))
    print('\tmin - max: %.2f - %.2f' % (np.min(diff), np.max(diff)))


def show_common(data, labels, cmap='CMRmap'):
    # show heightmaps with common colorbar
    cmap = cm.get_cmap(cmap)
    normalizer = Normalize(np.ma.masked_array.min(np.ma.concatenate([i for i in data])),
                           np.ma.masked_array.max(np.ma.concatenate([i for i in data])))
    im = cm.ScalarMappable(norm=normalizer, cmap=cmap)

    fig, axes = plt.subplots(nrows=len(data), ncols=1, sharex=True)
    ax = axes.flat
    for i, vals in enumerate(data):
        ax[i].imshow(vals, cmap=cmap, norm=normalizer)
        ax[i].set_title(labels[i])
        ax[i].set(ylabel='tile row')
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label='tile height [m]')
    plt.xlabel('tile column')
    plt.ylabel('tile row')
    plt.show()


def main():
    nodata = 1000
    grid_origin = np.array([-6, -80])  # bottom left corner of dsm world coordinates
    h_meters = 150
    w_meters = 350
    res = 0.3
    hist_range = (0, 1.5)
    cumulative = True
    grow = False
    absolute = True

    h, w = h_meters/res, w_meters/res
    print('height X width = %d X %d' % (h,w))

    # create
    handler = DemHandler(grid_origin, h, w, res, nodata)
    # handler.create_heightmap(GTcloud, GTdem)
    # handler.create_heightmap(colmap_cloud, colmap_dem_ideal)
    # handler.create_heightmap(odm_cloud, odm_dem_ideal)
    # handler.create_heightmap(omvgs_cloud, omvgs_dem_ideal)

    # handler.create_heightmap(odm_cloud_dist, odm_dem_dist)
    # handler.create_heightmap(omvgs_cloud_dist, omvgs_dem_dist)


    # load
    gt = handler.load_heightmap(GTdem)
    dem_colmap = handler.load_heightmap(colmap_dem_ideal)
    dem_odm = handler.load_heightmap(odm_dem_ideal)
    dem_omvgs = handler.load_heightmap(omvgs_dem_ideal)

    omvgs_dist = handler.load_heightmap(omvgs_dem_dist)
    odm_dist = handler.load_heightmap(odm_dem_dist)

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
    exclude_mask[348:402, 797:847] = 1  # mask out the hollow tree
    # exclude_mask[103:121, 247:264] = 1  # mask out the hollow tree


    # masks
    gt = np.ma.masked_array(gt, mask=invalid_gt)
    dem_colmap = np.ma.masked_array(dem_colmap, mask=invalid_gt | (dem_colmap == nodata) | exclude_mask)
    dem_odm = np.ma.masked_array(dem_odm, mask=invalid_gt | (dem_odm == nodata) | exclude_mask)
    dem_omvgs = np.ma.masked_array(dem_omvgs, mask=invalid_gt | (dem_omvgs == nodata) | exclude_mask)

    odm_dist = np.ma.masked_array(odm_dist, mask=invalid_gt | (odm_dist == nodata) | exclude_mask)
    omvgs_dist = np.ma.masked_array(omvgs_dist, mask=invalid_gt | (omvgs_dist == nodata) | exclude_mask)


    # metrics
    print_metrics('colmap', dem_colmap, gt)
    print_metrics('odm', dem_odm, gt)
    print_metrics('omvgs', dem_omvgs, gt)

    print_metrics('odm distorted', odm_dist, gt)
    print_metrics('omvgs distorted', omvgs_dist, gt)


    # mask and difference
    colmap_h = prepare_for_hist(dem_colmap, gt, absolute)
    odm_h = prepare_for_hist(dem_odm, gt, absolute)
    omvgs_h = prepare_for_hist(dem_omvgs, gt, absolute)

    dist_odm_h = prepare_for_hist(odm_dist, gt, absolute)
    dist_omvgs_h = prepare_for_hist(omvgs_dist, gt, absolute)


    # plot histograms
    labels = ['Colmap', 'ODM', 'oMVG + oMVS']
    n, bins = plot_hist([colmap_h, odm_h, omvgs_h], labels, cumulative, rng=hist_range)
    plot_hist_lines(n, bins, labels)

    labels = ['ODM', 'oMVG + oMVS']
    n, bins = plot_hist([dist_odm_h, dist_omvgs_h], labels, cumulative, rng=hist_range)
    plot_hist_lines(n, bins, labels)

    show_common([dem_colmap, dem_odm, dem_omvgs, gt], ['Colmap', 'ODM', 'oMVG + oMVS', 'Ground truth'])
    show_common([odm_dist, omvgs_dist, gt], ['ODM', 'oMVG + oMVS', 'Ground truth'])

    # show diff heightmaps with common colorbar
    colmap_d = dem_diff(dem_colmap, gt, absolute)
    odm_d = dem_diff(dem_odm, gt, absolute)
    omvgs_d = dem_diff(dem_omvgs, gt, absolute)

    odm_dist_d = dem_diff(odm_dist, gt, True)
    omvgs_dist_d = dem_diff(omvgs_dist, gt, True)

    show_common([colmap_d, odm_d, omvgs_d], ['Colmap', 'ODM', 'oMVG + oMVS'], cmap='plasma')
    show_common([odm_dist_d, omvgs_dist_d], ['ODM', 'oMVG + oMVS'], cmap='plasma')


if __name__ == '__main__':
    main()
