import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Polygon
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.pylab as pylab
import matplotlib.patches as mpatches
from scipy.stats import skew
import scipy.stats as stats

def plot_raster_query(ax,spikes,nodes_df,cmap, plot_order, twindow=[0,3], marker=".", lw=0,s=10):
    '''
    Plot raster colored according to a query.
    Query's key defines node selection and the corresponding values defines color

    Parameters:
    -----------
        ax: matplotlib axes object
            axes to use
        spikes: tuple of numpy arrays
            includes [times, gids]
        nodes_df: pandas DataFrame
            nodes table
        cmap: dict
            key: query string, value:color
        twindow: tuple
            [start_time,end_time]

    '''
    tstart = twindow[0]
    tend = twindow[1]

    ix_t = np.where((spikes[0]>tstart) & (spikes[0]<tend))

    spike_times = spikes[0][ix_t]
    spike_gids = spikes[1][ix_t]

    counter = 0
    patch_colors = ['grey', 'w', 'grey', 'w', 'grey']
    for query in plot_order:
        # col = cmap[query]
        # query_df = nodes_df.query(query)
        # gids_query = query_df.index

        col = cmap[query]
        # query_df = nodes_df.pop_name.str.startswith(query)
        # gids_query = query_df.index
        gids_query = np.where(nodes_df.pop_name.str.startswith(query))[0]
        tuning_angles = nodes_df.tuning_angle[gids_query]

        print query, "ncells:", len(gids_query), col

        ix_g = np.in1d(spike_gids, gids_query)
        # gid_min  = gids_query.min()
        # num_gids = gids_query.max() - gid_min
        # spikes_times = spike_gids[ix_g] - gid_min + counter
        # counter = counter + num_gids + 1

        # ax.plot(spike_times[ix_g],spikes_times,
        #             marker= marker,
        #             color=col,
        #             label=query[query.index("=")+3:-1],
        #             lw=lw
        #             );

        spikes_gids_temp = spike_gids[ix_g]
        # print np.shape(spike_times[ix_g])
        # spikes_gids_temp = stats.rankdata(spikes_gids_temp, method='dense') - 1
        gids_temp = stats.rankdata(tuning_angles, method='dense') - 1
        gids_temp = gids_temp + counter
        for i, gid in enumerate(gids_query):
            inds = np.where(spikes_gids_temp == gid)
            spikes_gids_temp[inds] = gids_temp[i]

        counter += len(gids_query)


        ax.plot(spike_times[ix_g], spikes_gids_temp,
                    marker= marker,
                    color = col,
                    label=query,
                    lw=lw,
                    markersize = s
                    );

        if ('Htr3a' in query):
            # if ('1' not in query):
            #     plt.plot([0, twindow[1] + 500], [counter, counter], 'k', lw = 3)


            if 'xy' not in locals():
                xy = (0,0)
                w = twindow[1] + 500
                h = counter
                h_cumsum = h
            else:
                xy = (0, h_cumsum)
                h = counter - h_cumsum
                h_cumsum += h

            ax.add_patch(Rectangle(xy, w, h, color=patch_colors.pop(), alpha=0.2))




def plot_raster_unique_attributes(ax,spikes,nodes_df, attribute, twindow=[0,3], marker=".", lw=0,s=10):
    '''
    Plot raster colored according to a query.
    Query's key defines node selection and the corresponding values defines color

    Parameters:
    -----------
        ax: matplotlib axes object
            axes to use
        spikes: tuple of numpy arrays
            includes [times, gids]
        nodes_df: pandas DataFrame
            nodes table
        cmap: dict
            key: query string, value:color
        twindow: tuple
            [start_time,end_time]

    '''
    tstart = twindow[0]
    tend = twindow[1]

    ix_t = np.where((spikes[0]>tstart) & (spikes[0]<tend))

    spike_times = spikes[0][ix_t]
    spike_gids = spikes[1][ix_t]

    counter = 0
    for feature in nodes_df[attribute].unique():
        query_df = nodes_df.ix[np.where(nodes_df[attribute] == feature)]
        gids_query = query_df.index

        print feature,  "ncells:", len(gids_query)

        ix_g = np.in1d(spike_gids, gids_query)
        gid_min  = gids_query.min()
        num_gids = gids_query.max() - gid_min
        spike_gids[ix_g] = spike_gids[ix_g] - gid_min + counter
        counter = counter + num_gids + 1

        ax.plot(spike_times[ix_g],spike_gids[ix_g],
                    marker= marker,
                    label=feature,
                    lw=lw
                    );




def plot_sliding_window_rate_unique_attributes(ax, firingRates, nodes_df, attribute,  sliding_window_size = 100, twindow=[0,3]):
    '''
        Plot raster colored according to a query.
        Query's key defines node selection and the corresponding values defines color

        Parameters:
        -----------
            ax: matplotlib axes object
                axes to use
            spikes: tuple of numpy arrays
                includes [times, gids]
            nodes_df: pandas DataFrame
                nodes table
            cmap: dict
                key: query string, value:color
            twindow: tuple
                [start_time,end_time]

        '''
    tstart = twindow[0]
    tend = twindow[1]
    half_window = sliding_window_size / 2

    counter = 0
    for feature in nodes_df[attribute].unique():
        query_df = nodes_df.ix[np.where(nodes_df[attribute] == feature)]
        gids_query = query_df.index

        tempRates = firingRates[gids_query]
        meanRates = np.zeros(len(gids_query))
        for i in range(len(gids_query)):
            if half_window != 0:
                meanRates[i] = np.mean(tempRates[max(0, i - half_window):min(i + half_window, len(gids_query))])
            elif half_window == 0:
                meanRates[i] = tempRates[i]

        print feature, "ncells:", len(gids_query)

        # gid_min  = gids_query.min()
        num_gids = len(meanRates)  # gids_query.max() - gid_min
        nrn_ids = np.arange(counter, counter + num_gids)
        counter = counter + num_gids  # + 1

        ax.plot(nrn_ids, meanRates,
                label=feature
                );



def plot_rates_over_time(ax, spikes,nodes_df,cmap, plot_order, binsize = 10, twindow=[0,3]):

    '''
    plot firing rates over time for every population based on the quesry condition
    # binsize in ms
    '''
    time_bins = np.arange(twindow[0] - 0.5, twindow[1] + 0.5, binsize)
    time = np.linspace(twindow[0], twindow[1],  twindow[1]/binsize, endpoint= True)

    offset = 2.0

    for query in plot_order:
        col = cmap[query]
        query_df = nodes_df.query(query)
        gids_query = query_df.index

        firingRates = np.zeros([len(gids_query), len(time)])

        for i, gid in enumerate(gids_query):
            spks = spikes[0, np.where(spikes[1, :] == gid)[0]]
            firingRates[i, :] = np.histogram(spks, bins=time_bins)[0]

        firingRateOverTime = np.mean(firingRates, axis=0)/(binsize/1000.0) + offset
        offset = offset + 5.0
        ax.plot(time, firingRateOverTime,
                    color=col,
                    label=query[query.index("=")+3:-1], lw = 3
                    );

def calculate_rates_over_time(spikes,nodes_df, population, binsize = 50, twindow=[0,3]):

    '''
    Calulculate firing rates over time for every population based on the query condition
    # binsize in ms
    '''
    time_bins = np.arange(twindow[0]-0.5, twindow[1] + 0.5, binsize)
    time = np.linspace(twindow[0], twindow[1],  (twindow[1] - twindow[0])/binsize, endpoint= True)

    for pop in population:
        gids_query = np.where(nodes_df.pop_name.str.startswith(pop))[0]

        firingRates = np.zeros([len(gids_query), len(time)])

        for i, gid in enumerate(gids_query):
            spks = spikes[0, np.where(spikes[1, :] == gid)[0]]
            firingRates[i, :] = np.histogram(spks, bins=time_bins)[0]/(binsize/1000.0)

    return firingRates



def plot_sliding_window_rate(ax,firingRates,nodes_df,cmap, plot_order, sliding_window_size = 100, twindow=[0,3], return_rate = False, alpha = 1.0):
    '''
    Plot raster colored according to a query.
    Query's key defines node selection and the corresponding values defines color

    Parameters:
    -----------
        ax: matplotlib axes object
            axes to use
        spikes: tuple of numpy arrays
            includes [times, gids]
        nodes_df: pandas DataFrame
            nodes table
        cmap: dict
            key: query string, value:color
        twindow: tuple
            [start_time,end_time]

    '''
    tstart = twindow[0]
    tend = twindow[1]
    half_window = sliding_window_size / 2

    counter = 0
    for query in plot_order:
        col = cmap[query]
        query_df = nodes_df.query(query)
        gids_query = query_df.index

        tempRates = firingRates[gids_query]
        meanRates = np.zeros(len(gids_query))
        for i in range(len(gids_query)):
            if half_window != 0:
                meanRates[i] = np.mean(tempRates[max(0, i - half_window) :min(i + half_window, len(gids_query))])
            elif half_window == 0:
                meanRates[i] = tempRates[i]

        print query,  "ncells:", len(gids_query), col

        #gid_min  = gids_query.min()
        num_gids = len(meanRates)#gids_query.max() - gid_min
        nrn_ids = np.arange(counter, counter + num_gids)
        counter = counter + num_gids# + 1

        ax.plot(nrn_ids, meanRates,
                #nodes_df.ix[gids_query, 'tuning_angle'], meanRates,
                    color=col,
                    label=query[query.index("=")+3:-1], #, lw = 4
                    alpha = alpha
                    );
        print np.mean(tempRates)

        if return_rate:
            return np.mean(tempRates)


def plot_rates_query(ax,spikes,nodes_df,cmap,
				dT=0.01,
				twindow=[0,3],
				marker=".",
				lw=2,
				linestyle='-',
				s=10,
				offset=0
				):
    '''
    Plot raster colored according to a query.
    Query's key defines node selection and the corresponding values defines color

    Parameters:
    -----------
        ax: matplotlib axes object
            axes to use
        spikes: tuple of numpy arrays
            includes [times, gids]
        nodes_df: pandas DataFrame
            nodes table
        cmap: dict
            key: query string, value:color
        twindow: tuple
            [start_time,end_time]

    '''
    tstart = twindow[0]
    tend = twindow[1]

    ix_t = np.where((spikes[0]>tstart) & (spikes[0]<tend))

    spike_times = spikes[0][ix_t]
    spike_gids = spikes[1][ix_t]

    dT = dT;   tbins = np.arange(twindow[0],twindow[1]+dT,dT)
    rate_offset=0

    counter = 0
    for query,col in cmap.items():
        query_df = nodes_df.query(query)
        gids_query = query_df.index
        ncells = len(gids_query)
        print query,  "ncells:", len(gids_query), col

        ix_g = np.in1d(spike_gids, gids_query)

        hist,bins = np.histogram(spike_times[ix_g], bins=tbins)
        rate = hist/(ncells*dT)

        rate_offset+=offset
        ax.plot(tbins[0:-1]+dT/2,rate+rate_offset,
                    color=col,
                    linewidth=lw,
                    linestyle=linestyle,
                    label=query[query.index("=")+3:-1]);




def plot_OSI_query(ax,OSI_DSI_DF,nodes_df,cmap, plot_order):
    '''
    Plot OSI colored according to a query.
    Query's key defines node selection and the corresponding values defines color

    Parameters:
    -----------
        ax: matplotlib axes object
            axes to use
        OSI_DSI_DF: pandas DataFrame
            OSI and DSI values per cell
        nodes_df: pandas DataFrame
            nodes table
        cmap: dict
            key: query string, value:color
    '''


    boxes = []
    for i, query in enumerate(plot_order):
        col = cmap[query]
        query_df = nodes_df.query(query)
        gids_query = query_df.index

        print query
        print 'mean OSI: ', np.nanmean(OSI_DSI_DF.ix[gids_query, 'OSI'])
        print 'mean DSI: ', np.nanmean(OSI_DSI_DF.ix[gids_query, 'DSI'])

        data = np.array(OSI_DSI_DF.ix[gids_query, 'DSI'])
        data = data[~np.isnan(data)]
        boxes.append(data)


    ax.boxplot(boxes)
    # ax.boxplot(np.array(OSI_DSI_DF.ix[gids_query, 'OSI']),
    #            positions=np.array([float(i)]),
    #            patch_artist=True,
    #            boxprops=dict(facecolor=col, color=col))

def plot_metric_vs_theta(dfmetrics_list,  colors, labels, save = False, pop = 'e', exp_data = [], metric ='max_mean_rate(Hz)' ):

    # metric = 'Avg_Rate(Hz)'
    orientations = np.arange(0., 360., 45.)
    num_dfs = len(dfmetrics_list)
    color_patches = []
    fig, ax3000 = plt.subplots(figsize=(18, 18), num = 3000)
    fig, ax3001 = plt.subplots(figsize=(30, 20), num = 3001)
    fig3002, ax3002 = plt.subplots(figsize=(30, 20), num = 3002)
    compression_inds = [0, 1, 2, 1, 0, 1, 2, 1]
    compress_oris = np.array([0, 45, 90])
    for zz, dfmetrics in enumerate(dfmetrics_list):
        boxes = []
        rates = np.zeros(len(orientations))
        error_rates = np.zeros(len(orientations))
        compressed_all = [[], [], []]
        for i, theta in enumerate(orientations):
            gids = np.intersect1d(np.where(dfmetrics.pop_name.str.startswith(pop))[0], np.where(dfmetrics[metric] > 0.0))
            gids = np.intersect1d(np.where(dfmetrics.preferred_angle == theta), gids)

            rates[i]       = np.median((dfmetrics.ix[gids, metric]))
            error_rates[i] = stats.sem((dfmetrics.ix[gids, metric]))    # Take the log here for the error else taking log of fractions and the smaller the error the bigger the error bar!!!

            compressed_all[compression_inds[i]] = np.append(np.array(compressed_all[compression_inds[i]]), np.array(dfmetrics.ix[gids, metric]))

        #     boxes.append(list(dfmetrics.ix[gids, metric]))
        # pos = (1. + max(1, (num_dfs - 1))) * np.arange(len(orientations)) - (0.5 / num_dfs) * (num_dfs - 1) + zz * 0.5
        # bp = plt.boxplot(boxes, positions=pos)
        # plt.setp(bp['boxes'], color='black')
        # plt.setp(bp['whiskers'], color='black')
        # plt.setp(bp['fliers'], markeredgecolor='black', markerfacecolor=colors[zz], marker='.', markersize = 10.0)
        #
        #
        # numBoxes = np.shape(boxes)[0]
        # for i in range(numBoxes):
        #    box = bp['boxes'][i]
        #    boxX = []
        #    boxY = []
        #    for j in range(len(box.get_ydata())):
        #        boxX.append(box.get_xdata()[j])
        #        boxY.append(box.get_ydata()[j])
        #    boxCoords = list(zip(boxX, boxY))
        #    boxPolygon = Polygon(boxCoords, facecolor=colors[zz])
        #    ax.add_patch(boxPolygon)
        #
        # color_patches.append(mpatches.Patch(color=colors[zz], label=labels[zz]))
        # plt.legend(handles=color_patches)

        plt.figure(3002)
        pos_org = np.array([0.25, 2.25, 4.25])
        pos = pos_org + zz*1#(1. + max(1, (num_dfs - 1))) * np.arange(len(compress_oris)) - (0.5 / num_dfs) * (num_dfs - 1) + zz * 0.5
        bp = plt.boxplot(compressed_all/np.median(compressed_all[0]), positions=pos)
        plt.setp(bp['boxes'], color='black')
        plt.setp(bp['whiskers'], color='black')
        plt.setp(bp['fliers'], markeredgecolor='black', markerfacecolor=colors[zz], marker='.', markersize=10.0)

        numBoxes = np.shape(compressed_all)[0]
        for i in range(numBoxes):
            box = bp['boxes'][i]
            boxX = []
            boxY = []
            for j in range(len(box.get_ydata())):
                boxX.append(box.get_xdata()[j])
                boxY.append(box.get_ydata()[j])
            boxCoords = list(zip(boxX, boxY))
            boxPolygon = Polygon(boxCoords, facecolor=colors[zz])
            ax3002.add_patch(boxPolygon)

        color_patches.append(mpatches.Patch(color=colors[zz], label=labels[zz]))


        plt.figure(3000)
        plt.errorbar(orientations, (rates/rates[0]), yerr=error_rates, c = colors[zz], ecolor= colors[zz], lw = 10, elinewidth= 10, label = labels[zz])


        plt.figure(3001)
        rates_compressed = np.zeros(3)
        error_compressed = np.zeros(3)
        for j in range(3):
            rates_compressed[j] = np.median((compressed_all[j]))
            error_compressed[j] = stats.sem((compressed_all[j]))
        plt.errorbar(compress_oris, rates_compressed/rates_compressed[0] ,yerr=error_compressed, c = colors[zz], ecolor= colors[zz], lw = 3, elinewidth= 3, label = labels[zz])


    compressed_all = [[], [], []]
    if len(exp_data) > 0:
        rates = np.zeros(len(orientations))
        error_rates = np.zeros(len(orientations))
        for i, theta in enumerate(orientations):
            rates[i] = np.median((exp_data[i]))
            error_rates[i] = stats.sem((exp_data[i]))
            compressed_all[compression_inds[i]] = np.append(np.array(compressed_all[compression_inds[i]]), exp_data[i])
        for j in range(3):
            rates_compressed[j] = np.median((compressed_all[j]))
            error_compressed[j] = stats.sem((compressed_all[j]))
            print np.std(compressed_all[j]), len((compressed_all[j]))


        plt.figure(3000)
        plt.errorbar(orientations, (rates/rates[0]), yerr=(error_rates), c='grey', ecolor='grey', lw=10, elinewidth=10, label='Experiment')
        ax3000.set_xticks(orientations.astype(int))
        plt.xlabel('Theta (degrees)')
        plt.ylabel('[ ' + metric + ' ]')
        #plt.legend(loc = 'upper left')
        plt.xlim(-6, 340)
        plt.ylim(0.0, 1.94) #1.2
        plt.tight_layout()
        plt.savefig("/home/yazanb/Desktop/Final_V1_figs7fig8/Median_log_Rmax_vs_theta_phase_corrected_ALL_no_log.png")
        # plt.savefig('/home/yazanb/Desktop/OSI_DSI_boxplots/Median_log_Rmax_vs_theta.eps', format='eps', dpi=1000)
        # plt.savefig('/home/yazanb/Desktop/OSI_DSI_boxplots/Median_log_Rmax_vs_theta.svg', format='svg', dpi=1000)


        plt.figure(3001)
        plt.errorbar(compress_oris, rates_compressed/rates_compressed[0], yerr=error_compressed, c='grey', ecolor = 'grey', lw = 3, elinewidth = 3, label = 'Experiment')
        ax3001.set_xticks(compress_oris.astype(int))
        plt.xlabel('Theta (degrees)')
        plt.ylabel(metric)
        plt.legend(loc = 'lower right')
        plt.xlim(-2, 92)
        plt.tight_layout()

        plt.figure(3002)
        pos = pos_org + zz*0.5
        bp = plt.boxplot(compressed_all/rates_compressed[0], positions=pos)
        plt.setp(bp['boxes'], color='black')
        plt.setp(bp['whiskers'], color='black')
        plt.setp(bp['fliers'], markeredgecolor='black', markerfacecolor='grey', marker='.', markersize=10.0)

        numBoxes = np.shape(compressed_all)[0]
        for i in range(numBoxes):
            box = bp['boxes'][i]
            boxX = []
            boxY = []
            for j in range(len(box.get_ydata())):
                boxX.append(box.get_xdata()[j])
                boxY.append(box.get_ydata()[j])
            boxCoords = list(zip(boxX, boxY))
            boxPolygon = Polygon(boxCoords, facecolor='grey')
            ax3002.add_patch(boxPolygon)

        color_patches.append(mpatches.Patch(color='grey', label="Experiment"))
        plt.legend(handles=color_patches, loc = 'lower right')
        ax3002.set_xticklabels(['0', '45', '90'])
        ax3002.set_xlim(-1, 30)

        return compressed_all


        # plt.savefig('/home/yazanb/Desktop/OSI_DSI_boxplots/Rmax_Vs_Ori_' + pop)


def report_similarity_metrics(dfmetrics_bio, dfmetrics_glif, metric, exp_data):


    gids_bio  = np.where(dfmetrics_bio['pop_name'].str.startswith('e'))[0]
    gids_glif = np.where(dfmetrics_glif['pop_name'].str.startswith('e'))[0]

    gids_bio_e23  = np.where(dfmetrics_bio['pop_name'].str.startswith('e23'))[0]
    gids_glif_e23 = np.where(dfmetrics_glif['pop_name'].str.startswith('e23'))[0]
    gids_bio_e4  = np.where(dfmetrics_bio['pop_name'].str.startswith('e4'))[0]
    gids_glif_e4 = np.where(dfmetrics_glif['pop_name'].str.startswith('e4'))[0]
    gids_bio_e5  = np.where(dfmetrics_bio['pop_name'].str.startswith('e5'))[0]
    gids_glif_e5 = np.where(dfmetrics_glif['pop_name'].str.startswith('e5'))[0]
    gids_bio_e6  = np.where(dfmetrics_bio['pop_name'].str.startswith('e6'))[0]
    gids_glif_e6 = np.where(dfmetrics_glif['pop_name'].str.startswith('e6'))[0]
    print 'E_bio: ', len(gids_bio)
    print 'E_glif: ', len(gids_glif)


    bio_E  = np.array(dfmetrics_bio.ix[gids_bio, metric])
    glif_E = np.array(dfmetrics_glif.ix[gids_glif, metric])
    exp_data_E = np.concatenate((np.concatenate((exp_data[1],exp_data[3])), np.concatenate((exp_data[5],exp_data[7]))))

    bio_E23 = np.array(dfmetrics_bio.ix[gids_bio_e23, metric])
    glif_E23 = np.array(dfmetrics_glif.ix[gids_glif_e23, metric])
    exp_data_E23 = np.array(exp_data[1])

    bio_E4 = np.array(dfmetrics_bio.ix[gids_bio_e4, metric])
    glif_E4 = np.array(dfmetrics_glif.ix[gids_glif_e4, metric])
    exp_data_E4 = np.array(exp_data[3])

    bio_E5 = np.array(dfmetrics_bio.ix[gids_bio_e5, metric])
    glif_E5 = np.array(dfmetrics_glif.ix[gids_glif_e5, metric])
    exp_data_E5 = np.array(exp_data[5])

    bio_E6 = np.array(dfmetrics_bio.ix[gids_bio_e6, metric])
    glif_E6 = np.array(dfmetrics_glif.ix[gids_glif_e6, metric])
    exp_data_E6 = np.array(exp_data[7])

    # For Pvab populations
    gids_bio = np.intersect1d(np.where(dfmetrics_bio['pop_name'].str.contains('Pvalb')), np.where(dfmetrics_bio['pop_name'].str.startswith('i')))
    gids_glif = np.intersect1d(np.where(dfmetrics_glif['pop_name'].str.contains('Pvalb')), np.where(dfmetrics_bio['pop_name'].str.startswith('i')))

    gids_bio_i23  = np.where(dfmetrics_bio['pop_name'].str.startswith('i23P'))[0]
    gids_glif_i23 = np.where(dfmetrics_glif['pop_name'].str.startswith('i23P'))[0]
    gids_bio_i4  = np.where(dfmetrics_bio['pop_name'].str.startswith('i4P'))[0]
    gids_glif_i4 = np.where(dfmetrics_glif['pop_name'].str.startswith('i4P'))[0]
    gids_bio_i5  = np.where(dfmetrics_bio['pop_name'].str.startswith('i5P'))[0]
    gids_glif_i5 = np.where(dfmetrics_glif['pop_name'].str.startswith('i5P'))[0]
    gids_bio_i6  = np.where(dfmetrics_bio['pop_name'].str.startswith('i6P'))[0]
    gids_glif_i6 = np.where(dfmetrics_glif['pop_name'].str.startswith('i6P'))[0]


    print 'I_bio: ', len(gids_bio)
    print 'I_glif: ', len(gids_glif)

    bio_I  = np.array(dfmetrics_bio.ix[gids_bio, metric])
    glif_I = np.array(dfmetrics_glif.ix[gids_glif, metric])
    exp_data_I = np.concatenate((np.concatenate((exp_data[2],exp_data[4])), np.concatenate((exp_data[6],exp_data[8]))))

    bio_I23 = np.array(dfmetrics_bio.ix[gids_bio_i23, metric])
    glif_I23 = np.array(dfmetrics_glif.ix[gids_glif_i23, metric])
    exp_data_I23 = np.array(exp_data[2])

    bio_I4 = np.array(dfmetrics_bio.ix[gids_bio_i4, metric])
    glif_I4 = np.array(dfmetrics_glif.ix[gids_glif_i4, metric])
    exp_data_I4 = np.array(exp_data[4])

    bio_I5 = np.array(dfmetrics_bio.ix[gids_bio_i5, metric])
    glif_I5 = np.array(dfmetrics_glif.ix[gids_glif_i5, metric])
    exp_data_I5 = np.array(exp_data[6])

    bio_I6 = np.array(dfmetrics_bio.ix[gids_bio_i6, metric])
    glif_I6 = np.array(dfmetrics_glif.ix[gids_glif_i6, metric])
    exp_data_I6 = np.array(exp_data[8])

    print "For E_bio: " + metric
    print "median: ", np.nanmedian(bio_E)
    print "STD: ", np.nanstd(bio_E)

    print "\nFor E_glif: " + metric
    print "median: ", np.nanmedian(glif_E)
    print "STD: ", np.nanstd(glif_E)

    print "\nFor RS_exp: " + metric
    print "median: ", np.nanmedian(exp_data_E)
    print "STD: ", np.nanstd(exp_data_E)

    print "\nFor I_bio: " + metric
    print "median: ", np.nanmedian(bio_I)
    print "STD: ", np.nanstd(bio_I)

    print "\nFor I_glif: " + metric
    print "median: ", np.nanmedian(glif_I)
    print "STD: ", np.nanstd(glif_I)

    print "\nFor FS_exp: " + metric
    print "median: ", np.nanmedian(exp_data_I)
    print "STD: ", np.nanstd(exp_data_I)


    print "\n\nSIMILARITY DISTANCE: " + metric
    bio_E[np.isnan(bio_E)] = 0
    glif_E[np.isnan(glif_E)] = 0
    exp_data_E[np.isnan(exp_data_E)] = 0
    bio_I[np.isnan(bio_I)] = 0
    glif_I[np.isnan(glif_I)] = 0
    exp_data_I[np.isnan(exp_data_I)] = 0
    print "E-bio  Vs. E-exp: ",  1 - stats.ks_2samp(bio_E, exp_data_E)[0]
    print "E-glif Vs. E-exp: ",  1 - stats.ks_2samp(glif_E, exp_data_E)[0]
    print "E-bio  Vs. E-glif: ", 1 - stats.ks_2samp(bio_E, glif_E)[0]
    print "I-bio  Vs. I-exp: ",  1 - stats.ks_2samp(bio_I, exp_data_I)[0]
    print "I-glif Vs. I-exp: ",  1 - stats.ks_2samp(glif_I, exp_data_I)[0]
    print "I-bio  Vs. I-glif: ", 1 - stats.ks_2samp(bio_I, glif_I)[0]


    # print "\n\nP-VALUES L2/3: " + metric
    # bio_E23[np.isnan(bio_E23)] = 0
    # glif_E23[np.isnan(glif_E23)] = 0
    # exp_data_E23[np.isnan(exp_data_E23)] = 0
    # bio_I23[np.isnan(bio_I23)] = 0
    # glif_I23[np.isnan(glif_I23)] = 0
    # exp_data_I23[np.isnan(exp_data_I23)] = 0
    # print "E-bio  Vs. E-exp: ", stats.ks_2samp(bio_E23, exp_data_E23)[1]
    # print "E-glif Vs. E-exp: ", stats.ks_2samp(glif_E23, exp_data_E23)[1]
    # print "E-bio  Vs. E-glif: ", stats.ks_2samp(bio_E23, glif_E23)[1]
    # print "I-bio  Vs. I-exp: ", stats.ks_2samp(bio_I23, exp_data_I23)[1]
    # print "I-glif Vs. I-exp: ", stats.ks_2samp(glif_I23, exp_data_I23)[1]
    # print "I-bio  Vs. I-glif: ", stats.ks_2samp(bio_I23, glif_I23)[1]
    #
    #
    # print "\n\nP-VALUES L4: " + metric
    # bio_E4[np.isnan(bio_E4)] = 0
    # glif_E4[np.isnan(glif_E4)] = 0
    # exp_data_E4[np.isnan(exp_data_E4)] = 0
    # bio_I4[np.isnan(bio_I4)] = 0
    # glif_I4[np.isnan(glif_I4)] = 0
    # exp_data_I4[np.isnan(exp_data_I4)] = 0
    # print "E-bio  Vs. E-exp: ", stats.ks_2samp(bio_E4, exp_data_E4)[1]
    # print "E-glif Vs. E-exp: ", stats.ks_2samp(glif_E4, exp_data_E4)[1]
    # print "E-bio  Vs. E-glif: ", stats.ks_2samp(bio_E4, glif_E4)[1]
    # print "I-bio  Vs. I-exp: ", stats.ks_2samp(bio_I4, exp_data_I4)[1]
    # print "I-glif Vs. I-exp: ", stats.ks_2samp(glif_I4, exp_data_I4)[1]
    # print "I-bio  Vs. I-glif: ", stats.ks_2samp(bio_I4, glif_I4)[1]
    #
    # print "\n\nP-VALUES L5: " + metric
    # bio_E5[np.isnan(bio_E5)] = 0
    # glif_E5[np.isnan(glif_E5)] = 0
    # exp_data_E5[np.isnan(exp_data_E5)] = 0
    # bio_I5[np.isnan(bio_I5)] = 0
    # glif_I5[np.isnan(glif_I5)] = 0
    # exp_data_I5[np.isnan(exp_data_I5)] = 0
    # print "E-bio  Vs. E-exp: ", stats.ks_2samp(bio_E5, exp_data_E5)[1]
    # print "E-glif Vs. E-exp: ", stats.ks_2samp(glif_E5, exp_data_E5)[1]
    # print "E-bio  Vs. E-glif: ", stats.ks_2samp(bio_E5, glif_E5)[1]
    # print "I-bio  Vs. I-exp: ", stats.ks_2samp(bio_I5, exp_data_I5)[1]
    # print "I-glif Vs. I-exp: ", stats.ks_2samp(glif_I5, exp_data_I5)[1]
    # print "I-bio  Vs. I-glif: ", stats.ks_2samp(bio_I5, glif_I5)[1]
    #
    # print "\n\nP-VALUES L6: " + metric
    # bio_E6[np.isnan(bio_E6)] = 0
    # glif_E6[np.isnan(glif_E6)] = 0
    # exp_data_E6[np.isnan(exp_data_E6)] = 0
    # bio_I6[np.isnan(bio_I6)] = 0
    # glif_I6[np.isnan(glif_I6)] = 0
    # exp_data_I6[np.isnan(exp_data_I6)] = 0
    # print "E-bio  Vs. E-exp: ", stats.ks_2samp(bio_E6, exp_data_E6)[1]
    # print "E-glif Vs. E-exp: ", stats.ks_2samp(glif_E6, exp_data_E6)[1]
    # print "E-bio  Vs. E-glif: ", stats.ks_2samp(bio_E6, glif_E6)[1]
    # print "I-bio  Vs. I-exp: ", stats.ks_2samp(bio_I6, exp_data_I6)[1]
    # print "I-glif Vs. I-exp: ", stats.ks_2samp(glif_I6, exp_data_I6)[1]
    # print "I-bio  Vs. I-glif: ", stats.ks_2samp(bio_I6, glif_I6)[1]

    return (bio_E, exp_data_E, bio_I, exp_data_I)


    # plt.figure()
    # plt.plot(np.cumsum(bio_E)/np.max(np.cumsum(bio_E)))
    # plt.plot(np.cumsum(glif_E)/np.max(np.cumsum(glif_E)))
    # plt.plot(np.cumsum(exp_data_E)/np.max(np.cumsum(exp_data_E)))
    # plt.legend()
    # plt.title(metric)


def similarity_metric(model_data, exp_data, cellType = 'E'):
    exp_data_E  = np.concatenate((np.concatenate((exp_data[1], exp_data[3])), np.concatenate((exp_data[5], exp_data[7]))))
    exp_data_PV = np.concatenate((np.concatenate((exp_data[2],exp_data[4])), np.concatenate((exp_data[6],exp_data[8]))))

    # Remove nan because of errors
    exp_data_E[np.isnan(exp_data_E)] = 0
    exp_data_PV[np.isnan(exp_data_PV)] = 0
    model_data[np.isnan(model_data)] = 0

    if cellType == 'E':
        return 1 - stats.ks_2samp(model_data, exp_data_E)[0]
    else:
        return 1 - stats.ks_2samp(model_data, exp_data_PV)[0]


def plot_metric_boxplot(dfmetrics_list, populations, metric, colors, labels, save = False, log_dist = False, exp_data = []):

    color_patches = []
    num_dfs = len(dfmetrics_list)
    exp_counter = 0
    if not log_dist:                ## Plot regular boxplots
        fig, ax = plt.subplots(figsize=(30, 20))

        for zz, dfmetrics in enumerate(dfmetrics_list):
            boxes = []
            numNrns = []
            max_vals = []
            exp_boxes = []
            exp_numNrns = []
            for i, pop in enumerate(populations):
                if pop.startswith('L') and zz == len(dfmetrics_list) - 1:
                    data = np.array(exp_data[exp_counter])
                    data = data[~pd.isnull(data)]
                    exp_boxes.append(list(data))
                    numNrns.append(str(len(data)))
                    exp_counter += 1
                elif pop.startswith('L') and zz < 1:
                    continue
                else:

                    if metric == "OSI" or metric == "DSI":
                        gids = np.intersect1d(np.where(dfmetrics['pop_name'].str.startswith(pop)), np.where(dfmetrics['max_mean_rate(Hz)'] > 0.5))
                    else:
                        gids = np.where(dfmetrics['pop_name'].str.startswith(pop))[0]


                    data = np.array(dfmetrics.ix[gids, metric])
                    data = data[~pd.isnull(data)]
                    boxes.append(list(data))
                    numNrns.append(str(len(data)))
                    # max_vals.append(np.max(data))


            # pos = (1. + max(1,(num_dfs - 1)))*np.arange(len(populations))- (0.5/num_dfs)*(num_dfs - 1) + zz*0.5

            if len(exp_data) == 0:
                if zz == 0:
                    pos = np.array([0.25,   2.0,   3.25,   4.25,   5.25,  7.0, 8.0, 9.0,  10.0,  11.25,
                           12.25,  13.25,  15.,  16.,  17.25,  18.25,  19.25,  21.,  22.25,  23.25, 24.25])

                elif zz == 1:
                    pos += 0.5

            else:
                if zz == 0:

                    # Use this if splitting to all cre-lines (21 pops)
                    pos = np.array([0.25, 3.0, 5.25, 7.5, 8.5, 10.75, 11.75, 12.75, 13.75, 16.,
                                    18.25, 19.25, 21.5, 22.5, 24.75, 27., 28., 30.25, 32.5, 34.75, 35.75])

                    # Use this if merging excitatory cre-lines and experimental
                    if len(populations) == 26:
                        pos = np.array([0.25, 3.0, 5.25, 7.5, 8.5, 10.75, 13.0,
                                    15.25, 16.25, 18.5, 20.75, 23., 24., 26.25, 28.5, 30.75, 31.75])


                    # Only want to plot against one model
                    if len(dfmetrics_list) == 1:
                        pos = np.array([0.25,  # i1
                                        1.75,  # e23
                                        3.0,  # i23PV
                                        4.25, 4.75,  # i23 S/H
                                        6.25, 6.75, 7.25, 7.75,  # e4
                                        9.0,  # i4PV
                                        10.25, 10.75,  # i4 S/H
                                        12.25, 12.75,  # e5
                                        14.,  # i5PV
                                        15.25, 15.75,  # i5 S/H
                                        17.25,  # e6
                                        18.5,  # i6PV
                                        19.75, 20.25  # i6 S/H
                                        ])
                        pos_exp = [0.75, 2.25, 3.5, 8.25, 9.5, 13.25, 14.5, 17.75, 19.]
                        pos_all = np.copy(pos_exp) + 0.25
                        pos_all = np.append(pos_all, pos+0.25)
                        pos_all = np.sort(np.array(pos_all))


                    # For E4 only populations plus experimental
                    if len (populations) == 4:
                        print "5 populations given, this is a unqiue case here"
                        # pos = np.array([0.25, 1.25, 2.25, 3.25, 6.25])
                        pos = np.array([0.7, 1.6])


                elif zz == 1:
                    pos += 0.5
                    # pos_exp = [1.25, 4.0, 8.25, 14.0, 18.25, 22, 26.25, 29, 33.25]
                    pos_exp = [1.25, 4.0, 6.25, 14.75, 17., 23.5, 25.75, 31.25, 33.5]

                    # For merging excitatory cre-lines and experimental
                    if len(populations) == 26:
                        pos_exp = [1.25, 4.0, 6.25, 11.75, 14., 19.5, 21.75, 27.25, 29.5]
                        xy = (-.5, -.1)
                        w = 2.5
                        h = 1.5
                        if metric == "max_mean_rate(Hz)":
                            h = 75
                        ax.add_patch(Rectangle(xy, w, h, color='grey', alpha=0.2))

                        xy = (10., -.1)
                        w = 7.5
                        ax.add_patch(Rectangle(xy, w, h, color='grey', alpha=0.2))

                        xy = (25.5, -.1)
                        ax.add_patch(Rectangle(xy, w, h, color='grey', alpha=0.2))

                    # For E4 only populations plus experimental
                    if len (populations) == 4:
                        print "5 populations given, this is a unqiue case here"
                        # pos_exp = np.array([4.25, 7.25])
                        pos -= 0.3
                        pos_exp = np.array([0.5, 1.4])


                    pos_all = np.copy(pos_exp) + 0.25
                    pos_all = np.append(pos_all+0.05, pos-0.05)
                    pos_all = np.sort(np.array(pos_all))

                    # pos_all = np.array([0.75, $$, 3.5, $$, 5.75, 6.25, 7.25, 10.0, 11.0, 12.0, 13.0, 15.25,
                    #                 16.25, 17.25, 20., 21., 23.25, 24.25, 25.25, 28., 30.25, 31.25, 32.25])

            # np.array([0.25, 1.25, 3.0, 4.0, 4.25, 5.25, 6.25, 7.25, 9.0, 10.0, 11.0, 12.0, 13.0, 14.25,
            #           15.25, 16.25, 17.25, 19., 20., 21, 22.25, 23.25, 24.25, 25.25, 27., 28., 29.25, 30.25, 31.25,
            #           32.25])

            bp = plt.boxplot(boxes, positions=pos)
            plt.setp(bp['boxes'], color='black')
            plt.setp(bp['whiskers'], color='black')
            plt.setp(bp['fliers'], markeredgecolor='black', markerfacecolor=colors[zz], marker='.', markersize = 10.0)
            ax.set_xticklabels(populations)
            # plt.setp(bp['medians'], color='yellow')

            if len(exp_data) > 0 and zz == len(dfmetrics_list) - 1:
                bp2 = plt.boxplot(exp_boxes, positions=pos_exp)
                plt.setp(bp2['boxes'], color='black')
                plt.setp(bp2['whiskers'], color='black', linewidth = 3)
                plt.setp(bp2['caps'], color='black', linewidth=3)
                plt.setp(bp2['fliers'], markeredgecolor='black', markerfacecolor='gray', marker='.', markersize=12.0)
                # plt.setp(bp2['medians'], color='yellow')
                pops_temp = populations_with_exp = ['i1Htr3a', 'L1',
        'E2/3', 'L23RS', 'i23Pvalb', 'L23FS', 'i23Sst','i23Htr3a',
        'E4', 'L4RS', 'i4Pvalb', 'L4FS', 'i4Sst', 'i4Htr3a',
        'E5', 'L5RS', 'i5Pvalb', 'L5FS', 'i5Sst', 'i5Htr3a',
        'E6', 'L6RS', 'i6Pvalb',  'L6FS', 'i6Sst', 'i6Htr3a']
                plt.xticks(pos_all, pops_temp)

            plt.xticks(rotation=75)
            # ax.yaxis.grid(True, linestyle='-', which='major', color='grey', alpha=1.0)
            plt.ylabel(metric)
            if metric == "max_mean_rate(Hz)":
                plt.ylabel("Rate at preferred direction (Hz)")


            numBoxes = np.shape(boxes)[0]
            for i in range(numBoxes):
                box = bp['boxes'][i]
                boxX = []
                boxY = []
                for j in range(len(box.get_ydata())):
                    boxX.append(box.get_xdata()[j])
                    boxY.append(box.get_ydata()[j])
                boxCoords = list(zip(boxX, boxY))
                boxPolygon = Polygon(boxCoords, facecolor=colors[zz])
                ax.add_patch(boxPolygon)

            color_patches.append(mpatches.Patch(color= colors[zz], label=labels[zz]))
            # if metric != "max_mean_rate(Hz)":
            #     plt.legend(handles=color_patches, loc=2)

            if len(exp_data) > 0 and zz == len(dfmetrics_list) - 1:
                numBoxes = np.shape(exp_boxes)[0]
                for i in range(numBoxes):
                    box = bp2['boxes'][i]
                    boxX = []
                    boxY = []
                    for j in range(len(box.get_ydata())):
                        boxX.append(box.get_xdata()[j])
                        boxY.append(box.get_ydata()[j])
                    boxCoords = list(zip(boxX, boxY))
                    boxPolygon = Polygon(boxCoords, facecolor='gray')
                    ax.add_patch(boxPolygon)

                color_patches.append(mpatches.Patch(color= 'gray', label='Experiment'))
                # if metric != "max_mean_rate(Hz)":
                #     plt.legend(handles=color_patches, loc = 2)



            if len(exp_data) > 0 and zz == 1:
                for tick, label in enumerate(ax.get_xticklabels()):
                    # if metric == "OSI" or metric == "DSI":
                    #     ax.text(pos_all[tick], 1.1, numNrns[tick], horizontalalignment='center', size='large', color='k', weight='semibold', rotation = 90)
                    if metric == "max_mean_rate(Hz)":
                        ax.text(pos_all[tick], 170, numNrns[tick], horizontalalignment='center', size='large', color='k', weight='semibold', rotation = 90)
                        plt.ylim((0, 50))
            else:
                for tick, label in enumerate(ax.get_xticklabels()):
                    # if metric == "OSI" or metric == "DSI":
                    #     ax.text(pos[tick], 1.1, numNrns[tick], horizontalalignment='center', size='large', color='k', weight='semibold', rotation = 90)
                    if metric == "max_mean_rate(Hz)":
                        ax.text(pos[tick], 170, numNrns[tick], horizontalalignment='center', size='large', color='k', weight='semibold', rotation = 90)
                        plt.ylim((0, 50))




        plt.xlim([-0.3, np.max(pos) + 2.0])
        if metric == "OSI" or metric == "DSI":
            plt.ylim([-0.005, 1.05])
        plt.setp(ax.xaxis.get_majorticklabels(), ha='right')
        plt.tight_layout()

    else:
        for zz, dfmetrics in enumerate(dfmetrics_list):
            for i, pop in enumerate(populations):

                # if metric == "OSI" or metric == "DSI":
                #     gids = np.intersect1d(np.where(dfmetrics['pop_name'] == pop), np.where(dfmetrics['Avg_rate(Hz)'] > 0.4))
                # else:
                #     gids = np.where(dfmetrics['pop_name'] == pop)[0]
                gids = np.where(dfmetrics['pop_name'] == pop)[0]


                data = np.array(dfmetrics.ix[gids, metric])
                data = data[~pd.isnull(data)]
                if data.max() == 0.0:
                    continue
                fig, ax = plt.subplots(figsize=(30, 20))
                plt.hist(data, bins=np.logspace(np.log10(0.01),np.log10(100.0), 100))
                plt.xscale('log')
                plt.title(pop + ' - Skewness: {:0.2f}'.format(skew(data)))
                plt.ylabel('Occurences')
                plt.xlabel('Rate (Hz)')

    if save:
        # temp = "Ori_Dir_e4Scnn1a_i4PV"
        plt.savefig("/home/yazanb/Desktop/Final_V1_figs7fig8/boxplot_" + metric + "_asymmetric_only_model_forSI.png")
        # plt.savefig('/home/yazanb/Desktop/OSI_DSI_boxplots/boxplot_' + metric + '_dir_only_forSI.eps', format='eps', dpi=1000)
        # plt.savefig('/home/yazanb/Desktop/OSI_DSI_boxplots/boxplot_' + metric + '_dir_only_forSI.svg', format='svg', dpi=1000)


# def plot_metric_boxplot(dfmetrics, populations, metric, dfmetrics2, twoDFs=False, save=False):
#     boxes = []
#     for i, pop in enumerate(populations):
#         gids = np.intersect1d(np.where(dfmetrics['pop_name'] == pop), np.where(dfmetrics['Avg_rate(Hz)'] > 0.4))
#         data = np.array(dfmetrics.ix[gids, metric])
#         data = data[~pd.isnull(data)]
#         boxes.append(list(data))
#
#     fig, ax = plt.subplots(figsize=(30, 20))
#     bp = plt.boxplot(boxes, positions=1.5 * np.arange(len(populations)) - 0.3)
#     plt.setp(bp['boxes'], color='black')
#     plt.setp(bp['whiskers'], color='black')
#     plt.setp(bp['fliers'], color='red', marker='+')
#
#     ax.set_xticklabels(populations)
#     plt.xticks(rotation=40)
#     ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=1.0)
#     plt.ylabel(metric)
#
#     boxColor = 'royalblue'
#     numBoxes = np.shape(boxes)[0]
#     for i in range(numBoxes):
#         box = bp['boxes'][i]
#         boxX = []
#         boxY = []
#         for j in range(len(box.get_ydata())):
#             boxX.append(box.get_xdata()[j])
#             boxY.append(box.get_ydata()[j])
#         boxCoords = list(zip(boxX, boxY))
#         boxPolygon = Polygon(boxCoords, facecolor=boxColor)
#         ax.add_patch(boxPolygon)
#
#     if twoDFs:
#         boxes = []
#         for i, pop in enumerate(populations):
#             gids = np.intersect1d(np.where(dfmetrics2['pop_name'] == pop), np.where(dfmetrics2['Avg_rate(Hz)'] > 0.4))
#             data = np.array(dfmetrics2.ix[gids, metric])
#             data = data[~np.isnan(data)]
#             boxes.append(data)
#
#         bp = plt.boxplot(boxes, positions=1.5 * np.arange(len(populations)) + 0.3)
#         plt.setp(bp['boxes'], color='black')
#         plt.setp(bp['whiskers'], color='black')
#         plt.setp(bp['fliers'], color='red', marker='+')
#
#         ax.set_xticklabels(populations)
#         plt.xticks(rotation=40)
#         ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=1.0)
#         plt.ylabel(metric)
#
#         boxColor = 'orange'
#         numBoxes = np.shape(boxes)[0]
#         for i in range(numBoxes):
#             box = bp['boxes'][i]
#             boxX = []
#             boxY = []
#             for j in range(len(box.get_ydata())):
#                 boxX.append(box.get_xdata()[j])
#                 boxY.append(box.get_ydata()[j])
#             boxCoords = list(zip(boxX, boxY))
#             boxPolygon = Polygon(boxCoords, facecolor=boxColor)
#             ax.add_patch(boxPolygon)
#
#     blue_patch = mpatches.Patch(color='royalblue', label='Biophysical')
#     orange_patch = mpatches.Patch(color='orange', label='GLIF')
#     plt.legend(handles=[blue_patch, orange_patch])
#
#     plt.xlim([-1, 1.5 * len(populations) + 4])
#     if metric == "OSI" or metric == "DSI":
#         plt.ylim([0, 1])
#     plt.setp(ax.xaxis.get_majorticklabels(), ha='right')
#     # plt.tight_layout()
#     if save:
#         plt.savefig("/home/yazanb/Desktop/boxplot_" + metric + ".png")


def plot_tuning_curves (dfmetrics, dfrates, pop, metric, metric_threshold = 0.3, rate_threshold = 3.0, gids = None, linestyle = 'solid', marker = 'o',
                        plot_polar = True, label = ' ', second_metric = [], second_rates = [], alpha = 1.0, new_plot = True, color = 'royalblue'):

    if gids == None:
        gids = np.intersect1d(np.where(dfmetrics['pop_name'] == pop), np.where(dfmetrics[metric] > metric_threshold))
        gids = np.intersect1d(gids, np.where(dfmetrics['max_mean_rate(Hz)'] > rate_threshold))
        gids = np.intersect1d(gids, np.where(dfmetrics['preferred_angle'] == 90.0))

    if plot_polar:
        numDirections = 8
        theta = np.linspace(0.0, 2 * np.pi, numDirections, endpoint=False) - np.pi / 16
        width = np.pi / numDirections
        for i in gids[:10]:
            rates = np.array(dfrates.ix[dfrates['node_id'] == i, 'Avg_rate(Hz)'])
            plt.figure(figsize=(12,12))
            plt.subplot(1, 1, 1)
            ax = plt.subplot(111, projection='polar')
            ax.bar(theta, rates, width=width, bottom=0.0)
            plt.title(metric + ':%.3f \n Cell:%d \n ' % (dfmetrics.ix[i, metric], i))
            # plt.savefig('DSI_%.3f_cell_%d.eps'%(dfmetrics.ix[i, metric], i), format = 'eps', dpi = 1000)
            # plt.savefig('DSI_%.3f_cell_%d.png'%(dfmetrics.ix[i, metric], i))
    else:
        theta = np.arange(0, 360, 45)
        trials = 10
        for i in gids[:10]:
            rates = np.array(dfrates.ix[dfrates['node_id'] == i, 'Avg_rate(Hz)'])
            SD_rates = np.array(dfrates.ix[dfrates['node_id'] == i, 'SD_rate(Hz)']) /np.sqrt(trials)  # Convert to SEM by dividing by sqrt(trials)
            plt.figure(figsize=(21,18), num=i)
            # plt.errorbar(theta, rates, yerr=SD_rates, lw = '2', linestyle = linestyle, marker = marker, markersize = 12, c = 'royalblue', label = 'DSI:%.3f, OSI:%.3f'%(dfmetrics.ix[i, 'DSI'], dfmetrics.ix[i, 'OSI']))
            # plt.title('Cell:%d \n ' % (i))
            plt.errorbar(theta, rates, yerr=SD_rates, lw = '13', linestyle = linestyle, marker = marker, markersize = 10, c = color, label = label)
            plt.title(dfmetrics.pop_name[i])
            plt.xlabel('Drifting Grating Angle (degrees)')
            plt.ylabel ("Firing Rate (Hz)")
            plt.xticks(theta)
            plt.xlim(-3, 335)
            # plt.ylim(2, np.max(rates) + 3.5)
            # plt.grid()
            if len(second_metric) != 0:
                rates = np.array(second_rates.ix[second_rates['node_id'] == i, 'Avg_rate(Hz)'])
                SD_rates = np.array(second_rates.ix[second_rates['node_id'] == i, 'SD_rate(Hz)'])
                plt.errorbar(theta, rates, yerr=SD_rates, lw='2', c = 'orange', alpha = alpha, label = 'DSI:%.3f, OSI:%.3f'%(second_metric.ix[i, 'DSI'], second_metric.ix[i, 'OSI']))

            # plt.savefig(label + metric + '_%.3f_cell_%d.eps'%(dfmetrics.ix[i, metric], i), format = 'eps', dpi = 1000)
            # plt.savefig('/home/yazanb/Desktop/OSI_DSI_boxplots/LGN_only_' + dfmetrics.pop_name[i] + '_' +  metric + '_%.3f_cell_%d.png'%(dfmetrics.ix[i, metric], i))

    return gids[:10]
    plt.show()





def plot_peakFR_vs_preferred_angle (dfmetrics, dfrates, pop):#, metric, metric_threshold = 0.3, rate_threshold = 3.0, gids = None, plot_polar = True, label = ' ', second_metric = [], second_rates = []):

    boxes = []
    angles = np.arange(0, 360, 45.)
    fig, ax = plt.subplots(figsize=(30, 20))
    for ori in angles:
        gids = np.where(dfmetrics['pop_name'] == pop)[0]
        if ori == 0:
            gids1 = np.intersect1d(gids, np.where(dfmetrics['tuning_angle'] > 360 - 20.0))
            gids2 = np.intersect1d(gids, np.where(dfmetrics['tuning_angle'] < ori + 20.0))
            gids = np.concatenate((gids1, gids2))
        else:
            gids = np.intersect1d(gids, np.where(dfmetrics['tuning_angle'] > ori - 20.0))
            gids = np.intersect1d(gids, np.where(dfmetrics['tuning_angle'] < ori + 20.0))

        rates_all = np.zeros(len(gids))
        for zz, id in enumerate(gids):
            dfrates_inds = np.where(dfrates.node_id == id)[0]
            rates_id = np.array(dfrates['Avg_rate(Hz)'][dfrates_inds])
            rates_all[zz] = rates_id[int(ori/45)]

        rates_all= rates_all[~pd.isnull(rates_all)]
        boxes.append(list(rates_all))

    bp = plt.boxplot(boxes)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], markeredgecolor='black', marker='.', markersize=10.0)

    ax.set_xticklabels(angles)
    plt.xticks(rotation=40)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=1.0)
    plt.ylabel("Peak FR")
    plt.title(pop)

def plot_numCells_preferring_oris(dfmetrics, pop):#, metric, metric_threshold = 0.3, rate_threshold = 3.0, gids = None, plot_polar = True, label = ' ', second_metric = [], second_rates = []):

    gids = np.where(dfmetrics['pop_name'] == pop)[0]
    preferred_angles = np.array(dfmetrics.preferred_angle)[gids]
    angles = np.arange(0, 360, 45.)
    numCellsperAngle = np.zeros(len(angles))

    for zz, ori in enumerate(angles):
        numCellsperAngle[zz] = len(np.where(preferred_angles == ori)[0])

    plt.plot(angles, numCellsperAngle, label = pop, lw='3')
    plt.plot(angles, numCellsperAngle, 'o')
    plt.xticks(angles)
    plt.xlabel("Orientation")
    plt.ylabel("Number of neurons")



def create_maximal_grating_response_df(dfmetrics_list):

    numNrns = dfmetrics_list[0].shape[0]
    maximal_metric = pd.DataFrame(index=range(numNrns), columns=['node_id', 'DSI', 'OSI',
                                                                 'preferred_angle', 'depth', 'ei', 'pop_name',
                                                                 'location', 'Avg_rate(Hz)'])

    for gid in range(numNrns):
        rate_list = []
        if gid % 10000 == 0:
            print 'gid:', gid
        for i in range(len(dfmetrics_list)):
            rate_list.append(dfmetrics_list[i]['Avg_rate(Hz)'][gid])

        max_rate_idx = np.argmax(rate_list)
        maximal_metric.iloc[gid, :] = dfmetrics_list[max_rate_idx].iloc[gid, :]

    return maximal_metric



if __name__ == "__main__":

    directory_name = '/allen/aibs/mat/yazan/corticalCol/ice/sims/column/tc_only_input/tc_only_runs_2017/full_v1_col/SMART_Oct2017_runs'
    directory_name = '/allen/aibs/mat/yazan/corticalCol/ice/sims/column/tc_only_input/tc_only_runs_2017/full_v1_col/LGN3_production_runs'
    nodes_DF = pd.read_csv(directory_name + '/net/v1_nodes_with_tuning_angle_correctedLIFi5Htr3a.csv', sep = ' ')
    OSI_DSI_DF = pd.read_csv(directory_name + '/OSI_DSI_DF.csv', sep = ' ', index_col=False)





    ########################################################################################################################
    cmap1 = {
        "pop_name=='i1Htr3a'": 'indigo'
    }
    plot_order = ["pop_name=='i1Htr3a'"]

    fig, ax = plt.subplots(figsize=(24, 16))
    plot_OSI_query(ax, OSI_DSI_DF, nodes_DF, cmap1, plot_order)
    ########################################################################################################################


    ########################################################################################################################
    cmap1 = {"pop_name=='e23Cux2'": 'firebrick',
             "pop_name=='i23Pvalb'": 'blue',
             "pop_name=='i23Sst'": 'forestgreen',
             "pop_name=='i23Htr3a'": 'indigo'
             }
    plot_order = ["pop_name=='e23Cux2'", "pop_name=='i23Pvalb'", "pop_name=='i23Sst'", "pop_name=='i23Htr3a'"]

    fig, ax = plt.subplots(figsize=(24, 16))
    plot_OSI_query(ax, OSI_DSI_DF, nodes_DF, cmap1, plot_order)
    ########################################################################################################################


    ########################################################################################################################
    cmap1 = {"pop_name=='e4Scnn1a'": 'firebrick',
             "pop_name=='e4Rorb'": 'red',
             "pop_name=='e4Nr5a1'": 'indianred',
             "pop_name=='e4other'": 'orangered',
             "pop_name=='i4Pvalb'": 'blue',
             "pop_name=='i4Sst'": 'forestgreen',
             "pop_name=='i4Htr3a'": 'indigo'
             }
    plot_order = ["pop_name=='e4Scnn1a'", "pop_name=='e4Rorb'", "pop_name=='e4Nr5a1'", "pop_name=='e4other'",
                  "pop_name=='i4Pvalb'", "pop_name=='i4Sst'", "pop_name=='i4Htr3a'"]

    fig, ax = plt.subplots(figsize=(24, 16))
    plot_OSI_query(ax, OSI_DSI_DF, nodes_DF, cmap1, plot_order)
    ########################################################################################################################


    ########################################################################################################################
    cmap1 = {"pop_name=='e5Rbp4'": 'firebrick',
             "pop_name=='e5noRbp4'": 'red',
             "pop_name=='i5Pvalb'": 'blue',
             "pop_name=='i5Sst'": 'forestgreen',
             "pop_name=='i5Htr3a'": 'indigo'
             }
    plot_order = ["pop_name=='e5Rbp4'", "pop_name=='e5noRbp4'",
                  "pop_name=='i5Pvalb'", "pop_name=='i5Sst'", "pop_name=='i5Htr3a'"]

    fig, ax = plt.subplots(figsize=(24, 16))
    plot_OSI_query(ax, OSI_DSI_DF, nodes_DF, cmap1, plot_order)
    ########################################################################################################################


    ########################################################################################################################
    cmap1 = {"pop_name=='e6Ntsr1'": 'firebrick',
             "pop_name=='i6Pvalb'": 'blue',
             "pop_name=='i6Sst'": 'forestgreen',
             "pop_name=='i6Htr3a'": 'indigo'
             }
    plot_order = ["pop_name=='e6Ntsr1'", "pop_name=='i6Pvalb'", "pop_name=='i6Sst'", "pop_name=='i6Htr3a'"]

    fig, ax = plt.subplots(figsize=(24, 16))
    plot_OSI_query(ax, OSI_DSI_DF, nodes_DF, cmap1, plot_order)
    ########################################################################################################################

    plt.legend()
    plt.show()
