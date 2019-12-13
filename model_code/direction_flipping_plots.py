import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pylab as pylab

params = {'legend.fontsize': 15,
          'axes.labelsize':  15,
          'axes.titlesize':  15,
          'xtick.labelsize': 15,
          'ytick.labelsize': 15}

pylab.rcParams.update(params)

def calculate_DSI(f, k, delt,
                  total_time, tstep,
                  sustained_type, transient_type, tau_sustained, tau_transient,
                  filter = 'default', plot = False, ax = None):
    '''
    This function calculated the DSI to remove the double use of code in
    the two functions below of plot_slice and plot_heatmap. Will need
    to take in the taus, types, parameters, timestep and total
    time.
    :param f: Temporal frequency in Hz
    :param k: Spatial frequency in cycles per degree
    :param delt: Distance between filters in degrees
    :param total_time: Total simulation time in seconds
    :param tstep: Size of time-step
    :param sustained_type: Type on sustained filter (ON or OFF)
    :param transient_type: Type on transient filter (ON or OFF)
    :param tau_sustained: Time-constant for sustained filter
    :param tau_transient: Time-constant for transient filter
    :return: Final DSI value: (Pref - Null) / (Pref + Null)
    '''

    if transient_type == 'ON':
        scale_ratio_transient = 1.0
    elif transient_type == 'OFF':
        scale_ratio_transient = -1.0
    else:
        print ("transient filter can only be ON or OFF")
        return

    if sustained_type == 'ON':
        scale_ratio_sustained = 1.0
    elif sustained_type == 'OFF':
        scale_ratio_sustained = -1.0
    else:
        print ("sustained filter can only be ON or OFF")
        return


    time = np.arange(0, total_time, tstep)  # Time array
    filter_total_time = 2. * tau_sustained  # Since square-wave, need filter to go to zero

    rates = np.zeros(2)
    for j, degree in enumerate([0., 180.]):
        theta = degree * (np.pi / 180)  # np.pi/2
        delta = delt * np.cos(theta)

        x1 = 0.
        y1 = np.cos(k * x1 + 2 * np.pi * f * time)

        x2 = (x1 + delta) * (2 * np.pi)
        y2 = np.cos(k * x2 + 2 * np.pi * f * time)

        sustained_filter = np.zeros(int(filter_total_time / tstep))
        sustained_filter[:int(tau_sustained / tstep)] = scale_ratio_sustained

        transient_filter = np.zeros(int(filter_total_time / tstep))
        transient_filter[:int(tau_transient / tstep)] = scale_ratio_transient

         ### With a exponential profile
        if filter == 'exp':
            filter_time = np.arange(0, filter_total_time, tstep)
            sustained_filter = scale_ratio_sustained * np.exp(-filter_time/tau_sustained)
            transient_filter = scale_ratio_transient * np.exp(-filter_time/tau_transient)

        convolved1 = np.convolve(sustained_filter, y1, 'valid')
        convolved2 = np.convolve(transient_filter, y2, 'valid')
        # convolved1[convolved1 < 0] = 0
        # convolved2[convolved2 < 0] = 0
        convolved1 /= np.max(convolved1)
        convolved2 /= np.max(convolved2)

        if plot:
            ref_dict = {
                0.025: 0,
                0.1: 1,
                0.175: 2,
            }
            if degree == 0:
                time_plot = np.arange(0, 2/f, tstep)
            ax[j,ref_dict[k]].plot(time_plot, convolved1[:len(time_plot)], label='Sustained', alpha = 1)
            ax[j,ref_dict[k]].plot(time_plot, convolved2[:len(time_plot)], label='Transient', alpha = 1)
            ax[j,ref_dict[k]].plot(time_plot, convolved1[:len(time_plot)] + convolved2[:len(time_plot)], label = 'Sum', lw = 3, alpha = 0.5)

            if ref_dict[k] == 0:
                ax[j,ref_dict[k]].set_ylabel('Response (Arb. U.)')
            if degree == 180:
                ax[j, ref_dict[k]].set_xlabel('Time (Seconds)')
            # plt.xlim(0, np.max(time_plot) * 2.5)
            # plt.legend()
            # plt.title('f = ' + str(f) + ' for ' + str(degree))
            plt.savefig('k = ' + str(int(k*1000)))

        # threshold = 20.
        sum_filters = convolved2 + convolved1
        # sum_filters[sum_filters < threshold] = 0.
        rates[j] = np.max(convolved2 + convolved1)  # For no threshold.
        # rates[j] = np.mean(sum_filters)

     # Return DSI
    return (rates[0] - rates[1]) / (rates[1] + rates[0])

def plot_slice(parameter, param_range,  tau_sustained = 0.15, tau_transient = 0.03,
               sustained_type='ON', transient_type = 'OFF', tstep = 0.001, total_time = 10.0, save_flag = False,
               cat = False, ax = None):
    '''
    Plot Direction Selectivity Index (DSI) as function of a single parameter (TF, SF, or d).
    :param parameter: filter parameter to sweep: TF, SF, or d
    :param param_range: the range of values for the parameter
    :param tau_transient: Time constant of transient filter (seconds)
    :param tau_sustained: Time constant of sustained filter (seconds)
    :param transient_type: filter type for transient unit: ON or OFF
    :param sustained_type: filter type for sustained unit: ON or OFF
    :param tstep: time-step (i.e. resolution) to use for the convolutions (in seconds)
    :param total_time: total time, seconds: make at least 10x of tau_sustained or 10x the period of lowest frequency
    :return: Makes a plot of DSI vs. parameter and saves it
    '''

    if parameter =='TF':
        xlabel = 'TF (Hz)'
        save_name = 's' + sustained_type + 't' + transient_type + '_TF_slice.png'
    elif parameter == 'SF':
        xlabel = 'SF (CPD)'
        save_name = 's' + sustained_type + 't' + transient_type + '_SF_slice.png'
    elif parameter == 'd':
        xlabel = 'Filter Separation (Degrees)'
        save_name = 's' + sustained_type + 't' + transient_type + '_Distance_slice.png'
    else:
        print ("Invalid parameter given. Please choose 'd', 'SF', or 'TF'.")
        return

    if tau_transient > tau_sustained:
        print ("Transient time constant can't be greater than the sustained time constant")
        return

    f = 2.0                 # Default temporal frequency in Hz
    k = 0.04                # Defaul spatial frequency in cycles per degree
    delt = 5.#5.               # Default filter separation in degrees

    if cat:
        delt = 0.7
        tau_sustained = 0.05
        tau_transient = 0.04
        save_name = save_name + '_cat'


    DSI = np.zeros(len(param_range))              # DSI array to save all DSI values

    # Use this for time-plots (sinusoids)
    # fig, ax = plt.subplots(2, 3, figsize=(16, 8), sharey= True, sharex=True)

    for i, p in enumerate(param_range):
        if parameter == 'TF':
            f = p
        elif parameter == 'SF':
            k = p
        else:
            delt = p

        DSI[i] = calculate_DSI(f, k, delt,
                      total_time, tstep,
                      sustained_type, transient_type,
                      tau_sustained, tau_transient)#, ax = ax) Send ax for time-domain plots

    if ax == None:
        plt.figure(figsize=(11,11))
        plt.plot(param_range, DSI, c = 'royalblue', lw = 5.0)
        plt.plot([np.min(param_range), np.max(param_range)], [0, 0], 'k', lw = 1.75)
        plt.ylabel('DSI (unitless)')
        plt.xlabel(xlabel)
        plt.title('s' + sustained_type + 't' + transient_type)
        plt.ylim(-1, 1)
        if save_flag:
            plt.savefig(save_name + '.png')
    # if True:
    #     ax[0].plot(param_range, DSI, c = 'royalblue', lw = 5.0)
    #     ax[0].set_ylim(-1, 1)

    return

def plot_heatmap(parameter, param_range, parameter2 = 'tau_sustained', param2_range = np.arange(0.03, 0.23, 0.001),
                 tau_transient = 0.03, sustained_type='ON', transient_type = 'OFF',
                 tstep = 0.001, total_time = 10.0, cat = False, save_flag = False, ax = None):
    '''
    Plot a Direction Selectivity Index (DSI) heatmap as a function any two parameters.
    :param parameter: First parameter to sweep through (SF, TF, d, or tau_sustained)
    :param param_range: Range of selected first parameter
    :param parameter2: Optional second parameter to sweep through: default is tau_sustained
    :param param2_range: Optional parameter range for second parameter
    :param tau_transient: Transient time-constants
    :param sustained_type: Sustained type (ON or OFF)
    :param transient_type: Transient type (ON or OFF)
    :param tstep: time-step of simulation
    :param total_time: total time of simulation
    :param save_flag: Flag of whether or not to save the figure (heatmap)
    :return: a plot of the heatmap is created
    '''
    if parameter == parameter2:
        print ("Both parameters are identical - Use plot_slice() function")
        return

    f = 2.0                 # Default temporal frequency in Hz
    k = 0.04                # Defaul spatial frequency in cycles per degree
    delt = 5.#5.               # Default filter separation in degrees
    tau_sustained = 0.15    # Default sustained unit time constant in seconds

    if cat:
        delt = 0.7
        tau_sustained = 0.05
        tau_transient = 0.04
        if parameter2 == 'tau_sustained':
            param2_range = np.arange(0.04, 0.24, 0.001)

    if parameter =='TF':
        xlabel = 'TF (Hz)'
        label_extent = [min(param_range), max(param_range)]
        save_name = 's' + sustained_type + 't' + transient_type + '_TF_vs'
    elif parameter == 'SF':
        xlabel = 'SF (CPD)'
        label_extent = [min(param_range), max(param_range)]
        save_name = 's' + sustained_type + 't' + transient_type + '_SF_vs'
    elif parameter == 'd':
        xlabel = 'Filter Separation (Degrees)'
        save_name = 's' + sustained_type + 't' + transient_type + '_Distance_vs'
        label_extent = [min(param_range), max(param_range)]
    elif parameter == 'tau_sustained':
        xlabel = 'Delta Tau (milliseconds)'
        save_name = 's' + sustained_type + 't' + transient_type + 'DeltaTau_vs'
        label_extent = [1000 * (min(param2_range) - tau_transient), 1000 * (max(param2_range) - tau_transient)]
    else:
        print ("Invalid parameter given. Please choose 'd', 'SF', 'TF', or 'tau_sustained'.")
        return

    if parameter2 =='TF':
        ylabel = 'TF (Hz)'
        save_name += '_TF_heatmap'
        label_extent.append(min(param2_range))
        label_extent.append(max(param2_range))
    elif parameter2 == 'SF':
        ylabel = 'SF (CPD)'
        save_name += '_SF_heatmap'
        label_extent.append(min(param2_range))
        label_extent.append(max(param2_range))
    elif parameter2 == 'd':
        ylabel = 'Filter Separation (Degrees)'
        save_name += '_Distance_heatmap'
        label_extent.append(min(param2_range))
        label_extent.append(max(param2_range))
    elif parameter2 == 'tau_sustained':
        if tau_transient > np.min(param2_range):
            print ("Transient time constant can't be greater than the minimum sustained time constant")
            return
        ylabel = 'Delta Tau (milliseconds)'
        save_name += '_DeltaTau_heatmap'
        label_extent.append(1000 * (min(param2_range) - tau_transient))
        label_extent.append(1000 * (max(param2_range) - tau_transient))

    else:
        print ("Invalid parameter2 given. Please choose 'd', 'SF', 'TF', or 'tau_sustained'.")
        return

    if cat:
        save_name = save_name + '_cat'


    DSI = np.zeros((len(param_range), len(param2_range)))   # Heatmap DSI values

    for i, p in enumerate(param_range):
        if parameter == 'TF':
            f = p
        elif parameter == 'SF':
            k = p
        elif parameter == 'd':
            delt = p
        elif parameter == 'tau_sustained':
            tau_sustained = p

        for j, p2 in enumerate(param2_range):
            if parameter2 == 'TF':
                f = p2
            elif parameter2 == 'SF':
                k = p2
            elif parameter2 == 'd':
                delt = p2
            elif parameter2 == 'tau_sustained':
                tau_sustained = p2

            DSI[i, len(param2_range) - j - 1] = calculate_DSI(f, k, delt,
                                                                     total_time, tstep,
                                                                     sustained_type, transient_type,
                                                                     tau_sustained, tau_transient)

    if ax == None:
        plt.figure(figsize=(12,12))
        plt.imshow(np.transpose(DSI),
                   interpolation=None,
                   cmap='seismic',
                   extent=label_extent,
                   vmin = -1, vmax = 1,
                   aspect='auto')
        # np.save('TF_heatmap.npy', np.transpose(DSI))
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title('s' + sustained_type + 't' + transient_type)
        plt.colorbar(ticks= [-1, 0, 1])
        if parameter2 == 'tau_sustained' and not cat:
            plt.plot([np.min(param_range), np.max(param_range)], [120, 120], c = 'k', lw = 5.)
        elif parameter2 == 'tau_sustained' and cat:
            plt.plot([np.min(param_range), np.max(param_range)], [10, 10], c='k', lw = 5.)

        # plt.grid(alpha = 0.5)

        if save_flag:
            plt.savefig(save_name + '.png')

        plt.show()
    # if True:
    #     ax[1].imshow(np.transpose(DSI),
    #                interpolation=None,
    #                cmap='seismic',
    #                extent=label_extent,
    #                vmin = -1, vmax = 1,
    #                aspect='auto')
    #
    #     # ax[1].set_ylabel(ylabel)
    #     # ax[1].set_xlabel(xlabel)
    #     # plt.title('s' + sustained_type + 't' + transient_type)
    #     if parameter2 == 'tau_sustained' and not cat:
    #         plt.plot([np.min(param_range), np.max(param_range)], [120, 120], c = 'k', lw = 5.)
    #     elif parameter2 == 'tau_sustained' and cat:
    #         plt.plot([np.min(param_range), np.max(param_range)], [10, 10], c='k', lw = 5.)



def plot_filters(tau_list = [0.15, 0.03], totaltime = 0.3, filter_type = 'ON'):
    '''
    Function to plot square-wave filters used in the model
    :param tau_list: Values of time-constant to plot
    :param totaltime: Total time to plot
    :param filter_type: ON or OFF to determine the sign of the filter
    :return:
    '''

    plt.figure(figsize=(8,8))
    tstep = 0.001
    time = np.arange(0, totaltime, tstep)
    if filter_type == 'ON':
        scale_factor = 1.0
    elif filter_type == 'OFF':
        scale_factor = -1.0
    else:
        print ("filter_types can only be ON or OFF - can't plot filter")
        return

    colors = ['royalblue', 'orange']
    for i, tau in enumerate(tau_list):
        filter = np.zeros(len(time))
        filter[:int(tau / tstep)] = scale_factor
        print ('tau: ', tau)
        plt.plot(time, filter, lw = 5, c = colors[i%2])

    plt.xlabel('Tau (Seconds)')
    plt.ylabel('Response (unitless)')
    plt.savefig('Example Temporal Filters')

    return

def plot_sustained_trasnsient_for_V1(totaltime = 0.5):


     # This is for alpha functions
    tstep = 0.001
    time = np.arange(0, totaltime, tstep)
    tau = 0.3  ### HERE

    filter2 = np.zeros(len(time))
    delay = 0.2
    filter = (time / tau) * (np.exp(-time / tau))
    filter2[int(delay/tstep):] = ((time[int(delay/tstep):] - delay) / 0.03) * (np.exp(-(time[int(delay/tstep):] - delay) / 0.03))
    plt.plot(time, filter + filter2, lw = 12.0, color = 'darkmagenta')


if __name__ == "__main__":

     ####### Plot filter examples
    #tau_list = [0.1], totaltime = 0.8
    plot_filters()

    for sus in ['ON', 'OFF']:
        for tr in ['ON', 'OFF']:

            ##### Plots for TF
            f_range = np.arange(0.0, 25., 0.1)#0.1)
            # fig, ax = plt.subplots(1, 2, figsize=(22, 8))
            plot_slice('TF', f_range, sustained_type=sus, transient_type = tr, save_flag = True, cat = False)#, ax = ax)
            plot_heatmap('TF', f_range, sustained_type=sus, transient_type = tr, save_flag = True, cat = False)#, ax = ax)
            plt.savefig('TF_plots_combined_s' + sus + '_t' + tr + '.png')

            fig, ax = plt.subplots(1, 2, figsize=(22, 8))
            plot_slice('TF', f_range, sustained_type=sus, transient_type=tr, save_flag=True, cat=True, ax=ax)
            plot_heatmap('TF', f_range, sustained_type=sus, transient_type=tr, save_flag=True, cat=True, ax=ax)
            plt.savefig('TF_plots_combined_s' + sus + '_t' + tr + '_cat.png')


            # # ####### Plots for SF
            k_range = np.arange(0.005, 1.5, 0.005)#0.005)
            # fig, ax = plt.subplots(1, 2, figsize=(22, 8))
            plot_slice('SF', k_range, sustained_type=sus, transient_type = tr, save_flag = True)#, ax = ax)
            plot_heatmap('SF', k_range, sustained_type=sus, transient_type = tr, save_flag = True)#, ax = ax)
            plt.savefig('SF_plots_combined_s' + sus + '_t' + tr + '.png')

            fig, ax = plt.subplots(1, 2, figsize=(22, 8))
            plot_slice('SF', k_range, sustained_type=sus, transient_type=tr, save_flag=True, cat = True, ax=ax)
            plot_heatmap('SF', k_range, sustained_type=sus, transient_type=tr, save_flag=True, cat = True, ax=ax)
            plt.savefig('SF_plots_combined_s' + sus + '_t' + tr + '_cat.png')


            # ####### Plots for filter separation
            d_range = np.arange(0., 80., 1)#1)
            plot_slice('d', d_range, sustained_type=sus, transient_type = tr, save_flag = False)
            plot_heatmap('d', d_range, sustained_type=sus, transient_type = tr, save_flag = False)


            ####### Ability yo ploy TF/SF Heatmap
            plot_heatmap('TF', f_range, parameter2 = 'SF', param2_range = k_range,
                                            sustained_type=sus, transient_type=tr, save_flag=False)


    for f in [3.3, 8.3, 13.4]:                  # Default temporal frequency in Hz
        k = 0.04                                # Defaul spatial frequency in cycles pe
        delt = 5.                               # Default filter separation in degrees
        tstep = 0.001
        total_time = 10.0
        calculate_DSI(f, k, delt, total_time, tstep, 'ON', 'OFF', 0.15, 0.03, plot = True)
    plt.show()
