import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pylab as pylab

params = {'legend.fontsize': 21,
          'axes.labelsize':  21,
          'axes.titlesize':  21,
          'xtick.labelsize': 21,
          'ytick.labelsize': 21}

pylab.rcParams.update(params)

def calculate_DSI(f, k, delt,
                  total_time, tstep,
                  sustained_type, transient_type, tau_sustained, tau_transient):
    '''
    This function calculated the DSI to remove the double use of code in
    the two functions below of plot_slice and plot_heatmap. Will need
    to take in the taus, types, parameters, timestep and total
    time.
    :param f:
    :param k:
    :param delt:
    :param total_time:
    :param tstep
    :param sustained_type
    :param transient_type
    :param tau_sustained
    :param tau_transient
    :return: Final DSI value: (Pref - Null) / (Pref + Null)
    '''

    if transient_type == 'ON':
        scale_ratio_transient = 1.0
    elif transient_type == 'OFF':
        scale_ratio_transient = -1.0
    else:
        print "transient filter can only be ON or OFF"
        return

    if sustained_type == 'ON':
        scale_ratio_sustained = 1.0
    elif sustained_type == 'OFF':
        scale_ratio_sustained = -1.0
    else:
        print "sustained filter can only be ON or OFF"
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

        convolved1 = np.convolve(sustained_filter, y1, 'valid')
        convolved2 = np.convolve(transient_filter, y2, 'valid')
        convolved1 /= np.max(convolved1)
        convolved2 /= np.max(convolved2)
        # plt.figure()
        # time_plot = np.arange(0, 2, tstep)
        # plt.plot(time_plot, convolved1[:len(time_plot)], label='Sustained')
        # plt.plot(time_plot, convolved2[:len(time_plot)], label='Transient')
        # plt.legend()
        # plt.title('k = ' + str(k) + ' for ' + str(degree))
        # plt.savefig('k = ' + str(int(k * 100)) + ' for ' + str(int(degree)))
        # plt.show()
        rates[j] = np.max(convolved2 + convolved1)

    # Return DSI
    return (rates[0] - rates[1]) / (rates[1] + rates[0])

def plot_slice(parameter, param_range,  tau_sustained = 0.15, tau_transient = 0.03,
               sustained_type='ON', transient_type = 'OFF', tstep = 0.001, total_time = 10.0, save_flag = False):
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
        print "Invalid parameter given. Please choose 'd', 'SF', or 'TF'."
        return

    if tau_transient > tau_sustained:
        print "Transient time constant can't be greater than the sustained time constant"
        return

    f = 2.0                 # Default temporal frequency in Hz
    k = 0.04                # Defaul spatial frequency in cycles per degree
    delt = 5.               # Default filter separation in degrees

    DSI = np.zeros(len(param_range))              # DSI array to save all DSI values

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
                      tau_sustained, tau_transient)

    plt.figure(figsize=(8,8))
    plt.plot(param_range, DSI, c = 'royalblue')
    plt.plot([np.min(param_range), np.max(param_range)], [0, 0], 'k')
    plt.ylabel('DSI (unitless)')
    plt.xlabel(xlabel)
    plt.title('s' + sustained_type + 't' + transient_type)
    if save_flag:
        plt.savefig(save_name)

    return

def plot_heatmap(parameter, param_range, parameter2 = 'tau_sustained', param2_range = np.arange(0.03, 0.231, 0.001),
                 tau_transient = 0.03, sustained_type='ON', transient_type = 'OFF',
                 tstep = 0.001, total_time = 10.0, save_flag = False):
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
        print "Both parameters are identical - Use plot_slice() function"
        return

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
        print "Invalid parameter given. Please choose 'd', 'SF', 'TF', or 'tau_sustained'."
        return

    if parameter2 =='TF':
        ylabel = 'TF (Hz)'
        save_name += '_TF_heatmap.png'
        label_extent.append(min(param2_range))
        label_extent.append(max(param2_range))
    elif parameter2 == 'SF':
        ylabel = 'SF (CPD)'
        save_name += '_SF_heatmap.png'
        label_extent.append(min(param2_range))
        label_extent.append(max(param2_range))
    elif parameter2 == 'd':
        ylabel = 'Filter Separation (Degrees)'
        save_name += '_Distance_heatmap.png'
        label_extent.append(min(param2_range))
        label_extent.append(max(param2_range))
    elif parameter2 == 'tau_sustained':
        if tau_transient > np.min(param2_range):
            print "Transient time constant can't be greater than the minimum sustained time constant"
            return
        ylabel = 'Delta Tau (milliseconds)'
        save_name += '_DeltaTau_heatmap.png'
        label_extent.append(1000 * (min(param2_range) - tau_transient))
        label_extent.append(1000 * (max(param2_range) - tau_transient))

    else:
        print "Invalid parameter2 given. Please choose 'd', 'SF', 'TF', or 'tau_sustained'."
        return


    f = 2.0                 # Default temporal frequency in Hz
    k = 0.04                # Defaul spatial frequency in cycles per degree
    delt = 5.               # Default filter separation in degrees
    tau_sustained = 0.15    # Default sustained unit time constant in seconds

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
    plt.figure(figsize=(8,8))
    plt.imshow(np.transpose(DSI),
               interpolation=None,
               cmap='seismic',
               extent=label_extent,
               aspect='auto')
    # np.save('TF_heatmap.npy', np.transpose(DSI))
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title('s' + sustained_type + 't' + transient_type)
    plt.colorbar()
    if parameter2 == 'tau_sustained':
        plt.plot([np.min(param_range), np.max(param_range)], [120, 120], c = 'k')

    plt.grid(alpha = 0.5)

    if save_flag:
        plt.savefig(save_name)

    plt.show()



def plot_filters(tau_list = [0.03, 0.13, 0.23], totaltime = 0.5, filter_type = 'ON'):
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
        print "filter_types can only be ON or OFF - can't plot filter"
        return

    for tau in tau_list:
        filter = np.zeros(len(time))
        filter[:int(tau / tstep)] = scale_factor
        plt.plot(time, filter)
    plt.xlabel('Tau (Seconds)')
    plt.ylabel('Response (unitless)')
    plt.savefig('Example Temporal Filters')

    return


if __name__ == "__main__":

    ####### Plot filter examples
    plot_filters()

    for sus in ['ON', 'OFF']:
        for tr in ['ON', 'OFF']:
            ####### Plots for TF
            f_range = np.arange(1., 30., 0.1)#0.1)
            plot_slice('TF', f_range, sustained_type=sus, transient_type = tr, save_flag = True)
            plot_heatmap('TF', f_range, sustained_type=sus, transient_type = tr, save_flag = True)


            # ####### Plots for SF
            k_range = np.arange(0.005, 0.8, 0.005)#0.005)
            plot_slice('SF', k_range, sustained_type=sus, transient_type = tr, save_flag = True)
            plot_heatmap('SF', k_range, sustained_type=sus, transient_type = tr, save_flag = True)

            ####### Plots for filter separation
            d_range = np.arange(0., 80., 1)#1)
            plot_slice('d', d_range, sustained_type=sus, transient_type = tr, save_flag = True)
            plot_heatmap('d', d_range, sustained_type=sus, transient_type = tr, save_flag = True)


            ####### TF/SF Heatmap
            plot_heatmap('TF', f_range, parameter2 = 'SF', param2_range = k_range,
                                            sustained_type=sus, transient_type=tr, save_flag=True)


    plt.show()