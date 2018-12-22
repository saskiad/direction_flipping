import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pylab as pylab

params = {'legend.fontsize': 21,
          'axes.labelsize':  21,
          'axes.titlesize':  21,
          'xtick.labelsize': 21,
          'ytick.labelsize': 21}

pylab.rcParams.update(params)



def plot_slice(parameter, param_range,  tau_sustained = 0.15, tau_transient = 0.03,
               sustained_type='ON', transient_type = 'OFF', tstep = 0.001, total_time = 10.0, save_flag = False):
    '''

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
        save_name = 'TF_DSI.png'
    elif parameter == 'SF':
        xlabel = 'SF (CPD)'
        save_name = 'SF_DSI.png'
    elif parameter == 'd':
        xlabel = 'Filter Separation (Degrees)'
        save_name = 'Distance_DSI.png'
    else:
        print "Invalid parameter given. Please choose 'd', 'SF', or 'TF'."
        return

    if tau_transient > tau_sustained:
        print "Transient time constant can't be greater than the sustained time constant"
        return

    if transient_type == 'ON':
        scale_ratio_transient = 1.0
    elif transient_type == 'OFF':
        scale_ratio_transient = -1.0
    else:
        print "transient filter can only be ON or OFF - can't plot filter"
        return

    if sustained_type == 'ON':
        scale_ratio_sustained = 1.0
    elif sustained_type == 'OFF':
        scale_ratio_sustained = -1.0
    else:
        print "transient filter can only be ON or OFF - can't plot filter"
        return

    f = 2.0                 # Default temporal frequency in Hx
    k = 0.04                # Defaul spatial frequency in cycles per degree
    delt = 5.               # Default filter separation in degrees

    time = np.arange(0, total_time, tstep)      # Time array
    filter_total_time = 2.*tau_sustained         # Since square-wave, need filter to go to zero
    DSI = np.zeros(len(param_range))              # DSI array to save all DSI values


    for i, p in enumerate(param_range):
        if parameter == 'TF':
            f = p
        elif parameter == 'SF':
            k = p
        else:
            delt = p

        rates = np.zeros(2)
        for j, degree in enumerate([0., 180.]):
            theta = degree * (np.pi/180) #np.pi/2
            delta = delt * np.cos(theta)

            x1 = 0.
            y1 = np.cos(k*x1 + 2*np.pi*f*time)

            x2 = (x1 + delta)*(2*np.pi)
            y2 = np.cos(k*x2 + 2*np.pi*f*time)

            sustained_filter = np.zeros(int(filter_total_time/tstep))
            sustained_filter[:int(tau_sustained/tstep)] = scale_ratio_sustained

            transient_filter = np.zeros(int(filter_total_time / tstep))
            transient_filter[:int(tau_transient/tstep)] = scale_ratio_transient

            convolved1 = np.convolve(sustained_filter, y1, 'valid')
            convolved2 = np.convolve(transient_filter, y2, 'valid')
            convolved1 /= np.max(convolved1)
            convolved2 /= np.max(convolved2)
            plt.figure()
            time_plot = np.arange(0, 2, tstep)
            plt.plot(time_plot,convolved1[:len(time_plot)], label = 'Sustained')
            plt.plot(time_plot,convolved2[:len(time_plot)], label = 'Transient')
            plt.legend()
            plt.title('k = ' + str(k) + ' for ' + str(degree))
            plt.savefig('k = ' + str(int(k*100)) + ' for ' + str(int(degree)))
            # plt.show()

            rates[j] = np.max(convolved2 + convolved1)

        DSI[i] = (rates[0] - rates[1]) / (rates[1] + rates[0])

    plt.figure(figsize=(8,8))
    plt.plot(param_range, DSI, c = 'royalblue')
    plt.plot([np.min(param_range), np.max(param_range)], [0, 0], 'k')
    plt.ylabel('DSI (unitless)')
    plt.xlabel(xlabel)
    if save_flag:
        plt.savefig(save_name)

    return


def plot_heatmap(parameter, param_range, tau_sustained_range = np.arange(0.03, 0.231, 0.001), tau_transient = 0.03,
                 sustained_type='ON', transient_type = 'OFF', tstep = 0.001, total_time = 10.0, save_flag = False):

    if parameter =='TF':
        xlabel = 'TF (Hz)'
        save_name = 'TF_heatmap.png'
    elif parameter == 'SF':
        xlabel = 'SF (CPD)'
        save_name = 'SF_heatmap.png'
    elif parameter == 'd':
        xlabel = 'Filter Separation (Degrees)'
        save_name = 'Distance_heatmap.png'
    else:
        print "Invalid parameter given. Please choose 'd', 'SF', or 'TF'."
        return

    if tau_transient > np.min(tau_sustained_range):
        print "Transient time constant can't be greater than the minimum sustained time constant"
        return

    if transient_type == 'ON':
        scale_ratio_transient = 1.0
    elif transient_type == 'OFF':
        scale_ratio_transient = -1.0
    else:
        print "transient filter can only be ON or OFF - can't plot filter"
        return

    if sustained_type == 'ON':
        scale_ratio_sustained = 1.0
    elif sustained_type == 'OFF':
        scale_ratio_sustained = -1.0
    else:
        print "transient filter can only be ON or OFF - can't plot filter"
        return

    f = 2.0                 # Default temporal frequency in Hx
    k = 0.04                # Defaul spatial frequency in cycles per degree
    delt = 5.               # Default filter separation in degrees

    time = np.arange(0, total_time, tstep)                     # Time array
    filter_total_time = 2.*np.max(tau_sustained_range)         # Since square-wave, need filter to go to zero

    DSI = np.zeros((len(param_range), len(tau_sustained_range)))

    for i, p in enumerate(param_range):
        if parameter == 'TF':
            f = p
        elif parameter == 'SF':
            k = p
        else:
            delt = p

        for j, tau_sustained in enumerate(tau_sustained_range):
            rates = np.zeros(2)
            for z, degree in enumerate([0., 180.]):
                theta = degree * (np.pi / 180)
                delta = delt * np.cos(theta)
                x1 = 0.
                y1 = np.cos(k*x1 + 2*np.pi*f*time)

                x2 = (x1 + delta)*2.*np.pi
                y2 = np.cos(k*x2 + 2*np.pi*f*time)

                sustained_filter = np.zeros(int(filter_total_time / tstep))
                sustained_filter[:int(tau_sustained / tstep)] = scale_ratio_sustained

                transient_filter = np.zeros(int(filter_total_time / tstep))
                transient_filter[:int(tau_transient / tstep)] = scale_ratio_transient

                convolved1 = np.convolve(sustained_filter, y1, 'valid')
                convolved2 = np.convolve(transient_filter, y2, 'valid')
                convolved1 /= np.max(convolved1)
                convolved2 /= np.max(convolved2)

                rates[z] = np.max(convolved2+ convolved1)
            DSI[i, len(tau_sustained_range) - j - 1] = (rates[0] - rates[1]) / (rates[0] + rates[1])

    plt.figure(figsize=(8,8))
    plt.imshow(np.transpose(DSI),
               interpolation=None,
               cmap='seismic',
               extent=[min(param_range),
                       max(param_range),
                       1000*(min(tau_sustained_range) - tau_transient),
                       1000*(max(tau_sustained_range) - tau_transient)],
               aspect='auto')
    np.save('TF_heatmap.npy', np.transpose(DSI))
    plt.ylabel('Delta Tau (milliseconds)')
    plt.xlabel(xlabel)
    plt.colorbar()
    plt.plot([np.min(param_range), np.max(param_range)], [120, 120], c = 'k')
    plt.grid(alpha = 0.5)
    # plt.rc('xtick', labelsize=8)
    # plt.rc('ytick', labelsize=8)

    if save_flag:
        plt.savefig(save_name)

    plt.show()

def plot_filters(tau_list = [0.03, 0.13, 0.23], totaltime = 0.5, filter_type = 'ON'):

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

    ####### Plots for TF
    # f_range = np.arange(1., 30., 0.1)
    # plot_slice('TF', f_range, save_flag = False)
    # plot_heatmap('TF', f_range, save_flag = False)
    #
    # ####### Plots for SF
    k_range = np.arange(0.005, 0.8, 0.005)
    plot_slice('SF', [0.025, 0.1, 0.175], save_flag = False)
    # plot_heatmap('SF', k_range, save_flag = False)

    ####### Plots for filter separation
    # d_range = np.arange(0., 80., 1)
    # plot_slice('d', [0, 3, 12.5, 22, 25], save_flag = False)
    # plot_heatmap('d', d_range, save_flag = False)

    plt.show()


