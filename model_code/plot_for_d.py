import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pylab as pylab

params = {'legend.fontsize': 21,
          'axes.labelsize':  21,
          'axes.titlesize':  21,
          'xtick.labelsize': 21,
          'ytick.labelsize': 21}

pylab.rcParams.update(params)

sustained_cell = 'ON'
stimulus = 'grating'


f = 2.     #TF in Hz
k = 0.04   #SF in cpd
deltas = np.arange(0., 80., 1)  # distance in degrees
DSI = np.zeros(len(deltas))

tstep = 0.001
time = np.arange(0, 10.0, tstep)  # 3.65
tau_transient = 0.03
tau_sustained = 0.15

for i, delt in enumerate(deltas):
    rates = np.zeros(2)
    for j, degree in enumerate([0., 180.]):
        theta = degree * (np.pi/180) #np.pi/2
        delta = delt * np.cos(theta)

        x1 = 0.  #np.arange(0,500)
        y1 = np.cos(k*x1 + 2*np.pi*f*time)


        x2 = (x1 + delta)*(2*np.pi)  #np.arange(0,500)
        y2 = np.cos(k*x2 + 2*np.pi*f*time)

        # t = np.arange(0, 10.0 * tau_sustained, tstep)
        filter_total_time = 2.0

        if sustained_cell == 'OFF':
            # filter1 = -(t / tau_sustained) * (np.exp(-t / tau_sustained))
            filter1 = np.zeros(int(filter_total_time / tstep))
            filter1[:int(tau_sustained / tstep)] = -1.0
        elif sustained_cell == 'ON':
            # filter1 = (t / tau_sustained) * (np.exp(-t / tau_sustained))
            filter1 = np.zeros(int(filter_total_time / tstep))
            filter1[:int(tau_sustained / tstep)] = 1.0

        # filter2 = -(t / tau_transient) * (np.exp(-t / tau_transient))
        filter2 = np.zeros(int(filter_total_time / tstep))
        filter2[:int(tau_transient / tstep)] = -1.0

        # filter1 = np.zeros(int(2*tau_sustained/tstep))
        # filter1[0:int(tau_sustained/tstep + 1)] = 1
        # filter2 = np.zeros(int(2*tau_transient/tstep))
        # filter2[0:int(tau_transient/tstep + 1)] = 1


        convolved1 = np.convolve(filter1, y1, 'valid')
        convolved2 = np.convolve(filter2, y2, 'valid')
        convolved1 /= np.max(convolved1)
        convolved2 /= np.max(convolved2)
        # convolved1[convolved1 < 0] = 0
        # convolved2[convolved2 < 0] = 0

        # smaller_length = np.min([convolved1.shape[0], convolved2.shape[0]])
        # rates[j] = np.max(convolved2[:smaller_length] + convolved1[:smaller_length])
        rates[j] = np.max(convolved2 + convolved1)

        # plt.figure()
        # plt.plot(convolved1)
        # plt.plot(convolved2)
    DSI[i] = (rates[0] - rates[1]) / (rates[1] + rates[0])

plt.figure(figsize=(8,8))
plt.plot(deltas, DSI, c = 'royalblue')
plt.plot([np.min(deltas) -2, np.max(deltas) + 2], [0, 0], 'k')
plt.ylabel('DSI (unitless)')
plt.xlabel('Filter Separation (Degrees)')
plt.savefig('Distance_DSI.png')



# deltas = np.arange(0., 201., 1)
tau_S = np.arange(0.03, 0.231, 0.001)
DSI = np.zeros((len(deltas), len(tau_S)))

for i, delt in enumerate(deltas):
    for j, tau_sustained in enumerate(tau_S):
        rates = np.zeros(2)
        for z, degree in enumerate([0., 180.]):
            theta = degree * (np.pi / 180)  # np.pi/2
            delta = delt * np.cos(theta)
            x1 = 0.  #np.arange(0,500)
            y1 = np.cos(k*x1 + 2*np.pi*f*time)

            x2 = (x1 + delta)*(2*np.pi)  #np.arange(0,500)
            y2 = np.cos(k*x2 + 2*np.pi*f*time)

            # t = np.arange(0, 10.0 * tau_sustained, tstep)
            filter_total_time = 2.0

            if sustained_cell == 'OFF':
                # filter1 = -(t / tau_sustained) * (np.exp(-t / tau_sustained))
                filter1 = np.zeros(int(filter_total_time / tstep))
                filter1[:int(tau_sustained / tstep)] = -1.0
            elif sustained_cell == 'ON':
                # filter1 = (t / tau_sustained) * (np.exp(-t / tau_sustained))
                filter1 = np.zeros(int(filter_total_time / tstep))
                filter1[:int(tau_sustained / tstep)] = 1.0

            # filter2 = -(t / tau_transient) * (np.exp(-t / tau_transient))
            filter2 = np.zeros(int(filter_total_time / tstep))
            filter2[:int(tau_transient / tstep)] = -1.0

            # filter1 = np.zeros(int(2*tau_sustained / tstep))
            # filter1[0:int(tau_sustained / tstep + 1)] = 1
            # filter2 = np.zeros(int(2*tau_transient / tstep))
            # filter2[0:int(tau_transient / tstep + 1)] = 1
            #
            convolved1 = np.convolve(filter1, y1, 'valid')
            convolved2 = np.convolve(filter2, y2, 'valid')
            convolved1 /= np.max(convolved1)
            convolved2 /= np.max(convolved2)

            # convolved1[convolved1 < 0] = 0
            # convolved2[convolved2 < 0] = 0

            # smaller_length = np.min([convolved1.shape[0], convolved2.shape[0]])
            # rates[z] = np.max(convolved2[:smaller_length] + convolved1[:smaller_length])
            rates[z] = np.max(convolved2 + convolved1)

        DSI[i, len(tau_S) - j - 1] = (rates[0] - rates[1]) / (rates[0] + rates[1])

plt.figure(figsize=(8,8))
plt.imshow(np.transpose(DSI),
           interpolation=None,
           cmap='seismic',
           extent = [min(deltas), max(deltas), 1000*(min(tau_S) - tau_transient), 1000*(max(tau_S) - tau_transient)],
           aspect='auto')
plt.ylabel('Delta Tau (millseconds)')
plt.xlabel('Filter separation (Degrees)')
plt.colorbar()
plt.savefig('Distance_heatmap.png')

plt.show()
####
# plt.figure()
# t = np.arange(0, 300, 0.1)
# for tau in range(5, 55, 5):
#     filter1 = (t/tau)*(np.exp(-t/tau))
#     plt.plot(t, filter1)