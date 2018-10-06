import numpy as np
from matplotlib import pyplot as plt

sustained_cell = 'ON'
stimulus = 'grating'


f = 2.     #TF in Hz
k = 0.04   #SF in cpd
deltas = np.arange(0., 126., 1)  # distance in degrees
DSI = np.zeros(len(deltas))
for i, delt in enumerate(deltas):
    rates = np.zeros(2)
    for j, degree in enumerate([0., 180.]):
        theta = degree * (np.pi/180) #np.pi/2
        delta = delt * np.cos(theta)

        tstep = 0.001
        time = np.arange(0, 1.0, tstep)# 3.65

        x1 = 0.  #np.arange(0,500)
        y1 = np.cos(k*x1 + 2*np.pi*f*time)


        x2 = x1 + delta  #np.arange(0,500)
        y2 = np.cos(k*x2 + 2*np.pi*f*time)

        if stimulus == 'bar':
            time = np.arange(0, 5, tstep)
            x1 = 2.4
            x2 = 2.4 + 0.9 * np.cos(theta)
            y1 = -np.exp(-(time - x1)**2 / 0.2**2)
            y2 = -np.exp(-(time - x2)**2 / 0.2**2)

        t = np.arange(0, 300)
        tau_sustained = 150.
        tau_transient = 30.

        if sustained_cell == 'OFF':
            filter1 = -(t / tau_sustained) * (np.exp(-t / tau_sustained))
        elif sustained_cell == 'ON':
            filter1 = (t / tau_sustained) * (np.exp(-t / tau_sustained))

        filter2 = -(t / tau_transient) * (np.exp(-t / tau_transient))

        convolved1 = np.convolve(filter1, y1, 'valid')
        convolved2 = np.convolve(filter2, y2, 'valid')
        # convolved1[convolved1 < 0] = 0
        # convolved2[convolved2 < 0] = 0

        rates[j] = np.max(convolved2 + convolved1)
    DSI[i] = (rates[0] - rates[1]) / (rates[1] + rates[0])

plt.figure()
plt.plot(deltas, DSI, c = 'royalblue')
plt.plot([np.min(deltas) -2, np.max(deltas) + 2], [0, 0], 'k')
plt.ylabel('DSI (unitless)')
plt.xlabel('Filter Separation (Degrees)')
plt.savefig('Distance_DSI.png')



# deltas = np.arange(0., 201., 1)
tau_S = np.arange(30., 231, 1)
DSI = np.zeros((len(deltas), len(tau_S)))

tstep = 0.001
time = np.arange(0, 1.0, tstep)  # 3.65

for i, delt in enumerate(deltas):
    for j, tau_sustained in enumerate(tau_S):
        rates = np.zeros(2)
        for z, degree in enumerate([0., 180.]):
            theta = degree * (np.pi / 180)  # np.pi/2
            delta = delt * np.cos(theta)
            x1 = 0.  #np.arange(0,500)
            y1 = np.cos(k*x1 + 2*np.pi*f*time)

            x2 = x1 + delta  #np.arange(0,500)
            y2 = np.cos(k*x2 + 2*np.pi*f*time)

            if stimulus == 'bar':
                time = np.arange(0, 5, tstep)
                x1 = 2.4
                x2 = 2.4 + 0.9 * np.cos(theta)
                y1 = -np.exp(-(time - x1)**2 / 0.2**2)
                y2 = -np.exp(-(time - x2)**2 / 0.2**2)

            t = np.arange(0, 300)
            tau_transient = 30.

            if sustained_cell == 'OFF':
                filter1 = -(t / tau_sustained) * (np.exp(-t / tau_sustained))
            elif sustained_cell == 'ON':
                filter1 = (t / tau_sustained) * (np.exp(-t / tau_sustained))

            filter2 = -(t / tau_transient) * (np.exp(-t / tau_transient))

            convolved1 = np.convolve(filter1, y1, 'valid')
            convolved2 = np.convolve(filter2, y2, 'valid')

            # convolved1[convolved1 < 0] = 0
            # convolved2[convolved2 < 0] = 0

            rates[z] = np.max(convolved2 + convolved1)
        DSI[i, len(tau_S) - j - 1] = (rates[0] - rates[1]) / (rates[0] + rates[1])

plt.figure()
plt.imshow(np.transpose(DSI),
           interpolation=None,
           cmap='seismic',
           extent = [min(deltas), max(deltas), min (tau_S) - tau_transient, max(tau_S) - tau_transient],
           aspect='auto')
plt.ylabel('Delta Tau (millseconds)')
plt.xlabel('Filter separation (Degrees)')
# plt.xticks(np.arange(0, 126, 25), fontsize=9)
# plt.yticks(np.arange(0, 201, 20), fontsize=9)
plt.savefig('Distance_heatmap.png')
plt.colorbar()

plt.show()
####
# plt.figure()
# t = np.arange(0, 300, 0.1)
# for tau in range(5, 55, 5):
#     filter1 = (t/tau)*(np.exp(-t/tau))
#     plt.plot(t, filter1)