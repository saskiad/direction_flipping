# Iman Wahle
# July 2 2019
# Workflow to use any classifier(s) on a session across % DRNs

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import allensdk.brain_observatory.stimulus_info as stim_info

import numpy as np
import os
import sys
import h5py
import pandas as pd
import progressbar
from scipy import stats
import time

from sklearn.metrics import accuracy_score
from sklearn import neighbors, svm, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier


# from keras import models
# from keras import layers
# from keras.utils import to_categorical
# from keras import backend as K
# from keras import optimizers



# set paths
# got these from here: https://github.com/AllenInstitute/ ...
# visual_coding_2p_analysis/blob/master/visual_coding_2p_analysis/core.py

if sys.platform == "darwin":
    home = "/Volumes"
else:
    home = "/allen"

save_path = home + "/programs/braintv/workgroups/cortexmodels/michaelbu/" + \
    "ObservatoryPlatformPaperAnalysis/event_analysis_files_2018_09_25" 
manifest_file = home + "/programs/braintv/workgroups/cortexmodels/michaelbu/"+\
    "ObservatoryPlatformPaperAnalysis/platform_boc_2018_09_25/manifest.json"

# pickle_path = home + "/programs/braintv/workgroups/nc-ophys/Iman/direction_flipping/stimulus_classification/pickle_jar/"
# retrieve data cache
boc = BrainObservatoryCache(manifest_file=manifest_file)


# figure out which sessions to use as tests
container_ids = [595906107, 561472631, 566307034, 679702882, 653122665, 652842570, 657391623]
sessions = boc.get_ophys_experiments(experiment_container_ids=container_ids, 
                              stimuli=[stim_info.DRIFTING_GRATINGS])

# set to false if using drn-criteria defined drns instead
bootstrap_drns = True
#dsi25 = True

# For a given session id, this function calls the specified classifier(s)
# with an incrementally increasing number of DRNs and plots the
# resulting accuracies
# arguments:
#   session_id: int session id
#   methods: string list of methods to run
#   N_iter: int number of iterations per DRN-configuration and method
#   use_events: boolean of whether to use events of DF/F
#   exclude_blank_sweep: boolean of whether to include blank sweeps as category
#   include_tf: boolean of whether to classify dir x tf or just dir
#   generate_plots: boolean of whether to generate and create plots across methods
#   folder: what folder save plots and accuracies in, within figures/ and results/
#   series: which series so save plots and accuracies in, within folder/
#   generate_random: boolean indicating whether to randomly select which drns/nondrns
#       to switch out or to use previously saved random series (for consistency across
#       models and/or series)
# returns:
#   x: percent DRNs in each configuration
#   accs: (N_iter, # drns, # methods) array of classification accuracies
def classification( session_id, methods, N_iter, use_events, 
                    exclude_blank_sweep, include_tf, generate_plots,
                    folder, series, generate_random, dsi25):

    # load in csv data
    if bootstrap_drns:
        if dsi25:
            csv = pd.read_csv(home + "/programs/braintv/workgroups/nc-ophys/Iman/" + \
                "direction_flipping/resources/dgtf_events_all_bootstrap_dsi25.csv") 
            cell_id_key = csv['cell_specimen_id']
            is_drn_key = csv['is_drn']
            # dir_key = csv['pref_dir']
        else:
            csv = pd.read_csv(home + "/programs/braintv/workgroups/nc-ophys/Iman/" + \
                "direction_flipping/resources/dgtf_events_all_bootstrap.csv") 
            cell_id_key = csv['cell_specimen_id']
            is_drn_key = csv['is_drn']
            is_dsn_key = csv['is_dsn']
            # dir_key = csv['pref_dir']
    else:
        csv = pd.read_csv(home + "/programs/braintv/workgroups/nc-ophys/Iman/" + \
            "direction_flipping/resources/dgtf_events_all.csv") 
        cell_id_key = csv['cell_specimen_id']
        is_drn_key = csv['DRN']
        # dir_key = csv['pref_dir']

    # load cell ids for this session
    data_set = boc.get_ophys_experiment_data(session_id)
    cells = data_set.get_cell_specimen_ids()

    if "cnn2" in methods:
        # get masks for specific cells
        roi_mask_list = data_set.get_roi_mask(cell_specimen_ids=cells)

        cell_coords = np.zeros((len(cells), 3))
        for i in range(len(roi_mask_list)):
            rm = roi_mask_list[i]
            cell_coords[i,:] = [rm.x, rm.y, cells[i]]
        
        # determine NN input size from number of cells
        # if len(cells) > 112:
        #     dimy = 16
        # elif len(cells) >85:
        #     dimy = 12
        # else:
        dimy = 8
        dimx = int(np.ceil(len(cells)/float(dimy)))

        # order coordinates by x
        ord_coords = cell_coords[cell_coords[:,0].argsort()]

        # construct topographic matrix by taking the first ydim
        # cells and sorting them by their y coordinates for the 
        # first row, and continuing this way for each row
        cell_map = np.zeros((dimx, dimy))
        for i in range(dimx):
            if i==dimx-1:
                row = ord_coords[i*dimy:,:]
            else:
                row = ord_coords[i*dimy:(i+1)*dimy,:]
            ord_row = row[row[:,1].argsort()]
            if ord_row.shape[0] < dimy:
                cell_map[i,:] = np.concatenate((ord_row[:,2], 
                    np.zeros(dimy-ord_row.shape[0])))
            else:
                cell_map[i,:] = ord_row[:,2]


    # construct vector indicating whether cells for this session are DRNs
    is_drn = np.zeros(len(cells))
    for i in range(len(cells)):
        idx = np.where(cell_id_key==cells[i])[0][0]
        if is_drn_key[idx]==True:
            is_drn[i] = 1
    is_drn = is_drn.astype(int)

    # construct lists of indices for drns and non-drns
    cell_idx = np.arange(len(cells))
    nondrn_idx = cell_idx[is_drn==0]
    drn_idx = cell_idx[is_drn==1]    
    N_drns = len(drn_idx)
    
    # get stimulus info for drifting gratings in this experient
    stim_table = data_set.get_stimulus_table(stim_info.DRIFTING_GRATINGS)
    
    # collect response data
    if use_events:
        data_file_dg = os.path.join(save_path, 
                                'DriftingGratings', 
                                str(session_id) + "_dg_events_analysis.h5")
        X = pd.read_hdf(data_file_dg, 'mean_sweep_events')
        if "rnn" in methods:
            X_time = pd.read_hdf(data_file_dg, "sweep_events")
            # X_time = pd.read_pickle(pickle_path + str(session_id) + ".pkl")
    else:    
        analysis_set = boc.get_ophys_experiment_analysis(
            ophys_experiment_id=session_id, 
            stimulus_type=stim_info.DRIFTING_GRATINGS)
        X = analysis_set.mean_sweep_response
        if "rnn" in methods:
            X_time = analysis_set.sweep_response

    # # format direction labels for accuracy evaluation
    # y_raw = stim_table['orientation']
    # y = (y_raw/45)
    # y = y.fillna(8).astype(int)
    
    # # exclude blank sweep class if necessary
    # if (exclude_blank_sweep):
    #     X = X[y != 8]
    #     if "rnn" in methods:
    #         X_time = X_time[y != 8]
    #     y = y[y != 8]

    # format direction labels for accuracy evaluation
    dir_raw = stim_table['orientation'].fillna(-1).astype(int) 
    tf_raw = stim_table['temporal_frequency'].fillna(-1).astype(int)
    
    # exclude blank sweep class if necessary
    if (exclude_blank_sweep):
        X = X[dir_raw != -1]
        if "rnn" in methods:
            X_time = X_time[dir_raw != -1]
        dir_raw = dir_raw[dir_raw != -1]
        tf_raw = tf_raw[tf_raw != -1]

    # encode as labels
    le = preprocessing.LabelEncoder()    
    if include_tf:
        y_raw = np.squeeze(np.dstack((dir_raw, tf_raw)))
        y_raws = [str(y_raw[i]) for i in range(len(y_raw))]
        y = le.fit_transform(y_raws)
    else:
        y = le.fit_transform(dir_raw)

    # progress bar during classification
    bar = progressbar.ProgressBar(maxval=(N_drns+1)*N_iter*len(methods), \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    barcnt = 0

    if generate_random:
        random_drns = np.zeros((N_iter, N_drns+1, N_drns))
        random_nondrns = np.zeros((N_iter, N_drns+1, N_drns))
    else:
        if dsi25:
            rset = "random_set_dsi25"
        else:
            rset = "random_set_dsi3"
        random_drns = np.load(home + "/programs/braintv/workgroups/nc-ophys/Iman/" + \
            "direction_flipping/resources/random/" + rset + "/random_drns_" + \
            str(session_id) + ".npy").astype(int)
        random_nondrns = np.load(home + "/programs/braintv/workgroups/nc-ophys/Iman/" + \
            "direction_flipping/resources/random/" + rset + "/random_nondrns_" + \
            str(session_id) + ".npy").astype(int)

    train_accs = np.zeros((N_iter, N_drns+1, len(methods)))
    test_accs = np.zeros((N_iter, N_drns+1, len(methods)))
    for n_drns in range(N_drns+1):
        for n_iter in range(N_iter):

            if generate_random:
                # determine nondrns and drns to switch
                switch_drns = np.random.choice(len(drn_idx), n_drns, replace=False)
                switch_nondrns = np.random.choice(len(nondrn_idx), n_drns, replace=False)
                random_drns[n_iter, n_drns, :n_drns] = switch_drns
                random_nondrns[n_iter, n_drns, :n_drns] = switch_nondrns
            else:
                switch_drns = random_drns[n_iter, n_drns, :n_drns]
                switch_nondrns = random_nondrns[n_iter, n_drns, :n_drns]
            chosen_neurons = np.copy(nondrn_idx)
            for d in range(n_drns):
                chosen_neurons[switch_nondrns[d]] = drn_idx[switch_drns[d]]

            st = time.time()
            for m, method in enumerate(methods):

                if method in ['knn', 'svm', 'lda', 'gnb', 'mlg', 'mlp', 'ncc', 'qda', 'rfc']:
                    train_acc, test_acc = baseline_classifiers(method, X, y, chosen_neurons)
                elif method == 'fnn':
                    train_acc, test_acc = fnn(X, y, chosen_neurons)
                elif method == 'cnn1':
                    train_acc, test_acc = cnn1(X, y, chosen_neurons, dir_key, cell_id_key, cells)
                elif method == 'cnn2':
                    train_acc, test_acc = cnn2(X, y, chosen_neurons, cells, cell_map)
                elif method == 'rnn':
                    train_acc, test_acc = rnn(X_time, y, chosen_neurons)
                else:
                    print("The method specified is not supported")
                    return

                train_accs[n_iter, n_drns, m] = train_acc
                test_accs[n_iter, n_drns, m] = test_acc

                barcnt += 1
                bar.update(barcnt)
            print(time.time()-st)
    bar.finish()

    if generate_random:
        if dsi25:
            rset = "random_set_dsi25"
        else:
            rset = "random_set_dsi3"
        np.save(home + "/programs/braintv/workgroups/nc-ophys/Iman/" + \
            "direction_flipping/resources/random/" + rset + "/random_drns_" + \
            str(session_id) + ".npy", random_drns)
        np.save(home + "/programs/braintv/workgroups/nc-ophys/Iman/" + \
            "direction_flipping/resources/random/" + rset + "/random_nondrns_" + \
            str(session_id) + ".npy", random_nondrns)
    
    if generate_plots:
        # calculate stats for plot
        means = np.mean(test_accs, axis=0)
        sems = stats.sem(test_accs, axis=0)

        x = np.round(np.arange(N_drns+1) / float(len(chosen_neurons)) * 100., decimals=2)

        plt.figure()
        for m, method in enumerate(methods):
            plt.errorbar(   x, 
                            np.mean(test_accs[:,:,m], axis=0)-np.mean(test_accs[:,:,m][:,0]), 
                            yerr=stats.sem(test_accs[:,:,m], axis=0), 
                            fmt='-o', 
                            label=method)
        plt.legend()
        plt.title("Demeaned Acc, Session: " + str(session_id) + ", #DRNs: " + str(N_drns) + ", #Cells: " + str(len(chosen_neurons)))

        plt.xlabel("% DRNs")
        plt.ylabel("Accuracy")
        plt.ylim(-.03,.03)
        plt.savefig(home + "/programs/braintv/workgroups/nc-ophys/Iman/" + \
            "direction_flipping/classification_code/figures/" + folder + \
            "/" + series + "/demeaned_accuracies_session" + str(session_id))
        

        plt.figure()
        for m, method in enumerate(methods):
            plt.errorbar(   x, 
                            np.mean(test_accs[:,:,m], axis=0), 
                            yerr=stats.sem(test_accs[:,:,m], axis=0), 
                            fmt='-o', 
                            label=method)
        plt.legend()
        plt.title("Acc, Session: " + str(session_id) + ", #DRNs: " + str(N_drns) + ", #Cells: " + str(len(chosen_neurons)))
        plt.xlabel("% DRNs")
        plt.ylabel("Accuracy")
        plt.savefig(home + "/programs/braintv/workgroups/nc-ophys/Iman/" + \
            "direction_flipping/classification_code/figures/" + folder + \
            "/" + series + "/accuracies_session" + str(session_id))

    return(x, test_accs, train_accs)


# function for knn, svm, lda, gnb classification
# arguments:
#   clf_method: string flag of which classification method to use. Options are:
#       'knn': k-nearest neighbors
#       'svm': support vector machine classifier
#       'lda': linear discriminant analysis
#       'gnb': gaussian naive bayes
#   X: (trials, neurons) array of neuron activity
#   y: (trials,) array of classification labels
#   chosen_neurons: array of indices of neurons to include in this fit
# returns: 
#     acc: float value of classification accuracy (percent of trials labeled correctly)
def baseline_classifiers(clf_method, X, y, chosen_neurons): 
     
    # select specified cells from X
    Xnew = X.iloc[:,chosen_neurons]
    
    # classify trials
    if clf_method == "knn":
        n_neighbors = 8
        clf = neighbors.KNeighborsClassifier(n_neighbors)  
    elif clf_method ==  "svm":
        clf = svm.LinearSVC(random_state=0)
    elif clf_method == "lda":
        clf = LinearDiscriminantAnalysis()
    elif clf_method == "gnb":
        clf = GaussianNB()   
    elif clf_method == "mlg":
        clf = LogisticRegression(random_state=0, solver='sag', multi_class='multinomial')
    elif clf_method == "mlp":
        clf = MLPClassifier()
    elif clf_method == "ncc":
        clf = NearestCentroid()
    elif clf_method == "qda":
        clf = QuadraticDiscriminantAnalysis()
    elif clf_method == "rfc":
        clf = RandomForestClassifier(random_state=0)
    else:
        print("please pass valid clf_method")
        return
    
    scores = cross_validate(clf, Xnew, y, cv=5, return_train_score=True)   
    train_acc = scores['train_score'].mean()    
    test_acc = scores['test_score'].mean()
    return(train_acc, test_acc)


# function for FNN classification
# arguments:
#   X: (trials, neurons) array of neuron activity
#   y: (trials,) array of classification labels
#   chosen_neurons: array of indices of neurons to include in this fit
# returns: 
#     acc: float value of classification accuracy (percent of trials labeled correctly)
def fnn(X, y, chosen_neurons):

    Xnew = X.iloc[:,chosen_neurons]
    
    # record data dimensions
    n_neurons = Xnew.shape[1]
    n_dirs = len(np.unique(y))

    # make training and test sets
    X_train, X_test, y_train, y_test = \
        train_test_split(Xnew, y, test_size=0.2, random_state=42)
    
    # define feedforward neural network
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(n_neurons,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(n_dirs, activation='softmax'))
    # model.summary()

    # compile model
    model.compile(  optimizer='rmsprop', 
                    loss='sparse_categorical_crossentropy', 
                    metrics=['accuracy'])

    # fit model
    n_epochs= 120
    model.fit(X_train, y_train, epochs=n_epochs, batch_size=64, verbose=0)
    
    # training acc
    y_train_pred_prob = model.predict(X_train)
    y_train_pred = np.argmax(y_train_pred_prob, axis=1)
    train_acc = accuracy_score(y_train_pred, y_train)

    # test acc
    y_test_pred_prob = model.predict(X_test)
    y_test_pred = np.argmax(y_test_pred_prob, axis=1)
    test_acc = accuracy_score(y_test_pred, y_test)

    K.clear_session()
    return(train_acc, test_acc)


def get_next_coords(x, y, xdim,ydim):
    X=xdim-1
    Y = ydim-1
    xmid = int(round((X)/2))
    if y==Y and x==X:
        return None
    elif y==Y and x==xmid:
        return xmid+1, 0
    elif x==xmid and y in [3, 7, 11]:
            return 0, y+1
    elif x==X and y in [3, 7, 11]:
            return xmid+1, y+1
    elif y in [3, 7, 11, 15]:
        return x+1, y-3
    else:
        return x, y+1


# function for CNN classification
# This CNN organizes neurons mean activity per trial into a 2D array where
# neurons with similar direction preference are near each other
# arguments:
#   X: (trials, neurons) array of neuron activity
#   y: (trials,) array of classification labels
#   chosen_neurons: array of indices of neurons to include in this fit
#   dir_key: int list of direction preference of all cells from dgtf_all.csv
#   cell_id_key: int list of all cell ids from dgtf_all.csv
#   cells: int list of cell ids for cells in this session
# returns: 
#     acc: float value of classification accuracy (percent of trials labeled correctly)
def cnn1(X, y, chosen_neurons, dir_key, cell_id_key, cells):

    Xnew = X.iloc[:,chosen_neurons]
    n_dirs = len(np.unique(y))
    
    # find preferred directions for these selected neurons
    dirX = np.zeros(len(chosen_neurons))
    for i in range(len(chosen_neurons)):
        dirX[i] = dir_key[np.where(cell_id_key==cells[chosen_neurons[i]])[0][0]]
        
    # order these cells by direction
    dir_order = np.argsort(dirX)
    
    # what size should we make our inputs
    # if len(cells) > 112:
    #     ydim = 16
    # elif len(cells) >85:
    #     ydim = 12
    # else:
    ydim = 8
    xdim = int(np.ceil(Xnew.shape[1]/float(ydim)))
    n_trials = Xnew.shape[0]
    X2d = np.zeros((n_trials, xdim, ydim, 1))

    # now we can fill in X2d so that data for neurons with similiar
    # direction preference will be near each other
    for i in range(n_trials):
        x1, x2 = 0, -1
        for j in range(len(dir_order)):
            x1, x2 = get_next_coords(x1, x2, xdim,ydim)
            X2d[i, x1, x2, 0] = Xnew.iloc[i, dir_order[j]]
        
    # make training and test sets
    X_train, X_test, y_train, y_test = \
        train_test_split(X2d, y, test_size=0.2, random_state=42)
    
    # define convolutional neural network
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3,3), activation='relu', input_shape=(xdim, ydim, 1)))
    model.add(layers.MaxPool2D(2,2))
    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(n_dirs, activation='softmax'))
    # model.summary()

    # compile model
    model.compile(  optimizer='rmsprop', 
                    loss='sparse_categorical_crossentropy', 
                    metrics=['accuracy'])
    
    # fit model
    n_epochs= 30
    history = model.fit(X_train,
                    y_train,
                    epochs=n_epochs,
                    batch_size=128,
                    verbose=0)

    # training acc
    y_train_pred_prob = model.predict(X_train)
    y_train_pred = np.argmax(y_train_pred_prob, axis=1)
    train_acc = accuracy_score(y_train_pred, y_train)

    # test acc
    y_test_pred_prob = model.predict(X_test)
    y_test_pred = np.argmax(y_test_pred_prob, axis=1)
    test_acc = accuracy_score(y_test_pred, y_test)

    K.clear_session()

    return(train_acc, test_acc)




# function for CNN classification
# This CNN organizes neurons mean activity per trial into a 2D array where
# the data for neurons that are physically proximal is also close together
# arguments:
#   X: (trials, neurons) array of neuron activity
#   y: (trials,) array of classification labels
#   chosen_neurons: array of indices of neurons to include in this fit
#   cells: int list of cell ids for cells in this session
#   cell_map: 2D array with cell_ids indicating where cell's data should go
# returns: 
#     acc: float value of classification accuracy (percent of trials labeled correctly)
def cnn2(X, y, chosen_neurons, cells, cell_map):

    Xnew = X.iloc[:,chosen_neurons]
    n_trials = Xnew.shape[0]
    n_dirs = len(np.unique(y))

    X2d = np.zeros((n_trials, cell_map.shape[0], cell_map.shape[1], 1))
    for i in range(len(chosen_neurons)):
        locx, locy = np.where(cell_map==cells[chosen_neurons[i]])
        X2d[:, locx[0], locy[0], 0] = Xnew.iloc[:,i]
        
    # make training and test sets
    X_train, X_test, y_train, y_test = \
        train_test_split(X2d, y, test_size=0.2, random_state=42)
    
    # define convolutional neural network
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu', 
        input_shape=(cell_map.shape[0], cell_map.shape[1], 1)))
    model.add(layers.MaxPool2D(2,2))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))    
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(n_dirs, activation='softmax'))
    # model.summary()

    # compile model
    model.compile(  optimizer='rmsprop', 
                    loss='sparse_categorical_crossentropy', 
                    metrics=['accuracy'])
    
    # fit model
    n_epochs= 30
    history = model.fit(X_train,
                    y_train,
                    epochs=n_epochs,
                    batch_size=128,
                    verbose=0)

    # training acc
    y_train_pred_prob = model.predict(X_train)
    y_train_pred = np.argmax(y_train_pred_prob, axis=1)
    train_acc = accuracy_score(y_train_pred, y_train)

    # test acc
    y_test_pred_prob = model.predict(X_test)
    y_test_pred = np.argmax(y_test_pred_prob, axis=1)
    test_acc = accuracy_score(y_test_pred, y_test)

    K.clear_session()
    return(train_acc, test_acc)


# function for RNN classification
# arguments:
#   X: (trials, neurons, time) array of neuron activity over time
#   y: (trials,) array of classification labels
#   chosen_neurons: array of indices of neurons to include in this fit
# returns: 
#     acc: float value of classification accuracy (percent of trials labeled correctly)
def rnn(X, y, chosen_neurons):

    Xnew = X.iloc[:,chosen_neurons]
    Xnew = Xnew.values
    n_trials = Xnew.shape[0]
    n_dirs = len(np.unique(y))
    n_neurons = Xnew.shape[1]
    
    # now we need to rearrange X
    n_time = Xnew[0,0].shape[0]-30 # we're not gonna look at time before stim for now
    Xr = np.zeros((n_trials, n_time, n_neurons))
    for i in range(n_trials):
        for j in range(n_neurons):
            Xr[i,:, j] = Xnew[i,j][30:] 

    # make training and test sets
    X_train, X_test, y_train, y_test = \
        train_test_split(Xr, y, test_size=0.2, random_state=42)
    
    # define convolutional neural network
    model = models.Sequential()
    model.add(layers.GRU(64, # new 
                         input_shape=(n_time, n_neurons), 
                         return_sequences=False, 
                         dropout=0.2, # new
                         recurrent_dropout=0.2,)) # new
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(n_dirs, activation='softmax'))
    # model.summary()

    # compile model
    # rmsprop = optimizers.RMSprop(lr=0.02, rho=0.9, epsilon=None, decay=0.0) # new
    model.compile(  optimizer='rmsprop', # new
                    loss='sparse_categorical_crossentropy', 
                    metrics=['accuracy'])
    
    # fit model
    # n_epochs= 175
    n_epochs = 150 # new
    history = model.fit(X_train,
                    y_train,
                    epochs=n_epochs,
                    batch_size=128,
                    verbose=0)

    # training acc
    y_train_pred_prob = model.predict(X_train)
    y_train_pred = np.argmax(y_train_pred_prob, axis=1)
    train_acc = accuracy_score(y_train_pred, y_train)

    # test acc
    y_test_pred_prob = model.predict(X_test)
    y_test_pred = np.argmax(y_test_pred_prob, axis=1)
    test_acc = accuracy_score(y_test_pred, y_test)

    K.clear_session()
    return(train_acc, test_acc)




def main():
    session_id = int(sys.argv[1]) # sessions[int(sys.argv[1])]['id']
    n_iter = int(sys.argv[2])
    folder = sys.argv[3]
    series = sys.argv[4]
    generate_random = bool(int(sys.argv[5]))
    include_tf = bool(int(sys.argv[6]))
    dsi25 = bool(int(sys.argv[7]))
    methods = sys.argv[8:]

    xaccs = classification(session_id, methods, n_iter, use_events=1, 
                exclude_blank_sweep=1, include_tf=include_tf, generate_plots=1,
                folder=folder, series=series, generate_random=generate_random, dsi25=dsi25)

    fp = home + "/programs/braintv/workgroups/nc-ophys/Iman/" + \
        "direction_flipping/classification_code/results/"
    fn = fp + folder + "/" + series + "/accuracies_session" + str(session_id)
    np.save(fn, xaccs)
    fn_key = fp + folder + "/" + series + "/methods_session" + str(session_id)
    np.save(fn_key, methods)

if __name__ == "__main__":
    main()


# command line call:
# python classifier_main.py session_id n_iter folder series methods




