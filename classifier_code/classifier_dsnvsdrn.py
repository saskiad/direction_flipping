# Iman Wahle
# August 26 2019
# Classify drns only vs dsns only vs non-ds cells only

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

# retrieve data cache
boc = BrainObservatoryCache(manifest_file=manifest_file)

# set to false if using drn-criteria defined drns instead
bootstrap_drns = True
dsi25 = True

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
#       to switch out or to use previously saved random series (for consistency across
#       models and/or series)
# returns:
#   x: percent DRNs in each configuration
#   accs: (N_iter, # drns, # methods) array of classification accuracies
def classification( session_id, methods, N_iter, use_events, 
                    exclude_blank_sweep, include_tf, generate_plots, 
                    folder, series):

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
    
    # construct vector indicating whether cells for this session are DRNs or DSNs
    is_drn = np.zeros(len(cells))
    is_dsn = np.zeros(len(cells))
    for i in range(len(cells)):
        idx = np.where(cell_id_key==cells[i])[0][0]
        if is_drn_key[idx]==True:
            is_drn[i] = 1
        if is_dsn_key[idx]==True:
            is_dsn[i] = 1
    is_drn = is_drn.astype(int)
    is_dsn = is_dsn.astype(int)

    # construct set of drns
    cell_idx = np.arange(len(cells))
    drn_set = cell_idx[is_drn==1] 
    N_drns = len(drn_set)
    
    # construct set of dsns
    dsn_idx = cell_idx[is_dsn==1]
    dsn_set = np.random.choice(dsn_idx, N_drns, replace=False)
    
    # construct set of non-ds cells
    nonds_idx = cell_idx[is_dsn==0]
    nonds_set = np.random.choice(nonds_idx, N_drns, replace=False)
    
    sets = [drn_set, dsn_set, nonds_set]
#     # construct lists of indices for drns and non-drns
#     nondrn_idx = cell_idx[is_drn==0]
#     drn_idx = cell_idx[is_drn==1]    
#     N_drns = len(drn_idx)
    
    # get stimulus info for drifting gratings in this experient
    stim_table = data_set.get_stimulus_table(stim_info.DRIFTING_GRATINGS)
    
    # collect response data
    if use_events:
        data_file_dg = os.path.join(save_path, 
                                'DriftingGratings', 
                                str(session_id) + "_dg_events_analysis.h5")
        X = pd.read_hdf(data_file_dg, 'mean_sweep_events')

    else:    
        analysis_set = boc.get_ophys_experiment_analysis(
            ophys_experiment_id=session_id, 
            stimulus_type=stim_info.DRIFTING_GRATINGS)
        X = analysis_set.mean_sweep_response


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
    bar = progressbar.ProgressBar(maxval=N_drns*N_iter*len(methods)*len(sets), \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    barcnt = 0

    train_accs = np.zeros((N_iter, N_drns, len(sets), len(methods)))
    test_accs = np.zeros((N_iter, N_drns, len(sets), len(methods)))
    for ins,input_set in enumerate(sets):
        for n_drns in range(N_drns):
            for n_iter in range(N_iter):
                
                chosen_neurons = np.random.choice(input_set, n_drns+1, replace=True)
                # st = time.time()
                for m, method in enumerate(methods):

                    if method in ['knn', 'svm', 'lda', 'gnb', 'mlg', 'mlp', 'ncc', 'qda', 'rfc']:
                        train_acc, test_acc = baseline_classifiers(method, X, y, chosen_neurons)
                    else:
                        print("The method specified is not supported")
                        return

                    train_accs[n_iter, n_drns, ins, m] = train_acc
                    test_accs[n_iter, n_drns, ins, m] = test_acc

                    barcnt += 1
                    bar.update(barcnt)
                # print(time.time()-st)
    bar.finish()


    set_labels = ["DRNs", "DSNs", "NonDS"]
    if generate_plots:
    
        for m, method in enumerate(methods):

            x = range(N_drns)

            plt.figure()
            for ins,input_set in enumerate(sets):
                plt.errorbar(   x, 
                                np.mean(test_accs[:,:,ins,m], axis=0)-np.mean(test_accs[:,:,ins,m][:,0]), 
                                yerr=stats.sem(test_accs[:,:,ins,m], axis=0), 
                                fmt='-o', 
                                label=set_labels[ins])
            plt.legend()
            plt.title("Demeaned Acc " + method + ", Session: " + str(session_id) + ", #DRNs: " + str(N_drns) + ", #Cells: " + str(len(chosen_neurons)))

            plt.xlabel("% DRNs")
            plt.ylabel("Accuracy")
            plt.ylim(-.03,.03)
            plt.savefig(home + "/programs/braintv/workgroups/nc-ophys/Iman/" + \
                "direction_flipping/classification_code/figures/" + folder + \
                "/" + series + "/demeaned_accuracies_" + str(method) + "_session" + str(session_id))
            # plt.show()
        
            plt.figure()
            for ins,input_set in enumerate(sets):
                plt.errorbar(   x, 
                                np.mean(test_accs[:,:,ins,m], axis=0), 
                                yerr=stats.sem(test_accs[:,:,ins,m], axis=0), 
                                fmt='-o', 
                                label=set_labels[ins])
            plt.legend()
            plt.title("Acc " + method + ", Session: " + str(session_id) + ", #DRNs: " + str(N_drns) + ", #Cells: " + str(len(chosen_neurons)))
            plt.xlabel("% DRNs")
            plt.ylabel("Accuracy")
            plt.savefig(home + "/programs/braintv/workgroups/nc-ophys/Iman/" + \
                "direction_flipping/classification_code/figures/" + folder + \
                "/" + series + "/accuracies_" + str(method) + "_session" + str(session_id))
            # plt.show()

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




def main():
    session_id = int(sys.argv[1]) 
    n_iter = int(sys.argv[2])
    folder = sys.argv[3]
    series = sys.argv[4]
    include_tf = bool(int(sys.argv[5]))
    methods = sys.argv[6:]

    xaccs = classification(session_id, methods, n_iter, use_events=1, 
                exclude_blank_sweep=1, include_tf=include_tf, generate_plots=1,
                folder=folder, series=series)

    fp = home + "/programs/braintv/workgroups/nc-ophys/Iman/" + \
        "direction_flipping/classification_code/results/"
    fn = fp + folder + "/" + series + "/accuracies_session" + str(session_id)
    np.save(fn, xaccs)
    fn_key = fp + folder + "/" + series + "/methods_session" + str(session_id)
    np.save(fn_key, methods)

if __name__ == "__main__":
    main()


















