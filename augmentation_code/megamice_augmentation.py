# Iman Wahle
# August 29 2019
# Megamice with incrementing number of DRNs in each mouse

# game plan
#     collect bootstrap dsi 0.3 drns and dsn soulmates
#     combine all into one pool
#     for n in range [5,10,15,20,25,30]:
#        for big_iter in range 5:
#            sample n drns and dsn soulmates
#            for m in range :n:
#                for lil_iter in range 100:
#                    for each method:
#                        classify with m drns and and n-m dsns


from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import allensdk.brain_observatory.stimulus_info as stim_info

import numpy as np
import matplotlib.pyplot as plt
import pprint
import os
import sys
import h5py
import pandas as pd
import progressbar
import warnings
import time
from itertools import permutations

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

if sys.platform == "darwin":
    home = "/Volumes"
else:
    home = "/allen"

# set paths
# got these from here: https://github.com/AllenInstitute/visual_coding_2p_analysis/blob/master/visual_coding_2p_analysis/core.py
save_path = home + "/programs/braintv/workgroups/cortexmodels/michaelbu/ObservatoryPlatformPaperAnalysis/event_analysis_files_2018_09_25" 
manifest_file = home + "/programs/braintv/workgroups/cortexmodels/michaelbu/ObservatoryPlatformPaperAnalysis/platform_boc_2018_09_25/manifest.json"

# retrieve data cache
boc = BrainObservatoryCache(manifest_file=manifest_file)


# base path
path = home + "/programs/braintv/workgroups/nc-ophys/Iman/direction_flipping/"

# load in sessions with >4 drns under the bootstrap and dsi>.3 criteria
sessions = np.load(path + "resources/drn4bs_sessions.npy")

# load soulmate data
sessions_soulmates = np.load(path + "/drn_analog_code/sessions.npy")
drn_soulmates = np.load(path + "/drn_analog_code/drns.npy", allow_pickle=True)
dsn_soulmates = np.load(path + "/drn_analog_code/soulmates.npy", allow_pickle=True)

csv = pd.read_csv(home + "/programs/braintv/workgroups/nc-ophys/Iman/" + \
                "direction_flipping/resources/dgtf_events_all_bootstrap.csv") 
cell_id_key = csv['cell_specimen_id']
is_drn_key = csv['is_drn']

dirs = np.arange(0,360,45)
tfs = np.array([1,2,4,8,15])

# load in augmented data sets
print "Loading augmented data"
aug_Xdrn = pd.read_csv(path + "augmentation_code/augmented_data/aug_Xdrn.csv")
aug_Xdsn = pd.read_csv(path + "augmentation_code/augmented_data/aug_Xdsn.csv")
tf_check = np.load(path + "augmentation_code/augmented_data/tf_check.npy")
permutations = np.load(path + "augmentation_code/augmented_data/permutations.npy")
y_check = np.load(path + "augmentation_code/augmented_data/y_check.npy")
agg_cells = np.load(path + "augmentation_code/augmented_data/agg_cells.npy")
print "Done"



# function for knn, svm, lda, gnb, mlg, mlp, ncc, qda, rfc classification
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
def baseline_classifiers(clf_method, Xnew, y): 

    # # select specified cells from X
    # Xnew = X.iloc[:,chosen_neurons]

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


def classification_main(mouse_size, n_mice, stride, save_suffix):
    print "Beginning Classification"
    # constants
    methods = ['knn', 'svm', 'lda', 'gnb', 'mlg', 'mlp', 'ncc', 'rfc']
    miter = n_mice # number of times to sample mice of given size
    niter = 100 # number of times to sample ndrns to replace ndsns

    # iterate over different sized megamice
    results = np.zeros((miter, int(mouse_size/stride+1), niter, len(methods), 2))
    results[:] = np.nan

    for mi in range(miter):
        print "MOUSE SIZE ITER: " + str(mi)
        midx = np.random.choice(aug_Xdrn.shape[1], mouse_size, replace=False)

        # iterate through all numbers of drns included
        for pi,p in enumerate(range(0,mouse_size+1,stride)): # go from 0 drns to ndrns inclusive
            print "DRN NUMBER: " + str(p)
            for ni in range(niter):
                # print "CLASSIFICATION ITER: " + str(ni)
                nidx = np.random.choice(mouse_size, p, replace=False)
                Xnew = aug_Xdsn.iloc[:,midx]
                for nid in nidx:
                    Xnew.iloc[:,nid] = aug_Xdrn.iloc[:,midx[nid]]
#                 print("midx[nidx]]:" + str(midx[nidx]))
#                 print aug_Xdrn.iloc[:,midx[nidx]].head()
                #aug_Xdrn.iloc[:,midx[nidx]].head()
                for m, method in enumerate(methods):
                    # print "METHOD: " + method
                    train_acc, test_acc = baseline_classifiers(method, Xnew, y_check)
                    results[mi,pi,ni,m,0] = train_acc
                    results[mi,pi,ni,m,1] = test_acc            


    np.save(path + "augmentation_code/results/" + "megamice_results" + str(mouse_size) + save_suffix + ".npy", results)


def main():
    mouse_size = int(sys.argv[1])
    n_mice = int(sys.argv[2])
    stride = int(sys.argv[3])
    save_suffix = sys.argv[4]

    classification_main(mouse_size, n_mice, stride, save_suffix)

if __name__ == "__main__":
    main()















