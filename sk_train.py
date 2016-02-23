## Ad-hoc tau ID training with sklearn using ROOT trees as input
# Requires root_numpy https://github.com/rootpy/root_numpy
# Jan Steggemann 27 Aug 2015

import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# from sklearn.cross_validation import train_test_split #cross_val_score
from sklearn.cross_validation import KFold

from sklearn.metrics import roc_curve

# For model I/O
from sklearn.externals import joblib

from root_numpy import root2array, root2rec

def trainVars():
    return [
    'recTauEta', 'recTauPt', 'recTauDecayMode', 
    'leadPFChargedHadrCandPt', 'tauIsoDeltaR05PtThresholdsLoose3HitsChargedIsoPtSum', 'tauIsoDeltaR05PtThresholdsLoose3HitsNeutralIsoPtSumWeight', 'tauIsoDeltaR05PtThresholdsLoose3HitsFootprintCorrection', 'tauIsoDeltaR05PtThresholdsLoose3HitsPhotonPtSumOutsideSignalCone', 
    'recImpactParam', 
    'recImpactParamSign', 
    'recImpactParam3D', 
    'recImpactParamSign3D',
    #'recImpactParamTk2', 'recImpactParamSignTk2', 
    'recImpactParam3DTk2', 'recImpactParamSign3DTk2', 
    #'recImpactParamTk3', 'recImpactParamSignTk3', 
    'recImpactParam3DTk3', 'recImpactParamSign3DTk3', 'recDecayLengthTk1', 'recDecayLengthSignTk1', 'recDecayLengthTk2', 
    # 'recDecayLengthSignTk2', 
    'recDecayLengthTk3', 
    # 'recDecayLengthSignTk3', 
    'recDecayDist2D', 'recDecayDistSign2D', 
    #'recChi2DiffEvtVertex', 
    #'hasRecDecayVertex', 
    'recDecayVertexChi2', 'recDecayDistMag', 
    #'recDecayDistSign', 
    # 'numOfflinePrimaryVertices', 
    'recTauPtWeightedDetaStrip', 
    #'recTauPtWeightedDphiStrip', 
    #'recTauPtWeightedDrSignal', 
    'recTauPtWeightedDrIsolation', 'recTauNphoton', 
    'recTauEratio'
    #, 'recTauLeadingTrackChi2'
    ]

# calculates RO

def trainRandomForest(training_data, target, weights):
    clf = RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=7, min_samples_split=2, min_samples_leaf=1, max_features='auto', bootstrap=True, oob_score=True, n_jobs=1, random_state=1, verbose=1, min_density=None, compute_importances=None)
    return train(clf, training_data, target, weights)

def trainAdaBoost(training_data, target, weights):
    clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(compute_importances=None, criterion='gini', max_depth=6, max_features='auto', min_density=None, min_samples_leaf=2, min_samples_split=2, random_state=1, splitter='best'), n_estimators=5000, learning_rate=0.0025, algorithm='SAMME.R')
    return train(clf, training_data, target, weights)

def trainGBRT(training_data, target, weights, learning_rate=0.01, max_depth=6, n_estimators=1000, subSample=0.5):
    clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=1, loss='deviance', verbose=1, subsample=subSample, max_features=0.4) #loss='exponential'/'deviance'
     # loss='deviance', verbose=1, subsample=subSample)
    return train(clf, training_data, target, weights)


def train(clf, training_data, target, weights):
    print clf

    sumWeightsSignal = np.sum(weights * target)
    sumWeightsBackground = sum(weights * (1 - target))

    print 'Sum weights signal', sumWeightsSignal
    print 'Sum weights background', sumWeightsBackground

    aveWeightSignal = sumWeightsSignal/np.sum(target)
    print 'Average weight signal', aveWeightSignal
    aveWeightBG = sumWeightsSignal/np.sum(1-target)
    print 'Average weight background', aveWeightBG

    
    nCrossVal = 2
    kf = KFold(len(training_data), nCrossVal, shuffle=True, random_state=1)

    print 'Cross-validation:', nCrossVal, 'folds'

    for trainIndices, testIndices in kf:
        print 'Starting fold'

        d_train = training_data[trainIndices]
        d_test = training_data[testIndices]

        t_train = target[trainIndices]
        t_test = target[testIndices]

        w_train = weights[trainIndices]
        w_test = weights[testIndices]

        del training_data, target, weights, trainIndices, testIndices, kf

        # import pdb; pdb.set_trace()

        clf.fit(d_train, t_train, w_train)

        print 'Produce scores'
        scores = clf.decision_function(d_test)

        fpr, tpr, tresholds = roc_curve(t_test, scores, sample_weight=w_test)

        joblib.dump((fpr, tpr, tresholds), 'roc_vals.pkl')

        effs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        for eff in effs:
            print 'Fake rate at signal eff', eff, fpr[np.argmax(tpr>eff)]

        break
    
    # Can save with different features if necessary
    joblib.dump(clf, 'train/{name}_clf.pkl'.format(name=clf.__class__.__name__), compress=9)

    # if doCrossVal:
    print 'Feature importances:'
    print clf.feature_importances_

    varList = trainVars()
    for i, imp in enumerate(clf.feature_importances_):
        print imp, varList[i] if i<len(varList) else 'N/A'
    
    return clf


def readFiles():
    print 'Reading files...'


    weightsS = root2rec('data/reweightTreeTauIdMVA_mvaIsolation3HitsDeltaR05opt18aPuWeightLTBDT3_signal.root', 'reweightedTauIdMVATrainingNtuple', ['evtWeight'])
    weightsB = root2rec('data/reweightTreeTauIdMVA_mvaIsolation3HitsDeltaR05opt18aPuWeightLTBDT3_background.root', 'reweightedTauIdMVATrainingNtuple', ['evtWeight'])

    nS = len(weightsS)
    nB = len(weightsB)

    fullWeight = np.concatenate((weightsS, weightsB))
    fullWeight = fullWeight['evtWeight']

    del weightsS, weightsB

    arrSB = root2array(['data/reweightTreeTauIdMVA_mvaIsolation3HitsDeltaR05opt18aPuWeightLTBDT3_signal.root', 'data/reweightTreeTauIdMVA_mvaIsolation3HitsDeltaR05opt18aPuWeightLTBDT3_background.root'], 'reweightedTauIdMVATrainingNtuple', trainVars())

    # Need a matrix-like array instead of a 1-D array of lists for sklearn
    arrSB = (np.asarray([arrSB[var] for var in trainVars()])).transpose()

    targets = np.concatenate((np.ones(nS),np.zeros(nB)))

    print 'Done reading files.'

    return arrSB, fullWeight, targets


if __name__ == '__main__':

    classifier = 'GBRT' # 'Ada' #'GBRT'
    doTrain = True

    print 'Read training and test files...'
    training, weights, targets = readFiles()        

    print 'Sizes'
    print training.nbytes, weights.nbytes, targets.nbytes

    if doTrain:
        print 'Start training'

        if classifier == 'GBRT':
            clf = trainGBRT(training, targets, weights)
        elif classifier == 'Ada':
            clf = trainAdaBoost(training, targets, weights)
        elif classifier == 'RF':
            clf = trainRandomForest(training, targets, weights)
        else:
            print 'ERROR: no valid classifier', classifier
    
    # if doTest:
    #     print 'Loading classifier'
    #     if classifier == 'GBRT':
    #         clf = joblib.load('train/GradientBoostingClassifier_clf.pkl')
    #     elif classifier == 'RF':
    #         clf = joblib.load('train/RandomForestClassifier_clf.pkl')
    #     elif classifier == 'Ada':
    #         clf = joblib.load('train/AdaBoostClassifier_clf.pkl')
    #     else:
    #         print 'ERROR: no valid classifier', classifier

