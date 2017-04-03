from offline_learning import OfflineLearning

ol = OfflineLearning()
best_trials = ol.get_best_fit(10, 0.2)
for best_trial in best_trials:
    print "Action: %s Regression Score: %f \
    Cluster Accuracy: %f" % (best_trial.action.name,
                             best_trial.regression_score,
                             best_trial.cluster_accuracy)
    best_trial.action.save('models/')
