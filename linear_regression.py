from offline_learning import OfflineLearning

ol = OfflineLearning()
best_trials = ol.get_best_fit(20, 0.2)
for best_trial in best_trials:
    print "Action: %s Regression Score: %f" % (best_trial.action.name,
                             best_trial.regression_score)
    best_trial.action.save('models/')
