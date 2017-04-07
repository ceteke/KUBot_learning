class Trial:

    def __init__(self, action, regression_score, cluster_score):
        self.action = action
        self.regression_score = regression_score
        self.cluster_score = cluster_score

    def __str__(self):
        return '%s, %f, %f,' % (self.action.name,
                                self.regression_score,
                                self.cluster_score)
