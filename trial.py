class Trial:

    def __init__(self, action, regression_score):
        self.action = action
        self.regression_score = regression_score

    def __str__(self):
        return '%s, %f, %f,' % (self.action.name,
                                self.regression_score)
