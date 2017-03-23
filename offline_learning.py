from data_handler import DataHandler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pickle
from data_handler import DataHandler

class OfflineLearning():

    def __init__(self,data_set_size=-1):
        self.data_set_size = data_set_size
        self.dh = DataHandler()
        self.dh.collect_data(self.data_set_size)
        self.clusters = {0:'Stay', 1:'Roll'}
        self.effects = {'vcylinder': 0,
                    'hcylinder': 1,
                    'box': 0,
                    'sphere': 1}

    def train(self):
        for a in self.dh.actions:
            a.train()
            a.save('models/')

    def test(self,print_results=False):
        test_dh = DataHandler(data_path='/home/cem/learning/tests/')
        test_dh.collect_data(-1)
        for a in test_dh.actions:
            self.test_action(a,print_results=print_results)

    def test_action(self, action, print_results=True):
        print "Test: %s"%(action.name)
        model1 = pickle.load(open('models/%s_linear_regression' % (action.name), 'rb'))
        model2 = pickle.load(open('models/%s_effect_cluster' % (action.name), 'rb'))
        before_scaler = pickle.load(open('models/%s_before_scaler' % (action.name), 'rb'))
        effect_scaler = pickle.load(open('models/%s_effect_scaler' % (action.name), 'rb'))

        wrong_count = 0.0
        test_set_size = 0.0
        for o in action.objects:
            for X in o.X:
                X = before_scaler.transform(X.reshape(1,-1))
                predicted_effect = model1.predict(X)
                predicted_cluster = model2.predict(predicted_effect)[0]
                if print_results:
                    print '%s in pose %d, %s' % (o.name,o.pose,self.clusters[predicted_cluster])
                if not self.effects[o.name] == predicted_cluster:
                    wrong_count += 1.0
                test_set_size += 1.0
        print "%s: %.2f accuracy. Size: %d" % (action.name,((test_set_size-wrong_count)/test_set_size)*100.0,test_set_size)
