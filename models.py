from offline_learning import OfflineLearning

ol = OfflineLearning(data_set_size=100)
ol.train()
ol.test(print_results=False)
