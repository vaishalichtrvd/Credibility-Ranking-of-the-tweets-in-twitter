import Config as cfg
import numpy as np
import PreProcessing as prp
import NB_Classfier as nb
import SVM_Classifier as svm
import KNN_Classifier as knn
import Bar_graph_plot as br

cfg.dir

if __name__ == "__main__":
    training_data = prp.Data_preprocessing('RawTrainingDataSet.csv', 'Training')
    test_data = prp.Data_preprocessing('RawTestDataSet.csv', 'Test')
    SvmClassifier = svm.SVMClassifier('Training_feature_extracted.csv')
    KnnClassifier = knn.KNNClassifier('Training_feature_extracted.csv')
    nbclassifier = nb.NbClassifier('Training_feature_extracted.csv')

    test_point = np.array([5, 8, 3])
    test_point = test_point.reshape(1, 3)
    predicted_class = SvmClassifier.classify(test_point)
    Pred_test_data_class = SvmClassifier.classify_testdata('Test_feature_extracted.csv')
    accuracy_svm=SvmClassifier.confusionMatrix(Pred_test_data_class)
    SvmClassifier.plot()


    test_Knn_point = np.array([57, 82, 3])
    test_Knn_point = test_Knn_point.reshape(1, 3)
    predicted_class = KnnClassifier.classify(test_Knn_point)
    Pred_test_data_class = KnnClassifier.classify_testdata('Test_feature_extracted.csv')
    accuracy_knn=KnnClassifier.confusionMatrix(Pred_test_data_class)
    KnnClassifier.plot()

    a, accuracy_nb = nbclassifier.classify_all('Test_feature_extracted.csv')
    nbclassifier.plot_a()

    br.bar_plot(accuracy_svm,accuracy_knn,accuracy_nb)

