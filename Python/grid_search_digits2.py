"""
============================================================
Parameter estimation using grid search with cross-validation
============================================================

This examples shows how a classifier is optimized by cross-validation,
which is done using the :class:`sklearn.model_selection.GridSearchCV` object
on a development set that comprises only half of the available labeled data.

The performance of the selected hyper-parameters and trained model is
then measured on a dedicated evaluation set that was not used during
the model selection step.

More details on tools available for model selection can be found in the
sections on :ref:`cross_validation` and :ref:`grid_search`.

"""

from __future__ import print_function

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

#print(__doc__)


from os.path import dirname, exists, expanduser, isdir, join, splitext
import numpy as np
import csv


from sklearn.metrics import confusion_matrix
from sklearn import model_selection
import pickle


def avaliacao(cnf_matrix, imprimir=0):
    #print ("\n matriz")
    vp = cnf_matrix[0][0] # VP
    fn = cnf_matrix[0][1] # FN
    fp = cnf_matrix[1][0] # FP
    vn = cnf_matrix[1][1] # VN

    acuracia = (vp+vn)/(vp+vn+fp+fn) # perc de acertos total
    sensibilidade = vp/(vp+fn) # perc de positivos acertados # recall
    especificidade = vn/(vn+fp) # perc de negativos acertados
    precisao = vp/(vp+fp) # prob de repetir a acuracia

    if (imprimir==1):
        print ("acuracia = ", acuracia)
        print ("sensibilidade = ", sensibilidade)
        print ("especificidade = ", especificidade)
        print ("precisao = ", precisao)

    return acuracia, sensibilidade





def load_data(module_path, data_file_name):

    with open(join(module_path, data_file_name)) as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        target_names = np.array(temp[2:])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=np.float64)
            target[i] = np.asarray(ir[-1], dtype=np.int)

    return data, target, target_names



data, target, target_names = load_data('D:/Code/FA/arff/ext_csv', 'k8.csv')





# Loading the Digits dataset
digits = datasets.load_digits()

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
#X = digits.images.reshape((n_samples, -1))
#y = digits.target

X = data 
y = target


# melhor {'kernel': 'linear', 'C': 121}
# melhor: {'gamma': 0.08, 'kernel': 'rbf', 'C': 81}
# melhor 0.810 (+/-0.402) for {'C': 931, 'gamma': 0.01251, 'kernel': 'rbf'}

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=1)

listaC = [x for x in range(1,100,1)]  # [1, 10, 100, 1000]
#listaC = [81]
#listaC = [931]

#listaGama = [0.08]
#listaGama = [1e-2, 2e-2, 4e-2, 6e-2, 8e-2, 1e-3, 2e-3, 4e-3, 6e-3, 8e-3, 1e-4, 2e-4, 6e-4, 8e-4, 1e-5]
listaGama = [0.00001*x for x in range(1,10000,10)]  # [1, 10, 100, 1000]
#listaGama = [0.01251]

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': listaGama,
                     'C': listaC},
                    {'kernel': ['linear'], 'C': listaC}]
'''
tuned_parameters = [{'kernel': ['rbf'], 'gamma': listaGama, 'C': listaC}]
'''

maioracuracia = 0
maiorsensibilidade = 0
scores = ['recall', 'precision']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=10,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    #print("Grid scores on development set:") # para imprimir na tela
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    
    '''
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    '''
    

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)


    # adicionei
    cnf_matrix = confusion_matrix(y_true, y_pred) 

    acc, sens = avaliacao(cnf_matrix)
    if(acc>maioracuracia): # salva o modelo   
        maioracuracia=acc     
        pickle.dump(clf, open('model_acc.sav', 'wb'))
    if(sens>maiorsensibilidade): # salva o modelo      
        maiorsensibilidade=sens  
        print(maiorsensibilidade)
        pickle.dump(clf, open('model_sens.sav', 'wb'))

    
 

    #print(classification_report(y_true, y_pred))
    print()

# Note the problem is too easy: the hyperparameter plateau is too flat and the
# output model is the same for precision and recall with ties in quality.

print('\nmodelo de maior sensibilidade:')
# load the model from disk
loaded_model = pickle.load(open('model_sens.sav', 'rb'))
#result = loaded_model.score(X_test, y_test)
#print(result)
y_trueee, y_preddd = y_test, loaded_model.predict(X_test)
cnf_matrix = confusion_matrix(y_trueee, y_preddd) 
acc, sens = avaliacao(cnf_matrix,1)


print('\nmodelo de maior acuracia:')
loaded_model = pickle.load(open('model_acc.sav', 'rb'))
y_trueee, y_preddd = y_test, loaded_model.predict(X_test)
cnf_matrix = confusion_matrix(y_trueee, y_preddd) 
acc, sens = avaliacao(cnf_matrix,1)