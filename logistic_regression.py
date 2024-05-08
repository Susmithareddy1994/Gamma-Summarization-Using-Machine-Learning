import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.model_selection import train_test_split
import sys
import warnings
import pickle
warnings.filterwarnings('ignore')


def read_data(file_name):
  data = pd.read_csv(file_name, chunksize=10000)
  data_df = pd.concat(data, ignore_index = True)
  return data_df


def compute_gamma(X):
  df_ones = pd.DataFrame(np.ones(X.shape[0]))
  X.insert(0, "N", df_ones, True)
  X_transposed = X.T
  gamma = X_transposed.dot(X)
  return gamma


def compute_pca(gamma, X):
  corr_matrix = gamma.corr()
  u, s, vh = np.linalg.svd(corr_matrix)
  sorted_index = np.argsort(s)[::-1]
  sorted_eigenvalue = s[sorted_index]
  sorted_eigenvectors = u[:,sorted_index]


  n_components = 2
  eigenvector_subset = sorted_eigenvectors[:,0:n_components]
  X_final = X.iloc[:, :-1]
  X_reduced = np.dot(X, eigenvector_subset)
  return X_reduced


def logistic_regression_train(X_train, Y_train):
  model = LogisticRegressionImpl(learning_rate=0.001, no_of_iterations=1000)
    
  model.fit(X_train,Y_train)
  return model
def logistic_regression_test(model, X_test):
  y_pred = model.predict(X_test)
  return y_pred



def compute_accuracy(y_pred, y_test):
  accuracy = 0
  for i in range(len(y_pred)):

    if y_pred[i] == y_test.iloc[i]:
      accuracy += 1
  #print(f"Accuracy = {accuracy / len(y_pred)}")
  return accuracy/len(y_pred)

def reduce_dimensions(X):


  gamma_val = compute_gamma(X)
  q = gamma_val.iloc[1:-1, 1:-1]
  X_reduced = compute_pca(q, X.iloc[:,1:-1])
  return X_reduced


def preprocess_data(X):
  X = X.dropna()
  Y = X.iloc[:, -1 ]
  X = X.iloc[:, 1:]
  return X, Y

class LogisticRegressionImpl():

  def __init__(self, learning_rate, no_of_iterations):

    self.learning_rate = learning_rate
    self.no_of_iterations = no_of_iterations

  def fit(self, X, Y):

    self.rows, self.cols = X.shape
    self.weights = np.zeros(self.cols)
    self.bias = 0
    self.X = X
    self.Y = Y
    for i in range(self.no_of_iterations):
      self.update_weights()



  def update_weights(self):

    #sigmoid function
    Y_cap = 1 / (1 + np.exp( - (self.X.dot(self.weights) + self.bias ) ))    
    dw = (1/self.rows)*np.dot(self.X.T, (Y_cap - self.Y))
    db = (1/self.rows)*np.sum(Y_cap - self.Y)
    self.weights = self.weights - self.learning_rate * dw
    self.bias = self.bias - self.learning_rate * db

  def predict(self, X):
    Z= 1 / (1 + np.exp( - (X.dot(self.weights) + self.bias ) )) 
    Y = np.where( Z > 0.5, 1, 0)
    return Y

def logisticRegression_chunks_training(file_name):
  accuracy = 0
  for chunk in pd.read_csv(file_name, chunksize=10000):
      X, Y = preprocess_data(chunk)
      gamma_val = compute_gamma(X)
      q = gamma_val.iloc[1:-1, 1:-1]
      X_reduced = compute_pca(q, X.iloc[:,1:-1])
      
      X_train,X_test,Y_train,Y_test = train_test_split(X_reduced,Y,test_size=0.3,random_state=0)
      model = logistic_regression_train(X_train, Y_train)
      y_pred = logistic_regression_test(model, X_test)
      local_accuracy = compute_accuracy(y_pred, Y_test)
      if accuracy < local_accuracy:
        accuracy = local_accuracy
        filename =  "lr_impl.sav"
        pickle.dump(model, open(filename, 'wb'))

  return accuracy

def logisticRegression_chunks_testing(file_name):
  accuracy = 0
  i = 0
  print("Testing...")
  for chunk in pd.read_csv(file_name, chunksize=10000):
      i = i+1
      if i%20 == 0:
        print(i, "% tested")
      X, Y = preprocess_data(chunk)
      gamma_val = compute_gamma(X)
      q = gamma_val.iloc[1:-1, 1:-1]
      X_reduced = compute_pca(q, X.iloc[:,1:-1])
      
      model = pickle.load(open('lr_impl_100k.sav', 'rb'))
      y_pred = model.predict(X_reduced)
      local_accuracy = compute_accuracy(y_pred, Y)
      accuracy += local_accuracy
      #print(local_accuracy)

  return accuracy/i


def logisticRegression_libraries_traintest(file_name):
  accuracy = 0
  for chunk in pd.read_csv(file_name, chunksize=10000):
      X, Y = preprocess_data(chunk)
      gamma_val = compute_gamma(X)
      q = gamma_val.iloc[1:-1, 1:-1]
      X_reduced = compute_pca(q, X.iloc[:,1:-1])
      
      
      X_train,X_test,Y_train,Y_test = train_test_split(X_reduced,Y,test_size=0.3,random_state=0)
      model = LogisticRegression()
      model.fit(X_train, Y_train)
      y_pred = model.predict(X_test)
      local_accuracy = compute_accuracy(y_pred, Y_test)
      if accuracy < local_accuracy:
        accuracy = local_accuracy
        filename =  "lr_impl_lib_100k.sav"
        pickle.dump(model, open(filename, 'wb'))

  return accuracy

def logisticRegression_libraries_testing(file_name):
  accuracy = 0
  i = 0
  print("Testing...")
  for chunk in pd.read_csv(file_name, chunksize=10000):
      i = i+1
      if i%20 == 0:
        print(i, "% tested")
      X, Y = preprocess_data(chunk)
      gamma_val = compute_gamma(X)
      q = gamma_val.iloc[1:-1, 1:-1]
      X_reduced = compute_pca(q, X.iloc[:,1:-1])
      
      model = pickle.load(open('lr_impl_lib_100k.sav', 'rb'))
      y_pred = model.predict(X_reduced)
      local_accuracy = compute_accuracy(y_pred, Y)
      accuracy += local_accuracy
      #print(local_accuracy)

  return accuracy/i

def svm(file_name):
  from sklearn import svm
  i=0
  accuracy =0
  for chunk in pd.read_csv(file_name, chunksize=10000):
      i = i+1
      if i%20 == 0:
        print(i, "% tested")
      X, Y = preprocess_data(chunk)
      gamma_val = compute_gamma(X)
      q = gamma_val.iloc[1:-1, 1:-1]
      X_reduced = compute_pca(q, X.iloc[:,1:-1])
      
      classifier = svm.SVC(kernel= 'rbf')
      X_train,X_test,Y_train,Y_test = train_test_split(X_reduced,Y,test_size=0.3,random_state=0)
      classifier.fit(X_train,Y_train)
      prediction = classifier.predict(X_test)
      local_accuracy = compute_accuracy(prediction, Y_test)
      accuracy += local_accuracy
      print(local_accuracy)
  
      print(accuracy_score(Y_test,prediction))
      print(classification_report(Y_test,prediction))
      print(confusion_matrix(Y_test,prediction))


task = int(sys.argv[1])
filename = sys.argv[2]

#Task 1: LogisticRegression Training and Testing without libraries => accuracy percentage, precision, recall and confusion matrix
#Task 2: LogisticRegression Testing without libraries => accuracy percentage, precision, recall and confusion matrix
#Task 3: LogisticRegression Training and Testing using libraries => accuracy percentage and confusion matrix
#Task 4: LogisticRegression Testing using libraries => accuracy percentage and confusion matrix
#Task 5: SVM Training and Testing using libraries => accuracy percentage and confusion matrix
task_description = """Task 1: LogisticRegression Training and Testing without libraries => accuracy percentage, precision, recall and confusion matrix \n
Task 2: LogisticRegression Testing without libraries => accuracy percentage, precision, recall and confusion matrix \n
Task 3: LogisticRegression Training and Testing using libraries => accuracy percentage and confusion matrix \n
Task 4: LogisticRegression Testing using libraries => accuracy percentage and confusion matrix \n
Task 5: SVM Training and Testing using libraries => accuracy percentage and confusion matrix \n
"""
print(task_description)
#X = read_data(filename)

if task == 1:
  accuracy = logisticRegression_chunks_training(filename)
  print("Accuracy - Logistic Regression without Libraries - ", accuracy*100 , "%")
  #X, Y = preprocess_data(X)
  #X_reduced = reduce_dimensions(X)
  #X_train,X_test,Y_train,Y_test = train_test_split(X_reduced,Y,test_size=0.3,random_state=0)
  #model = logistic_regression_train(X_train, Y_train)
  #y_pred = logistic_regression_test(model, X_test)
  #print(y_pred)
  #compute_accuracy(y_pred, Y_test)
  #print("Accuracy - Task 1- ", accuracy)

elif task == 2:
  accuracy = logisticRegression_chunks_testing(filename)
  print("Accuracy - Logistic Regression Testing without Libraries - ", accuracy*100 , "%")

elif task == 3:
  accuracy = logisticRegression_libraries_traintest(filename)
  print("Accuracy - Logistic Regression with Libraries - ", accuracy*100 , "%")

elif task == 4:
  accuracy = logisticRegression_libraries_testing(filename)
  print("Accuracy - Logistic Regression Testing with Libraries - ", accuracy*100 , "%")

elif task == 5:
  svm(filename)
else:
  print("Invalid task number")
  print(task_description)


