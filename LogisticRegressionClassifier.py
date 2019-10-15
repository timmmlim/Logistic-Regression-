from math import *  
import pandas as pd  
import numpy as np
#source: https://zlatankr.github.io/posts/2017/03/06/mle-gradient-descent

class LogisticRegressionClassifier:
    def __init__(self):  
        pass 

    def sigmoid(self, X, beta):  
        return 1/(1 + np.exp(-np.dot(X, beta)))   

    def cost_function(self, beta, X, Y):   
        return -np.sum((Y * np.log(self.sigmoid(X, beta))) + ((1-Y) * np.log(1 - self.sigmoid(X, beta)))) 

    def gradient_descent(self, beta, X, Y, n = 100, tol = exp(-10), step = 0.01): 
        i = 0 
        change = 1
        while i < n and change > tol:    
            gradient = np.dot(X.transpose(), (Y - self.sigmoid(X, beta)))   
            beta_new = beta + (step * gradient)    
            change = abs(self.cost_function(beta, X, Y) - self.cost_function(beta_new, X, Y)) 
            beta = beta_new 

            if i % 10 == 0: 
                print('iter: ' + str(i) + ' cost: ' + str(self.cost_function(beta_new, X, Y))) 
                
            i += 1


        return beta   

    def fit(self, X, Y, n = 1000, tol = exp(-10), step = 0.01): 
        beta = np.array([0] * X.shape[1])

        beta = self.gradient_descent(beta, X, Y)
        return beta 
    
    def predict_probabilities(self, X_new): 
        probabilities = sigmoid(X_new) 
        return probabilities 
    
    def predit(self, X_new, threshold = 0.5): 
        probabilities = self.predict_probabilities(X_new) 
        classes = probabilities.map(lambda x: 0 if x <= threshold else 1) 
        return classes 


#test 
X = np.random.randn(100, 3) 
Y = np.random.randint(0, 2, 100)
ones = np.ones((100, 1))
Xb = np.concatenate((ones, X), axis=1) 

model = LogisticRegressionClassifier() 
print(model.fit(Xb, Y)) 

#compare with sklearn 
from sklearn.linear_model import LogisticRegression 

clf = LogisticRegression() 
clf.fit(Xb, Y) 
print(clf.coef_)
