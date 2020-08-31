import pandas as pd  
import numpy as np
# math reference: https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc

class LogisticRegressionClassifier:
    def __init__(self):
        pass 

    def sigmoid(self, X, beta):
        '''
        X is a matrix of shape (n, p)
        beta is a matrix of shape (p, 1)
        where n=no. of data points, p=no. of features
        '''
        return 1/(1 + np.exp(-np.dot(X, beta)))   

    def cost_function(self, beta, X, Y):   
        return -np.sum((Y * np.log(self.sigmoid(X, beta))) + ((1-Y) * np.log(1 - self.sigmoid(X, beta)))) 

    def gradient_descent(self, beta, X, Y, n=100, tol=np.exp(-10), step=0.01): 
        i = 0 
        change = 1
        for i in range(n):
            if change <= tol:
                break
            gradient = np.dot(X.transpose(), (Y - self.sigmoid(X, beta)))   
            beta_new = beta + (step * gradient)    
            change = abs(self.cost_function(beta, X, Y) - self.cost_function(beta_new, X, Y)) 
            beta = beta_new 
            
            if i % 10 == 0: 
                print(f'iter: {i}, cost: {self.cost_function(beta_new, X, Y)}') 

        return beta   

    def fit(self, X, Y, n=1000, tol=np.exp(-10), step=0.01): 
        '''
        update beta
        '''
        self.beta = np.array([0] * X.shape[1])
        self.beta = self.gradient_descent(self.beta, X, Y)
    
    def predict_probabilities(self, X_new): 
        probabilities = self.sigmoid(X_new, self.beta) 
        return probabilities 
    
    def predict(self, X_new, threshold=0.5): 
        probabilities = self.predict_probabilities(X_new) 
        classes = np.where(probabilities <= threshold, 0, 1)
        return classes 

