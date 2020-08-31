import numpy as np
import os
from pathlib import Path

wdir = Path(r'C:\Users\Timothy Lim\Documents\Logistic-Regression-')
os.chdir(wdir)

from LogisticRegressionClassifier import LogisticRegressionClassifier

# test case
np.random.seed(1)
n = 100
x = np.random.randn(n, 2)
beta = np.array([0.5, 0.3])
sigmoid_result = 1 / (1 + np.exp(-np.dot(x, beta)))
y = np.where(sigmoid_result <= 0.5, 0, 1)

LR_classifier = LogisticRegressionClassifier()
LR_classifier.fit(x, y)
beta_pred = LR_classifier.beta

# check yhat
yhat = LR_classifier.predict(x)
accuracy = np.sum(yhat == y) / n

# check beta values
print(beta_pred)
print(accuracy)
