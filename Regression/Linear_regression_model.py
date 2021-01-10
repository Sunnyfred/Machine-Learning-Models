import numpy as np 
import pandas as pd 
import math as m
import matplotlib.pyplot as plt 

def train_test_split(X, Y, train_size, shuffle):
    ''' Perform tran/test datasets splitting '''
    if shuffle:
        randomize = np.arange(len(X))
        np.random.shuffle(randomize)
        X = X[randomize]
        Y = Y[randomize]
    s_id = int(len(Y) * train_size)
    X_train, X_test = X[:s_id], X[s_id:]
    Y_train, Y_test = Y[:s_id], Y[s_id:]

    return X_train, X_test, Y_train, Y_test    


def metric_mse(Y_label, Y_pred):
    ''' Evaluate mean squared error (MSE) '''
    return np.mean(np.power(Y_label - Y_pred, 2))

def metric_rmse(Y_label, Y_pred):
    ''' Evaluate root mean squared error (RMSE) '''
    return m.sqrt(np.mean(np.power(Y_label - Y_pred, 2)))

def readin_data(path):
    ''' Evaluate root mean squared error (RMSE) '''
    df = pd.read_csv(path) 
    X = df.iloc[:,:-1].values 
    Y = df.iloc[:,1].values 
    return X, Y
    
def generate_dataset_simple(beta, n, std_dev):
    ''' Generate dataset '''
    X = np.random.rand(n)
    e = np.random.randn(n) * std_dev
    Y = X * beta + e
    X = X.reshape((n,1))
    return X, Y    

class LinearRegression() : 
    ''' Linear Regression model. 
        Used to obtain the relationship between dependent variable and independent variables.'''
    def __init__(self, iterations, learning_rate):   
        self.lr = learning_rate 
        self.it = iterations 
             
    def fit(self, X, Y): 
        # m instances, d atrributes 
        self.m, self.d = X.shape 
        # weight initialization 
        self.W = np.zeros(self.d+1) 
        self.X = X 
        self.XX = np.ones((self.m, self.d+1)) 
        self.XX[:,:-1] = self.X
        self.Y = Y      
        for i in range(self.it):   
            self.update_weights() 
        return self
      
    def update_weights(self): 
        Y_pred = self.predict(self.XX) 
        # calculate gradients   
        dW = (self.XX.T).dot(Y_pred - self.Y)/self.m  
        # update weights 
        self.W = self.W - self.lr * dW 
        return self
       
    def predict(self, X): 
        return X.dot(self.W)
  
def main(): 
    # Import data
    X, Y = generate_dataset_simple(10, 200, 0.5)
    # Splitting dataset into train and test set 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=.5, shuffle=False)
    # Model Learning
    model = LinearRegression(learning_rate = 0.01, iterations = 15000) 
    model.fit(X_train, Y_train) 
    # Model Working
    M, D = X_test.shape
    TEST = np.ones((M, D+1)) 
    TEST[:,:-1] = X_test
    Y_pred = model.predict(TEST)  
    # Statistics
    mse = metric_mse(Y_test, Y_pred)
    rmse = metric_rmse(Y_test, Y_pred)
    print('Coefficients: ', 'W = ', model.W[:-1], ', b = ', model.W[-1]) 
    print('MSE = ', mse)    
    print('RMSE = ', rmse)
    # Visualization
    plt.scatter( X_test, Y_test, color = 'black', s=8) 
    plt.plot( X_test, Y_pred, color = 'red', linewidth=3) 
    plt.title( 'X_test v.s. Y_test') 
    plt.xlabel( 'X_test') 
    plt.ylabel( 'Y_test') 
    X_actual = np.array([0, 1])
    Y_actual = X_actual*10
    plt.plot(X_actual, Y_actual, 'c--', linewidth=3) 
    plt.legend(('Regression Line', 'Actual Line'),loc='upper left', prop={'size': 15})
    plt.show()
    
if __name__ == '__main__':  
    main()