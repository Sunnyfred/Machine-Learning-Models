import numpy as np 
import pandas as pd 
import math as m
import matplotlib.pyplot as plt 

def train_test_split_po(X, Y, train_size, shuffle):
    ''' Perform tran/test datasets splitting '''
    if shuffle:
        randomize = np.arange(len(X))
        np.random.shuffle(randomize)
        X = X[randomize]
        Y = Y[randomize]
    s_id = int(len(Y) * train_size)
    X_train, X_test = X[:s_id], X[s_id:]
    Y_train, Y_test = Y[:s_id], Y[s_id:]
    Y_train = Y_train.reshape((-1, 1))
    X_train1 = np.append(X_train, Y_train, axis = 1) 
    X_train1 = X_train1[np.argsort(X_train1[:, 0])]
    Y_test = Y_test.reshape((-1, 1))
    X_test1 = np.append(X_test, Y_test, axis = 1) 
    X_test1 = X_test1[np.argsort(X_test1[:, 0])]
    X_train, X_test = X_train1[:,:-1], X_test1[:,:-1]
    Y_train, Y_test = X_train1[:,-1], X_test1[:,-1]
    Y_train=np.squeeze(Y_train)
    Y_test=np.squeeze(Y_test)
    return X_train, X_test, Y_train, Y_test    


def metric_mse(Y_label, Y_pred):
    ''' Evaluate mean squared error (MSE) '''
    return np.mean(np.power(Y_label - Y_pred, 2))

def metric_rmse(Y_label, Y_pred):
    ''' Evaluate root mean squared error (RMSE) '''
    return m.sqrt(np.mean(np.power(Y_label - Y_pred, 2)))

def readin_data(path):
    df = pd.read_csv(path) 
    X = df.iloc[:,:-1].values 
    Y = df.iloc[:,-1].values 
    return X, Y
    
def generate_dataset_simple(beta, n, std_dev):
    ''' Generate dataset '''
    X = np.random.rand(n)
    e = np.random.randn(n) * std_dev
    Y = X * beta + e
    X = X.reshape((n,1))
    return X, Y  

def generate_dataset_polynomial(beta, n, std_dev):
    ''' Generate polynomial dataset '''
    
    e = np.random.randn(n) * std_dev/n
    X = np.random.random_sample(n)
    X = np.sort(X)
    Y = 1- 6*X +36*X**2 - 53*X**3 + 22*X**5 + e
    X = X.reshape((n,1))
    return X, Y 


def standardization(X,degree):
    """ A scaling technique where the values
    are centered around the mean with 
    a unit standard deviation. 
    This means that the mean of the attribute 
    becomes zero and the resultant distribution 
    has a unit standard deviation. 
    ----------------------------------------
    degree: polynomial regression degree
    """
    X[:, :(degree)] = (X[:, :(degree)] - np.mean(X[:, :(degree)], axis = 0))/ \
    np.std(X[:, :(degree)], axis = 0)
    return X 

def normalization(X,degree):
    """ A scaling technique in which values 
    are shifted and rescaled so that they 
    end up ranging between 0 and 1. 
    It is also known as Min-Max scaling 
    ----------------------------------------
    degree: polynomial regression degree
    """
    X[:, :(degree)] = (X[:, :(degree)] - np.amin(X[:, :(degree)], axis = 0))/ \
    (np.amax(X[:, :(degree)], axis = 0) - np.amin(X[:, :(degree)], axis = 0))
    return X 


def transformation(m, X, degree):
    tmp = np.zeros([m, 1])
    for j in range(degree + 1):
        if j != 0:
            x_pow = np.power(X, j) 
            tmp = np.append(tmp, x_pow.reshape(-1, 1), axis = 1) 
    tmp = np.append(tmp, np.ones((m, 1)), axis = 1) 
    Xt=tmp[:,1:]
    return Xt

def transformation_predict(m, X, degree):
    tmp = np.zeros([m, 1])
    for j in range(degree + 1):
        if j != 0:
            x_pow = np.power(X, j) 
            tmp = np.append(tmp, x_pow.reshape(-1, 1), axis = 1) 
    tmp = np.append(tmp, np.ones((m, 1)), axis = 1) 
    Xt=tmp[:,1:-1]
    return Xt

class PolynomialRegression(): 
    ''' Univariate Polynomial Regression model. 
        Used to obtain the relationship between dependent variable and independent variables.
        -------------------------------------------------------------------------------------
        degree shall be great equal 2
        '''
    
    def __init__(self, iterations, learning_rate, degree):   
        self.lr = learning_rate 
        self.it = iterations 
        self.de = degree 
             
    def fit(self, X, Y): 
        # m instances, d atrributes 
        self.m, self.d = X.shape 
        # weight initialization 
        self.W = np.zeros(self.de+1) 
        self.X = X 
        self.Xt = transformation(self.m, self.X, self.de)
        self.Xs = standardization(self.Xt, self.de)
        self.Y = Y      
        for i in range(self.it):   
            self.update_weights() 
        return self
    
    def update_weights(self): 
        Y_pred = self.predict(self.X) 
        # calculate gradients   
        dW = (self.Xs.T).dot(Y_pred - self.Y)/self.m  
        # update weights 
        self.W = self.W - self.lr * dW 
        return self
       
    def predict(self, X):
        self.Xt = transformation(X.shape[0], X, self.de)
        self.Xs = standardization(self.Xt, self.de)
        return self.Xs.dot(self.W)
  
def main(): 
    # Import data
    X, Y = generate_dataset_polynomial(10, 200, 20)
    # Splitting dataset into train and test set 
    X_train, X_test, Y_train, Y_test = train_test_split_po(X, Y, train_size=.5, shuffle=True)
    # Model Learning
    model = PolynomialRegression(learning_rate = 0.2, iterations = 20000, degree=5) 
    model.fit(X_train, Y_train) 
    print(X_test.shape)
    Y_pred = model.predict(X_test)   
    print(Y_test.shape) 
    print(Y_pred.shape)
    # Statistics
    mse = metric_mse(Y_test, Y_pred)
    rmse = metric_rmse(Y_test, Y_pred)
    print('Coefficients: ', 'W = ', model.W[-1], model.W[:-1]) 
    print('MSE = ', mse)    
    print('RMSE = ', rmse)
    # Visualization
    l1=plt.scatter(X_train, Y_train, color = 'black', s=8) 
    l2=plt.scatter(X_test, Y_test, color = 'green', s=8) 
    l3=plt.plot( X_test, Y_pred, color = 'red', linewidth=3) 
    plt.title( 'X v.s. Y') 
    plt.xlabel( 'X') 
    plt.ylabel( 'Y') 
    plt.legend((l1, l2),('Train Points', 'Test Points'),loc='lower left', prop={'size': 20})
    plt.show()
    
if __name__ == '__main__':  
    main()