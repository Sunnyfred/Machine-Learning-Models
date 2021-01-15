import numpy as np

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, c=1, degree=3):
    '''degree: great equal 1'''
    return (c + np.dot(x, y)) ** degree

def gaussian_kernel(x, y, sigma=2):
    '''sigma: great than 0'''
    gamma=1/2/(sigma ** 2)
    return np.exp(-np.linalg.norm(x-y)**2 *gamma)

def laplace_kernel(x, y, sigma):
    '''sigma: great than 0'''
    return np.exp(-np.linalg.norm(x-y) / sigma)

def sigmoid_kernel(x, y, beta, theta):
    '''beta: great than 0
       theta: less than 0'''
    return np.tanh(beta*np.dot(x, y) + theta)

import cvxopt
import cvxopt.solvers
import matplotlib.pyplot as plt
import math as m

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
    
    
def generate_dataset_MVND_ls():
    '''generate a dataset that satisfies the multivariate normal distribution
       and the dataset is linearly sperable'''
    np.random.seed(24) # Fixing random state for reproducibility
    num_observations = 200
    X1 = np.random.multivariate_normal([0, 2], [[0.8, 0.6], [0.6, 0.8]], num_observations)
    X2 = np.random.multivariate_normal([2, 0], [[0.8, 0.6], [0.6, 0.8]], num_observations)
    X = np.vstack((X1, X2)).astype(np.float32)
    Y = np.hstack((np.ones(len(X1)),np.ones(len(X2))*-1))
    return X, Y

def generate_dataset_MVND_nls():
    '''generate a dataset that satisfies the multivariate normal distribution
       and the dataset is NOT linearly sperable'''
    np.random.seed(24) # Fixing random state for reproducibility
    num_observations = 100
    X1 = np.random.multivariate_normal([-1, 2], [[1.0,0.8], [0.8, 1.0]], num_observations)
    X1 = np.vstack((X1, np.random.multivariate_normal([4, -4], [[1.0,0.8], [0.8, 1.0]], num_observations)))
    X2 = np.random.multivariate_normal([1, -1], [[1.0,0.8], [0.8, 1.0]], num_observations)
    X2 = np.vstack((X2, np.random.multivariate_normal([-4, 4], [[1.0,0.8], [0.8, 1.0]], num_observations)))
    X = np.vstack((X1, X2)).astype(np.float32)
    Y = np.hstack((np.ones(len(X1)),np.ones(len(X2))*-1))
    return X, Y

def generate_dataset_MVND_lso():
    '''generate a dataset that satisfies the multivariate normal distribution
       and the dataset is linearly sperable but overlapping'''
    np.random.seed(24) # Fixing random state for reproducibility
    num_observations = 100
    X1 = np.random.multivariate_normal([0, 2], [[1.5, 1.0], [1.0, 1.5]], num_observations)
    X2 = np.random.multivariate_normal([2, 0], [[1.5, 1.0], [1.0, 1.5]], num_observations)
    X = np.vstack((X1, X2)).astype(np.float32)
    Y = np.hstack((np.ones(len(X1)),np.ones(len(X2))*-1))
    return X, Y

def generate_dataset_MVND_nlso2():
    '''generate a dataset that satisfies the multivariate normal distribution
       and the dataset is NOT linearly sperable but overlapping'''
    np.random.seed(24) # Fixing random state for reproducibility
    num_observations = 100
    radius1 = np.sqrt(np.array(np.random.uniform(0,1,num_observations))).reshape(-1,1)
    angle1 = 2*m.pi*np.array(np.random.uniform(0,1,num_observations)).reshape(-1,1)
    X1 = np.array(radius1*np.cos(angle1)).reshape(-1,1)
    X2 = np.array(radius1*np.sin(angle1)).reshape(-1,1)
    X_ = np.hstack((X1, X2))
    radius2 = np.sqrt(np.array(3.* np.random.uniform(0,1,num_observations) + .9)).reshape(-1,1)
    X3 = np.array(radius2*np.cos(angle1)).reshape(-1,1)
    X4 = np.array(radius2*np.sin(angle1)).reshape(-1,1)
    X_X = np.hstack((X3, X4))
    X = np.vstack((X_, X_X))
    Y = np.hstack((np.ones(len(X_)),np.ones(len(X_X))*-1))
    return X, Y

def generate_dataset_MVND_nls2():
    '''generate a dataset that satisfies the multivariate normal distribution
       and the dataset is NOT linearly sperable'''
    np.random.seed(24) # Fixing random state for reproducibility
    num_observations = 100
    radius1 = np.sqrt(np.array(np.random.uniform(0,1,num_observations))).reshape(-1,1)
    angle1 = 2*m.pi*np.array(np.random.uniform(0,1,num_observations)).reshape(-1,1)
    X1 = np.array(radius1*np.cos(angle1)).reshape(-1,1)
    X2 = np.array(radius1*np.sin(angle1)).reshape(-1,1)
    X_ = np.hstack((X1, X2))
    radius2 = np.sqrt(np.array(3.* np.random.uniform(0,1,num_observations) + 1.5)).reshape(-1,1)
    X3 = np.array(radius2*np.cos(angle1)).reshape(-1,1)
    X4 = np.array(radius2*np.sin(angle1)).reshape(-1,1)
    X_X = np.hstack((X3, X4))
    X = np.vstack((X_, X_X))
    Y = np.hstack((np.ones(len(X_)),np.ones(len(X_X))*-1))
    return X, Y

def metric_accuracy_count(Y_label, Y_pred):
    '''Evaluate the accuracy'''
    correct_amount = 0 
    for i in range(np.size(Y_pred)) :   #np.size: Number of elements in the array
        if Y_label[i] == Y_pred[i] :             
            correct_amount = correct_amount + 1
    return correct_amount 



def plot_contour(X1_train, X2_train, model):
    plt.figure(figsize = (12, 8))
    plt.scatter(X1_train[:,0], X1_train[:,1], s=50, c="r", cmap=plt.cm.jet, marker = '+', label = 'Positive ')
    plt.scatter(X2_train[:,0], X2_train[:,1], s=50, c="b", cmap=plt.cm.jet, marker = '_', label = 'Negative ')
    plt.scatter(model.support_vector[:,0], model.support_vector[:,1], s=120, facecolors='none', edgecolors='g', label = 'Support Vector')

    X1, X2 = np.meshgrid(np.linspace(-6,6,50), np.linspace(-6,6,50))
    X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
    Z = model.predict(X).reshape(X1.shape)
    plt.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
    plt.contour(X1, X2, Z + 1, [0.0], colors='grey', linestyles = 'dashed', linewidths=1, origin='lower')
    plt.contour(X1, X2, Z - 1, [0.0], colors='grey', linestyles = 'dashed', linewidths=1, origin='lower')

    plt.show()





class SVM():
    '''Suppor Vector Machine model. 
       Used to find the decision boundary/hyperplane 
       to separate the two-group classification problem 
       ------------------------------------------------
       kernel: kernel function of kernel trick, 
               designed for cases that are not linearly separable
       C: penality constant (inverse of regularization constant, 1/lambda). 
               It means how much you want to avoid misclassifying each sample.
               If large C is used, then the model tries to make sure all samples
               satisfy the constrains, namely, a smaller-margin hyperplane is 
               obtained even if the samples are all correctly classified, and 
               vise versa.'''
    
    def __init__(self, kernel, C=None):
        self.kernel = kernel
        self.C = C
        
    def fit(self, X, y):
        # m instances, d atrributes 
        self.m, self.d = X.shape 
        # Kernel matrix
        K = np.zeros((self.m, self.m))
        for i in range(self.m):
            for j in range(self.m):
                K[i,j] = self.kernel(X[i], X[j])
        # The Convex Quadratic Programming parameters (https://cvxopt.org/examples/tutorial/qp.html)
        # QP Problem like: min_x 0.5*x^T P x + q^T x, s.t. G x <= h and A x = b. 
        # May use list or np.array to create matrix. Note that if use list, 
        # matrixs should be defined by columns instead of rows
        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(self.m) * -1)
        A = cvxopt.matrix(y, (1, self.m))
        b = cvxopt.matrix(0.0)
        if not self.C:
            G = cvxopt.matrix(np.diag(np.ones(self.m) * -1))
            h = cvxopt.matrix(np.zeros(self.m))
        else:
            G1 = np.diag(np.ones(self.m) * -1)
            G2 = np.identity(self.m)
            G = cvxopt.matrix(np.vstack((G1, G2)))
            h1 = np.zeros(self.m)
            h2 = np.ones(self.m) * self.C
            h = cvxopt.matrix(np.hstack((h1, h2)))

        # Construct the QP
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        # Lagrange multipliers
        l_m = np.ravel(sol['x'])

        # The model, in its formation, is:　f(x) = w^T x + b = (sum_i^m l_m_i y_i K（x_i^T, x) + b 
        # if l_m_i = 0, it will not exist in the above summation, and will have no influence on f(x),
        # only consider cases that have l_m_i > 0. Based on KKT condition, if l_m_i > 0, y_if(x_i) = 1.
        # That means these samples are right on the large margins and they are support vectors. 
        # Only the support vectors are important in training the model f(x).
        support_vector_id = l_m > 1e-6
        new_id = np.arange(len(l_m))[support_vector_id]
        store_id = [i for i in range(len(l_m)) if support_vector_id[i]==True]
        store_id = np.array(store_id)
        self.l_m = l_m[support_vector_id]
        self.support_vector = X[support_vector_id]
        self.support_vector_label = y[support_vector_id]
        
        # Estimate the hyperplane intercept
        self.b = 0
        for i in range(len(self.l_m)):
            self.b += self.support_vector_label[i] / len(self.l_m)
            self.b -= np.sum(self.l_m * self.support_vector_label * K[store_id[i],support_vector_id]) / len(self.l_m)
        
        
    def predict(self, X):
        self.y_pred = np.zeros(len(X))
        for i in range(len(X)):
            tmp = 0
            for j in range(len(self.l_m)):
                tmp = tmp + self.l_m[j] * self.support_vector_label[j] * self.kernel(X[i], self.support_vector[j])
            self.y_pred[i] = tmp + self.b
        return self.y_pred
    
    
def main():   
    
    # Import data
    X, Y = generate_dataset_MVND_ls()
    print(X.shape, Y.shape)
    # Splitting dataset into train and test set 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=.9, shuffle=False)
    print( X_train.shape, Y_train.shape)
    print( X_test.shape, Y_test.shape)
    # Model Learning
    model = SVM(linear_kernel)
    model.fit(X_train, Y_train) 
    # Model Working
    Y_pred = model.predict(X_test) 
    Y_pred = np.sign(Y_pred)
    #Statistics
    #print( 'Accuracy count: ', metric_accuracy_count(Y_test, Y_pred), ' out of ', len(Y_test), ' are correct!' ) 
    #Visulaization
    plt.figure(figsize = (12, 8))
    c1 = np.ma.masked_where(Y == -1, Y)
    c2 = np.ma.masked_where(Y == 1, Y)
    plt.scatter(X[:,0], X[:,1], s=50, c=c1, cmap="RdBu_r", marker = '+', label = 'Positive ')
    plt.scatter(X[:,0], X[:,1], s=50, c=c2, cmap="RdBu_r", marker = '_', label = 'Negative ')
    plt.scatter(model.support_vector[:,0], model.support_vector[:,1], s=120, facecolors='none', edgecolors='c', label = 'Support Vectors')       
    X1, X2 = np.meshgrid(np.linspace(-3,5,10), np.linspace(-3,5,10))
    X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
    Y = model.predict(X).reshape(X1.shape)
    plt.contour(X1, X2, Y, [0], colors='k', linewidths=3) # W^t X = 0
    plt.contour(X1, X2, Y - 1, [0], colors='k', linestyles = 'dashed', linewidths=1) # W^t X = 1
    plt.contour(X1, X2, Y + 1, [0], colors='k', linestyles = 'dashed', linewidths=1) # W^t X = -1
    
    plt.legend()
    plt.show()
        
if __name__ == "__main__":
    main()
