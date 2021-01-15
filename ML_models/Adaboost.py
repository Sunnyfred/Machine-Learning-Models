import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from Decision_tree_model import DecisionTree

def metric_accuracy(Y_label, Y_pred):
    '''Evaluate the accuracy'''
    correct_amount = 0 
    for i in range(np.size(Y_pred)) :   #np.size: Number of elements in the array
        if Y_label[i] == Y_pred[i] :             
            correct_amount = correct_amount + 1
    return correct_amount / np.size(Y_pred) * 100

class AdaBoost():
    '''Adaptive Boosting: The output of the weak learner algorithms 
    (Decision tree for current model) is combined into a weighted sum 
    that represents the final output of the boosted classifier.
    AdaBoost is adaptive in the sense that subsequent weak learners 
    are tweaked in favor of those instances misclassified by previous 
    classifiers. 
    ------------------------------------
    Input data structure: numpy array with m x (d+1) shape, 
                          m rows of samples included, 
                          d columns of features/ttributes,
                          and 1 column of target
    n_estimators: number of training round '''
    
    def __init__(self, n_estimators):
        self.ne = n_estimators
        
    def fit(self, X):
        
        self.models = list()
        self.decision_stump = list()
        self.alpha = list()
        
        # initialize the distribution of sample weight
        w = np.full(X.shape[0], 1/X.shape[0])
        for i in range(self.ne):
            # Decision tree base learner 
            bl = DecisionTree(weight = w)
            decision_stump = bl.fit(X)
            y_pred = bl.decisionstump_prediction(decision_stump, X)
            y_label = X[:, -1].astype(int)
            # Estimate base learner weights
            alpha = .5 * np.log((1.0 - decision_stump['traning_error'] ) \
                                 / (decision_stump['traning_error'] + 1e-10))
            #Update distribution of sample weight to focus on misclassified samples
            w *= np.exp(-alpha * y_label * y_pred)
            w /= np.sum(w)
            self.models.append(bl)
            self.alpha.append(alpha)
            self.decision_stump.append(decision_stump) 
            
        return self
            
    def predict(self, X):
        '''weighted averaging'''
        return np.sign(np.sum([self.alpha[i] * self.models[i].decisionstump_prediction(self.decision_stump[i],X) for i in range(len(self.models))], axis = 0))
        
    def predict_split(self, X, i):
        return np.sign(np.array([self.alpha[i] * self.models[i].decisionstump_prediction(self.decision_stump[i],X)]))        
            
def plot_classification_contour(X, Y, model):
    '''Utility function to plot decision boundary and scatter plot of data'''
    n=50
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid( np.linspace(x_min, x_max, n), np.linspace(y_min, y_max, n))
    zz = np.array( [ model.predict(np.array([xi,yi]).reshape(1,-1)) for  xi, yi in zip(np.ravel(xx), np.ravel(yy)) ] )           
    Z = zz.reshape(xx.shape)
    ax = plt.gca()
    ax.contourf(xx, yy, Z, 2, cmap= "RdBu_r", alpha=.3)
    ax.contour(xx, yy, Z,  2, cmap= "RdBu_r")
    ax.scatter(X[:,0],X[:,1], c = Y, cmap =  ListedColormap(['r', 'b']), s=5)
    ax.set_xlabel('$X_1$')
    ax.set_ylabel('$X_2$')
    
    
def plot_classification_contour_split(X, Y, model, co):
    '''Utility function to plot decision boundary and scatter plot of data'''
    n=30
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid( np.linspace(x_min, x_max, n), np.linspace(y_min, y_max, n))
    zz = np.array( [ model.predict_split(np.array([xi,yi]).reshape(1,-1), co) for  xi, yi in zip(np.ravel(xx), np.ravel(yy)) ] )        
    Z = zz.reshape(xx.shape)
    ax = plt.gca()
    ax.contourf(xx, yy, Z, 2, cmap= "RdBu_r", alpha=.3)
    ax.contour(xx, yy, Z,  2, cmap= "RdBu_r")
    ax.scatter(X[:,0],X[:,1], c = Y, cmap = ListedColormap(['r', 'b']), s=5)
    ax.set_xlabel('$X_1$')
    ax.set_ylabel('$X_2$')
    
from sklearn import datasets
from sklearn.model_selection import train_test_split

def main():
    
    data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    y[y == 0] = -1

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=5)
    X_trainset = np.hstack((X_train[:, :2], Y_train.reshape(-1,1)[:, :]))
    X_testset = np.hstack((X_test[:, :2], Y_test.reshape(-1,1)[:, :]))

    n_estimators = 6
    model = AdaBoost(n_estimators)
    model.fit(X_trainset)
    preds = model.predict(X_testset)
    y_label = X_testset[:,-1]
    print('Accuracy = ', np.sum(preds == y_label) / X_testset.shape[0])
    
    fig = plt.figure(figsize = (18,8))
    for m in range(n_estimators):
        fig.add_subplot(int(n_estimators/3),3,m+1)
        #s_weights = (sample_weight_list[m,:] / sample_weight_list[m,:].sum() ) * 40
        plot_classification_contour_split(X, y, model, m)
        plt.title('Decision Boundary for weak learner_{}'.format(m))
    
    fig = plt.figure(figsize = (6,4))
    plot_classification_contour(X, y, model)
    plt.title('Decision Boundary for strong learner')
    
if __name__=='__main__':
    main()
    