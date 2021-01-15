import numpy as np
import matplotlib.pyplot as plt
import math as m 
import collections
import pandas as pd
import sys 
import time


def readin_csv_data(path):
    df = pd.read_csv(path) 
    X = df.iloc[:,:-1].values 
    Y = df.iloc[:,-1].values 
    return X, Y

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

def metric_accuracy(Y_label, Y_pred):
    '''Evaluate the accuracy'''
    correct_amount = 0 
    for i in range(np.size(Y_pred)) :   #np.size: Number of elements in the array
        if Y_label[i] == Y_pred[i] :             
            correct_amount = correct_amount + 1
    return correct_amount / np.size(Y_pred) * 100



class RandomForestClassification():
    '''Random Forest model for classification
    ------------------------------------
    Input data structure: numpy array with m x (d+1) shape, 
                          m rows of samples included, 
                          d columns of features/ttributes,
                          and 1 column of target
    criterion: gini, entropy or mse
    max_depth: max depth for each tree
    min_samples: min samples to stop splitting
    max_features: the max numbers of features considered in decision tree
    n_estimators: how many trees generated'''
    
    
    def __init__(self, max_depth, min_samples, max_features, criterion, n_estimators):
        self.md = max_depth
        self.ms = min_samples
        self.mf = max_features
        self.depth_init = 1
        self.criterion = criterion
        self.ne = n_estimators
    
    def evaluate_information_entropy(self, p):
        '''Evaluate information entropy component'''
        if p==0:
            return 0
        elif p==1:
            return 0
        else:
            return -p*np.log2(p)
        
    def evaluate_information_gain(self, left, right, classes):
        '''Evaluate information gain'''
        IG_after = 0
        IG_before = 0
        size_l = left.shape[0]
        size_r = right.shape[0]
        size_t = size_l +  size_r # Total sample No.
        lr = np.vstack((left, right))
        statis_lr = collections.Counter(lr[:,-1])
        IG_before = sum([self.evaluate_information_entropy(float(statis_lr[class_i])/float(size_t)) for class_i in classes])
        spaces = [left, right]
        for space in spaces:
            size = space.shape[0] # Sample No. in different group
            if size == 0:   # Avoid 0 in denominator
                continue     
            tmp = 0.
            statis = collections.Counter(space[:,-1])
            tmp=sum([self.evaluate_information_entropy(float(statis[class_i])/float(size)) for class_i in classes])
            IG_after += tmp * (size / size_t)
        return IG_before - IG_after         
       
    def evaluate_gini_index(self, left, right, classes):
        '''Gini impurity for classification'''
        gini = 0
        size_l = left.shape[0]
        size_r = right.shape[0]
        size_t = size_l +  size_r # Total sample No.
        spaces = [left, right]
        for space in spaces:
            size = space.shape[0] # Sample No. in different group
            if size == 0:   # Avoid 0 in denominator
                continue     
            tmp = 0.
            statis = collections.Counter(space[:,-1])
            tmp=sum([(float(statis[class_i])/float(size))**2.0 for class_i in classes])
            gini += (1.0 - tmp) * (size / size_t)
        return gini    
    
    def evaluate_mse_index(self, left, right):
        '''MSE index for regression'''
        mse = 0
        spaces = [left, right]
        for space in spaces:
            mse += np.mean((space[:,-1] - np.mean(space[:,-1]))**2.)
        return mse    
    
    def node(self, X):
        '''split with the optimal gini, entropy, or mse index'''
        classes = collections.Counter(X[:,-1])
        classes = list(classes.keys())
        tmp_value = 999     # This is a random choice, may need more careful for regession case.
        tmp_IG = -999
        tmp_gini = 999
        # randomly select features, add randomness to system
        column_id = sorted(np.random.choice(X.shape[1], self.mf, replace=True))
        #column_id = sorted(np.random.choice(X.shape[1], X.shape[1], replace=False)) # Kill feature randomness
        
        for column in range(X[:,column_id].shape[1]-1):
            for row in X:
                left = np.empty((0, X.shape[1]))
                right = np.empty((0, X.shape[1]))
                for row2 in X:
                    if row2[column] < row[column]:
                        left = np.append(left, row2.reshape(-1, np.size(X, 1)), axis=0)
                    else:
                        right = np.append(right, row2.reshape(-1, np.size(X, 1)), axis=0)
                if (self.criterion == 'gini'):
                    gini = self.evaluate_gini_index(left, right, classes)
                    if gini < tmp_gini:
                            node_c, node_value, tmp_gini = column, row[column], gini
                            left_branch, right_branch = left, right
                elif (self.criterion == 'entropy'):
                    IG = self.evaluate_information_gain(left, right, classes)
                    if IG > tmp_IG:
                            node_c, node_value, tmp_IG = column, row[column], IG
                            left_branch, right_branch = left, right
                elif (self.criterion == 'mse'):
                    mse = self.evaluate_mse_index(left, right)
                    if mse < tmp_value:
                            node_c, node_value, tmp_value = column, row[column], mse
                            left_branch, right_branch = left, right
                else:
                    print('I\'m trying to add more criterion in it! ')
        return {'feature_id': node_c, 'node_value': node_value, \
            'Left_branch': left_branch, 'Right_branch': right_branch}

    def tree_grows(self, X, max_depth, min_samples, depth):
        '''Recursively growing binary tree'''
        left, right = X['Left_branch'], X['Right_branch']
        for key in ['Left_branch', 'Right_branch']:
            try:
                del X[key]
            except KeyError:
                pass
        #if np.size(left, 0)==0 or np.size(right, 0)==0:
        if left.shape[0]==0 or right.shape[0]==0: # No need to split if encounter empty branch
            X['Left_branch'] = self.leaf_node(np.vstack((left,right)))
            X['Right_branch'] = self.leaf_node(np.vstack((left,right)))
            return
        if depth >= max_depth:   # tree depth should be smaller then max_depth
            X['Left_branch'] = self.leaf_node(left)
            X['Right_branch'] = self.leaf_node(right)
            return
        dict_tmp={'Left_branch':left, 'Right_branch':right}
        for i in ['Left_branch', 'Right_branch']:   # Left/Right branches grow
            #if np.size(dict_tmp[i], 0) > min_samples: 
            if dict_tmp[i].shape[0] > min_samples: 
                X[i] = self.node(dict_tmp[i])
                self.tree_grows(X[i], self.md, self.ms, depth+1)
            else:
                X[i] = self.leaf_node(dict_tmp[i])
                
    def leaf_node(self, X):  
        '''node is viewed as leaf, the most voted label is the leaf node label'''
        if (self.criterion == 'gini' or 'entropy'):
            statis = collections.Counter(X[:,-1])
            max_votes=max(statis.values())
            lst=[i for i in statis.keys() if statis[i]==max_votes] 
            return sorted(lst)[0]    
        elif (self.criterion == 'mse'): 
            return np.mean(X[:,-1])  
            
    
    def fit(self, X):
        '''Used to obtain root node and build decision tree'''
        #start1 = time.time()
        Node = self.node(X) # generate node
        #end1 = time.time()
        #print(f"Runtime of the node is {end1 - start1}")
        self.tree_grows(Node, self.md, self.ms, self.depth_init)
        return Node
    
    def predict_sample(self, X, Y):
        '''Used to predict each sample data Y based on tree X'''
        if Y[int(X['feature_id'])] < X['node_value']:
            if isinstance(X['Left_branch'], dict):
                return self.predict_sample(X['Left_branch'], Y)
            else:
                y_pred = X['Left_branch']
        else:
            if isinstance(X['Right_branch'], dict):
                return self.predict_sample(X['Right_branch'], Y)
            else:
                y_pred = X['Right_branch']
        return y_pred     
    
    def make_prediction(self, X, Y):
        '''make prediction for given test data Y
           based on tree X'''
        Y_pred = np.empty((0,1))
        for i in range(Y.shape[0]):
            tmp = self.predict_sample(X, Y[i])
            tmp1 = np.array(tmp)
            tmp1 = tmp1.reshape(-1,1)
            Y_pred = np.append(Y_pred, tmp1, axis=0)
        Y_pred = np.squeeze(Y_pred)
        return Y_pred
    
    def export_tree(self, X, depth=0):
        if isinstance(X, dict):
            print(('%s%sfeature_%d <= %f')% (depth*'| ', '|--', X['feature_id'], X['node_value']))
            self.export_tree(X['Left_branch'], depth+1)
            print(('%s%sfeature_%d > %f')% (depth*'| ', '|--', X['feature_id'], X['node_value']))
            self.export_tree(X['Right_branch'], depth+1)
        else:
            if self.criterion == 'gini' or 'entropy':
                print(('%s%sclass: %f')%(depth*'| ', '|--', X))
            elif self.criterion == 'mse':
                print(('%s%savg value: %f')%(depth*'| ', '|--', X))
            
    def bootstrapping(self, X):
        '''Leave-one-out cross validation to perform out-of-bag(oob) estimation,
           the oob prob. is (1-1/m)^m, if m->inf, prob. -> e^-1 ~ 0.36787.
           The bootstrapping is the process of sampling m points with replacement,
           this add randomness to each decision tree.'''
        b_id = sorted(np.random.choice(X.shape[0], X.shape[0], replace=True))
        X_b = X[b_id, :]
        oob_id = [i for i in range(X.shape[0]) if i not in b_id]
        X_oob = X[oob_id, :]
        return X_b, X_oob
            
    def oob_accuracy(self, X, Y):
        '''evaluate accuracy for Y
           based on tree X'''
        y_pred_oob = np.zeros(Y.shape[0])
        y_pred_oob = self.make_prediction(X, Y)
        y_label_oob = Y[:, -1].astype(int)
        return metric_accuracy(y_label_oob, y_pred_oob.astype(int))

        
    def forest_fit(self, X):
        random_forest = []
        for i in range(self.ne):
            X_b, X_oob = self.bootstrapping(X)
            single_tree = self.fit(X_b)
            print('OOB accuracy for single tree_', i+1, ': ', self.oob_accuracy(single_tree, X_oob), '%')
            random_forest.append(single_tree)
        return random_forest
  
    def make_prediction_rf(self, X, Y):
        '''make prediction for given test data Y
           based on random forest(rf) X'''
        Y_pred = np.empty((0,1))
        for i in range(Y.shape[0]):
            tmp_l = []
            tmp_l = [self.predict_sample(tree, Y[i]) for tree in X]
            tmp = max(tmp_l, key = tmp_l.count)
            tmp1 = np.array(tmp)
            tmp1 = tmp1.reshape(-1,1)
            Y_pred = np.append(Y_pred, tmp1, axis=0)
        Y_pred = np.squeeze(Y_pred)
        return Y_pred
    
def main():

    truncate_train=200
    truncate_test=20
    X, Y = readin_csv_data('data_banknote_authentication.csv')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=.9, shuffle=True)
    X_trainset = np.hstack((X_train[:truncate_train, :], Y_train.reshape(-1,1)[:truncate_train, :]))
    X_testset = np.hstack((X_test[:truncate_test, :], Y_test.reshape(-1,1)[:truncate_test, :]))
    model =  RandomForestClassification(n_estimators=20, max_features=4, max_depth = 3, min_samples = 10, criterion='gini')   
    tree = model.forest_fit(X_trainset)
    #print(tree)
    #model.export_tree(tree)
    y_pred = np.zeros(X_testset.shape[0])
    y_pred = model.make_prediction_rf(tree, X_testset)
    y_label = X_testset[:, -1].astype(int)
    print('Random Forest Accuracy: ', metric_accuracy(y_label, y_pred.astype(int)), ' %') 
    #g = Digraph('G', filename='test.gv')
    #visualize_tree_root_node(g, tree)
    #visualize_tree(g, tree)
    #print(g)

if __name__ == '__main__':  
    main()
