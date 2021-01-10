import numpy as np
import collections
import pandas as pd
from graphviz import Digraph

def readin_csv_data(path):
    df = pd.read_csv(path) 
    X = df.iloc[:,:-1].values 
    Y = df.iloc[:,-1].values 
    return X, Y


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
    for i in range(np.size(Y_pred)) : #np.size: Number of elements in the array
        if Y_label[i] == Y_pred[i] :             
            correct_amount = correct_amount + 1
    return correct_amount / np.size(Y_pred) * 100



class ClassificationDecisionTree():
    '''Decision Tree for Classification'''
    
    def __init__(self, max_depth, min_samples):
        self.md = max_depth
        self.ms = min_samples
        self.depth_init = 1
       
    def evaluate_gini_index(self, left, right, classes):
        gini = 0
        size_l = np.size(left, 0)
        size_r = np.size(right, 0)
        size_t = size_l +  size_r # Total sample No.
        spaces = [left, right]
        for space in spaces:
            size = np.size(space, 0) # Sample No. in different group
            if size == 0:   # Avoid 0 in denominator
                continue     
            tmp = 0.
            statis = collections.Counter(space[:,-1])
            for class_i in classes:
                p_k = statis[class_i] / size # +/- instances prob. in different group
                tmp += p_k*p_k
            gini += (1.0 - tmp) * (size / size_t)
        return gini    
    
    def node(self, X):
        '''split with the optimal gini index'''
        classes = collections.Counter(X[:,-1])
        classes = list(classes.keys())
        tmp_gini = 1
        for column in range(np.size(X, 1)-1):
            for row in X:
                left = np.empty((0, np.size(X, 1)))
                right = np.empty((0, np.size(X, 1)))
                for row2 in X:
                    if row2[column] < row[column]:
                        left = np.append(left, row2.reshape(-1,np.size(X, 1)), axis=0)
                    else:
                        right = np.append(right, row2.reshape(-1,np.size(X, 1)), axis=0)
                gini = self.evaluate_gini_index(left, right, classes)
                if gini < tmp_gini:
                        node_c, node_value, tmp_gini = column, row[column], gini
                        left_branch, right_branch = left, right
        #print('Node column: %f' % node_c ) 
        #print('Node value: %f' % node_value )
        #print('Node gini index: %f' % tmp_gini )
        #print('Left branch: ', left_branch )
        #print('Right branch: ', right_branch )
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
        if np.size(left, 0)==0 or np.size(right, 0)==0: # No need to split if encounter empty branch
            X['Left_branch'] = self.leaf_node(np.vstack((left,right)))
            X['Right_branch'] = self.leaf_node(np.vstack((left,right)))
            return
        if depth >= max_depth:   # tree depth should be smaller then max_depth
            X['Left_branch'] = self.leaf_node(left)
            X['Right_branch'] = self.leaf_node(right)
            return
        dict_tmp={'Left_branch':left, 'Right_branch':right}
        for i in ['Left_branch', 'Right_branch']:   # Left/Right branches grow
            if np.size(dict_tmp[i], 0) > min_samples: 
                X[i] = self.node(dict_tmp[i])
                self.tree_grows(X[i], self.md, self.ms, depth+1)
            else:
                X[i] = self.leaf_node(dict_tmp[i])
                
    def leaf_node(self, X):  
        '''node is viewed as leaf, the most voted label is the leaf node label'''
        statis = collections.Counter(X[:,-1])
        #print('The leaf node labels and counts: ', statis)
        max_votes=max(statis.values())
        lst=[i for i in statis.keys() if statis[i]==max_votes] 
        #print('The most voted leaf node label is: ', sorted(lst)[0]) 
        return sorted(lst)[0]        
    
    def fit(self, X):
        '''Used to obtain root node and build decision tree'''
        Node = self.node(X) # generate node
        self.tree_grows(Node, self.md, self.ms, self.depth_init)
        return Node
       
    def predict(self, X, Y):
        '''Used to predict test data Y based on tree X'''
        y_pred = np.zeros(np.size(Y, 0))
        for i in range(np.size(Y, 0)):
            if Y[i, int(X['feature_id'])] < X['node_value']:
                if isinstance(X['Left_branch'], dict):
                    return self.predict(X['Left_branch'], Y)
                else:
                    y_pred[i] = X['Left_branch']
            else:
                if isinstance(X['Right_branch'], dict):
                    return self.predict(X['Right_branch'], Y)
                else:
                    y_pred[i] = X['Right_branch']
        return y_pred.astype(int)    
    
    def export_tree(self, X, depth=0):
        if isinstance(X, dict):
            print(('%s%sfeature_%d <= %f')% (depth*'| ', '|--', X['feature_id'], X['node_value']))
            self.export_tree(X['Left_branch'], depth+1)
            #print(('%s%sfeature_%d > %f')% (depth*'| ', '|--', X['feature_id'], X['node_value']))
            self.export_tree(X['Right_branch'], depth+1)
        else:
            print(('%s%sclass: %f')%(depth*'| ', '|--', X))
            
            
    def print_tree(self, node, depth=0):
        if isinstance(node, dict):
            print('%s[X%d < %.3f]' % ((depth*' ', (node['feature_id']+1), node['node_value'])))
            self.print_tree(node['Left_branch'], depth+1)
            self.print_tree(node['Right_branch'], depth+1)
        else:
            print('%s[%s]' % ((depth*' ', node)))
            
            
def visualize_tree_root_node(g, X):
    g.node(name = 'F_%d <= %.2f'%(X['feature_id'],X['node_value']), \
           shape='box', color='black',style='filled',fillcolor='lemonchiffon')

def visualize_tree(g, X):
       
    if not isinstance(X['Left_branch'], dict) and not isinstance(X['Right_branch'], dict):  
        #if X['Left_branch']!=X['Right_branch']:
            #g.node('%.2f'%(X['node_value']))
        g.node(name='%.2f left child'%(X['node_value']), label='%.2f'%(X['Left_branch']),\
               shape='diamond', color='black',style='filled',fillcolor='darkorange1')
        g.node(name='%.2f right child'%(X['node_value']), label='%.2f'%(X['Right_branch']),\
               shape='diamond', color='black',style='filled',fillcolor='darkorange1')
        g.edge('F_%d <= %.2f'%(X['feature_id'],X['node_value']), '%.2f left child'%(X['node_value']), label='True')
        g.edge('F_%d <= %.2f'%(X['feature_id'],X['node_value']), '%.2f right child'%(X['node_value']), label='False')
        #else:
            #g.node('%.2f'%(X['node_value']))
            #g.node(name='%.2f child'%(X['node_value']), label='%.2f'%(X['Left_branch']))
            #g.edge('F_%d <= %.2f'%(X['feature_id'],X['node_value']), '%.2f'%(X['Left_branch']))

    elif isinstance(X['Left_branch'], dict) and not isinstance(X['Right_branch'], dict): 
        #g.node('%.2f'%(X['node_value']))
        #g.node(name='%.2f left child'%(X['node_value']), label='%.2f'%(X['Left_branch']['node_value']))
        #g.edge('%.2f'%(X['node_value']), '%.2f left child'%(X['node_value']), label='true')
        g.edge('F_%d <= %.2f'%(X['feature_id'],X['node_value']), 'F_%d <= %.2f'%(X['feature_id'],X['Left_branch']['node_value']),\
               label='True')
        g.node(name='%.2f right child'%(X['node_value']), label='%.2f'%(X['Right_branch']), \
               shape='diamond', color='black',style='filled',fillcolor='darkorange1')
        g.edge('F_%d <= %.2f'%(X['feature_id'],X['node_value']), '%.2f right child'%(X['node_value']), label='False')
        visualize_tree(g, X['Left_branch'])
    elif not isinstance(X['Left_branch'], dict) and isinstance(X['Right_branch'], dict):
        #g.node('%.2f'%(X['node_value']))
        g.node(name='%.2f left child'%(X['node_value']), label='%.2f'%(X['Left_branch']),\
               shape='diamond', color='black',style='filled',fillcolor='darkorange1')
        #g.node(name='%.2f right child'%(X['node_value']), label='%.2f'%(X['Right_branch']['node_value']))
        g.edge('F_%d <= %.2f'%(X['feature_id'],X['node_value']), '%.2f left child'%(X['node_value']), label='True')
        #g.edge('%.2f'%(X['node_value']), '%.2f right child'%(X['node_value']), label='false')
        g.edge('F_%d <= %.2f'%(X['feature_id'],X['node_value']), 'F_%d <= %.2f'%(X['feature_id'],X['Right_branch']['node_value']), \
               label='False')
        visualize_tree(g, X['Right_branch'])
    else:
        #g.node('%.2f'%(X['node_value']))
        #g.node(name='%.2f left child'%(X['node_value']), label='%.2f'%(X['Left_branch']['node_value']))
        #g.node(name='%.2f right child'%(X['node_value']), label='%.2f'%(X['Right_branch']['node_value']))
        g.edge('F_%d <= %.2f'%(X['feature_id'],X['node_value']),\
               'F_%d <= %.2f'%(X['Left_branch']['feature_id'],X['Left_branch']['node_value']), label='True')
        g.edge('F_%d <= %.2f'%(X['feature_id'],X['node_value']), \
               'F_%d <= %.2f'%(X['Right_branch']['feature_id'],X['Right_branch']['node_value']), label='False')
        visualize_tree(g, X['Left_branch'])
        visualize_tree(g, X['Right_branch'])
        
        
def main():
    truncate_train=300
    truncate_test=50
    X, Y = readin_csv_data('data_banknote_authentication.csv')
    X_train, X_test, Y_train, Y_test = train_test_split_po(X, Y, train_size=.8, shuffle=True)
    X_trainset = np.hstack((X_train[:truncate_train, :], Y_train.reshape(-1,1)[:truncate_train, :]))
    X_testset = np.hstack((X_test[:truncate_test, :], Y_test.reshape(-1,1)[:truncate_test, :]))
    #X_trainset = np.hstack((X_train[:, :], Y_train.reshape(-1,1)[:, :]))
    #X_testset = np.hstack((X_test[:, :], Y_test.reshape(-1,1)[:, :]))
    model =  ClassificationDecisionTree(max_depth = 3, min_samples = 10)   
    tree = model.fit(X_trainset)
    print(tree)
    model.export_tree(tree)
    y_pred = model.predict(tree, X_testset)
    y_label = X_testset[:, -1].astype(int)
    print('Accuracy: ', metric_accuracy(y_label, y_pred), ' %') 
    g = Digraph('G', filename='DT.gv', format='png')
    #g = Digraph()
    visualize_tree_root_node(g, tree)
    visualize_tree(g, tree)
    g.view()

if __name__ == '__main__':  
    main()