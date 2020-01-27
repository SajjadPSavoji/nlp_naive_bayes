import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm , tqdm_notebook , trange
import operator
import copy
from sklearn.model_selection import train_test_split


def take_data(file_name , test_sz = 0.2):
    '''
        file_name should refer to a csv file with 2 coulumns text and label
        returns f_train , f_test , l_train , l_test
    '''
    data = pd.read_csv(file_name)
    labels = data['label']
    features = data['text']
    return train_test_split(features , labels , test_size = test_sz , random_state = 1234  , shuffle = True)

def take_valid(file_name):
    '''
        reads valid data as pd.df 
        return features , labels
    '''
    data = pd.read_csv(file_name)
    return data

def recall(l_est , l_test , name = 'hafez'):
    # correct detected hafez
    num = np.sum(np.logical_and((l_est == name).to_numpy() , (l_test == name).to_numpy()))
    # all hafezez
    den = np.sum((l_test == name).to_numpy())
    return num/den

def precision(l_est , l_test , name = 'hafez'):
    # correct detected hafez
    num = np.sum(np.logical_and((l_est == name).to_numpy() , (l_test == name).to_numpy()))
    # detected hafezez
    den = np.sum((l_est == name).to_numpy())
    return num/den

def accuracy(l_est , l_test):
    # correct detected
    num = sum(l_est == l_test)
    # all
    den = len(l_test)
    return num/den



class NLP_classifier():
    def __init__(self , num_classes = 2):
        self.num_class = num_classes
        self.trained = False
        self.NULL = '<NULL>'
    
    def fit(self , data , labels):
        self.init_prrior(labels)
        self.init_max_word(data)
        self.feature_conditioning(data)
        self.dists = {}
        for clss in self.prriors.keys():
            self.dists[clss] = [Distribution() for i in range(self.max_word)]
        for clss in self.prriors.keys():
            for i in range(self.max_word):
                self.dists[clss][i].fit(data[labels == clss].apply(lambda x : x[i]))
        self.trained = True
        
    def transform(self , data):
        self.feature_conditioning(data)
        return data.apply(lambda x:self.single_row_transform(x))

    def fit_transform(self , train_data  , train_labels , test_data):
        self.fit(train_data , train_labels)
        return self.transform(test_data)

    def single_row_transform(self , row):
        disc_func = {}
        for key in self.prriors.keys():
            disc_func[key] = self.disc_function(key , row)
        
        temp = max(disc_func.items(), key=operator.itemgetter(1))[0]
        return temp

    def disc_function(self , key , data):
        p = 1
        for i in range(self.max_word):
            p *= self.dists[key][i].transform(data[i])
        p *= self.prriors[key]
        return p  

    def init_prrior(self , labels):
        self.prriors = {}
        # init prriors
        LEN_LABELS = len(labels)
        for label in labels : 
            if label in self.prriors : 
                self.prriors[label] += 1/LEN_LABELS
            else :
                self.prriors[label] = 1/LEN_LABELS 
    
    def init_max_word(self , data):
        lens = [len(i.split()) for i in data]
        # self.max_word = np.max(lens)
        # self.max_word = int(np.mean(lens) + 3*np.std(lens))
        self.max_word = np.min(lens)
    
    def feature_conditioning(self , data):
        for i in tqdm_notebook(range(len(data)) , desc= " feature conditioning: "):
            data.iloc[i] = self.clip_fill(data.iloc[i])
    
    def clip_fill(self , txt):
        txt = (str(txt)).split()
        if (len(txt) > self.max_word):  #clip
            txt = txt[0:self.max_word]
        elif (len(txt) < self.max_word): #fill
            for i in range(self.max_word - len(txt)):
                txt.append(self.NULL)
        return txt

class Simple_NLP(NLP_classifier):
    def __init__(self , num_classes = 2 , n = 1):
        self.num_class = num_classes
        self.trained = False
        self.NULL = '<NULL>'
        self.n = n

    def fit(self , data , labels):
        self.init_prrior(labels)
        self.feature_conditioning(data)

        self.dists = {}
        self.count = {}
        for key in self.prriors.keys():
            self.dists[key] = {}
            self.count[key] = 0
            self.train(data[labels == key] , key)

        self.trained = True
    
    def train(self , data_key , key):

        for i in range(len(data_key)):
            for j in range(len(data_key.iloc[i])):
                self.count[key] += 1
                if data_key.iloc[i][j] in self.dists[key] : 
                    self.dists[key][data_key.iloc[i][j]] += 1
                else :
                    self.dists[key][data_key.iloc[i][j]] = 1

        for word in self.dists[key] : 
            self.dists[key][word] /= self.count[key] 
        self.soften(key , self.n)

    def soften(self , key , n):
        # make an unkown word for each distribution
        p_min = 0
        for i in range(n):
            min_key = min(self.dists[key] , key=self.dists[key].get)
            p_min += self.dists[key][min_key]
            self.dists[key].pop(min_key)
        self.UNK = '<UNK>'
        self.dists[key][self.UNK] = p_min
    
    def clip_fill(self , txt):
        txt = (str(txt)).split()
        return txt

    def disc_function(self , key , row):
        p = 1
        for i in range(len(row)):
            if row[i] in self.dists[key]:
                p *= self.dists[key][row[i]]
            else : 
                p *= self.dists[key][self.UNK]
        p *= self.prriors[key]
        return p 

class Laplace_NLP(Simple_NLP): # n should be in [0 , 1]
    def soften(self , key , n):
       # make an unkown word for each distribution
        self.UNK = '<UNK>'
        p_min = 0
        self.dists[key][self.UNK] = p_min
        for word in self.dists[key]:
            self.dists[key][word] = (self.dists[key][word]*self.count[key] + n )/ (self.count[key] + n*len(self.dists[key]))

class Distribution():
    def __init__(self ):
        self.UNK = '<UNK>'
        self.PERC = 0

    def fit(self , data):
        self.p = {}
        LEN = len(data)
        for word in data:
            if word in self.p : 
                self.p[word] += 1/LEN
            else :
                self.p[word] = 1/LEN

        min_key = min(self.p , key=self.p.get)
        p_unk = self.p[min_key]
        self.p.pop(min_key)
        self.p[self.UNK] = p_unk


    def transform(self , word):
        if word in self.p : 
            return self.p[word]
        return self.p[self.UNK]
        

if __name__ == "__main__":
    pass
    # file_name = 'data.csv'
    # f_train , f_test , l_train , l_test = take_data(file_name)
    # f_temp = copy.deepcopy(f_train)
    # nlp_clf = NLP_classifier()
    # nlp_clf.fit(f_temp , l_train)

    # l_estimate = nlp_clf.transform(f_test)
    # print(sum(l_estimate == l_test)/len(l_test))
    # print(len(l_estimate))
    # print(sum(l_test == 'hafez'))
    # print(sum(l_estimate == 'hafez'))


    # file_name = 'data.csv'
    # f_train , f_test , l_train , l_test = take_data(file_name)
    # f_temp = copy.deepcopy(f_train)
    # nlp_clf = Simple_NLP()
    # nlp_clf.fit(f_temp , l_train)

    # l_estimate = nlp_clf.transform(f_test)
    # print(sum(l_estimate == l_test)/len(l_test))
    # print(len(l_estimate))
    # print(sum(l_test == 'hafez'))
    # print(sum(l_estimate == 'hafez'))