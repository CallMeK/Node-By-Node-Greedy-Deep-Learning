

#main part

import numpy as np
from sklearn.linear_model import LogisticRegression
import standard_tool as stool
import sys

# filename = 'N0.csv'
# Geometry = np.array([40]) #There could be a lot of layers

def groupby(classes):
    result = {}
    for i, x in enumerate(classes):
        if (x not in result):  # as key
            result[x] = []
        result[x].append(i)
    return result

class BaseMethod(object):
    def __init__(self,Geometry,opt_method='SGD',IO_agent=None,alpha=0.1,epoch_limit=300,alpha_ft=0.1,epoch_limit_ft=100,FT=False,AF=None,batch_size=20):
        self.Geometry = Geometry
        self.opt_method = opt_method
        self.FT = FT
        self.alpha = alpha
        self.epoch_limit = epoch_limit
        self.AF = AF
        self.alpha_ft = alpha_ft
        self.epoch_limit_ft = epoch_limit_ft
        self.batch_size = batch_size

        if(self.opt_method == 'unsup'):
            self.pg = __import__('propagation')
        elif(self.opt_method == 'super'):
            self.pg = __import__('supervise_propogation')
        elif(self.opt_method == 'greedy'):
            self.pg = __import__('propagation_greedy')
        elif(self.opt_method == 'greedy_class'):
            self.pg = __import__('propagation_greedy_class')
        else:
            #Must be NN
            self.FT = True

        if(IO_agent is None):
            from IO_Wrapper import IO_Wrapper_stdout
            self.IO_agent = IO_Wrapper_stdout
        else:
            self.IO_agent = IO_agent

    def Draw_Features(self):
        from matplotlib import pylab as plt
        #The first level features
        num_f = self.Geometry[0]
        for i in xrange(num_f):
            if (i % 4 == 0):
                filename = str(i) + '-' + str(i + 3) + '_features.png'
                fig, axes = plt.subplots(2, 2)
                ax = axes.flat
            im = ax[i % 4].imshow(self.weights[0].T[i, :-1].reshape(16, 16), vmin=-0.3, vmax=0.3, interpolation='none')
            if (i + 1 == num_f or i % 4 == 3):
                fig.subplots_adjust(right=0.8)
                cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                fig.colorbar(im, cax=cbar_ax)
                self.IO_agent.savefig(fig,filename )

class BasicNN(BaseMethod):
    def train(self, data, response):
        N = data.shape[0]
        dim_feature = data.shape[1]  #should be 256
        lower_layer = dim_feature - 1  #to be consistent with the geometry

        #construct the weights
        self.weights = {}
        counter = 0
        for n in self.Geometry:
            self.weights[counter] = (np.random.rand(lower_layer + 1, n) - 0.5) / lower_layer
            #define as inputs*weights = outputs
            counter = counter + 1
            lower_layer = n

        self.inputs = {}
        self.inputs[0] = data
        self.Training_step = []

        #Now to the final layer
        # self.LR = LogisticRegression(C=1.0, tol=0.0001)
        # self.LR.fit(self.inputs[len(self.Geometry)][:,:-1], response)
        self.trans_table = list(np.unique(response))
        node_response = np.array([self.trans_table.index(x) for x in response])
        self.weights[counter] = (np.random.rand(lower_layer + 1, len(self.trans_table)) - 0.5)/lower_layer

        #Now Fine Tuning...
        if(self.FT):
            stool.AllNet_Train_SGD(data,node_response,self.weights,
                                   IO_agent = self.IO_agent,alpha = self.alpha_ft,epoch_limit=self.epoch_limit_ft,batch_size=self.batch_size)


    def score(self,output,response):
        node_response = np.array([self.trans_table.index(x) for x in response])
        node_predict = np.argmax(output,axis=1)
        N = output.shape[0]
        return sum(node_response == node_predict)/float(N)

    def predict(self, test_data):
        output = stool.Forward_all(test_data, self.weights)
        return output

    def get_score(self, data, response):
        output = self.predict(data)
        return self.score(output, response)



class NormalMethod(BaseMethod):
    def train(self, data, response):
        N = data.shape[0]
        dim_feature = data.shape[1]  #should be 256
        lower_layer = dim_feature - 1  #to be consistent with the geometry

        #construct the weights
        self.weights = {}
        counter = 0
        for n in self.Geometry:
            self.weights[counter] = (np.random.rand(lower_layer + 1, n) - 0.5) / (lower_layer+1+n)
            #define as inputs*weights = outputs
            counter = counter + 1
            lower_layer = n

        self.inputs = {}
        self.inputs[0] = data
        self.Training_step = []

        for l in np.arange(len(self.Geometry)):
            #also can set tol and C_reg
            self.pg.Train_SGD(self.inputs[l], response, self.weights[l],self.IO_agent,self.alpha,self.epoch_limit,self.AF)
            #Now the weights[l] should already changed, due to directly passing by reference
            self.inputs[l + 1] = np.hstack([self.pg.Forward_1(self.inputs[l], self.weights[l]), np.ones([N, 1])])

        self.weight0prev=self.weights[0] #save the weights before tuning
        #Now to the final layer
        # self.LR = LogisticRegression(C=1.0, tol=0.0001)
        # self.LR.fit(self.inputs[len(self.Geometry)][:,:-1], response)
        self.trans_table = list(np.unique(response)) #no guarentee on order...
        node_response = np.array([self.trans_table.index(x) for x in response])
        self.weights[counter] = (np.random.rand(lower_layer + 1, len(self.trans_table)) - 0.5)/\
                                (lower_layer+len(self.trans_table)+1)


        stool.Logistic_Train_SGD(self.inputs[l+1],node_response,self.weights[counter],
                               IO_agent=self.IO_agent,alpha = self.alpha_ft,epoch_limit=self.epoch_limit,batch_size=self.batch_size)

        #Now Fine Tuning...
        if(self.FT):
            stool.AllNet_Train_SGD(data,node_response,self.weights,
                                   IO_agent = self.IO_agent,alpha = self.alpha_ft,epoch_limit=self.epoch_limit_ft,batch_size=self.batch_size)



    def score(self,output,response):
        node_response = np.array([self.trans_table.index(x) for x in response])
        node_predict = np.argmax(output,axis=1)
        N = output.shape[0]
        return sum(node_response == node_predict)/float(N)

    def predict(self, test_data):
        output = stool.Forward_all(test_data, self.weights)
        return output

    def get_score(self, data, response):
        output = self.predict(data)
        return self.score(output, response)


class SuperMethod(BaseMethod):
    #The supervised version of pretraining
    def train(self, data, response):
        N = data.shape[0]
        dim_feature = data.shape[1]  #should be 256
        lower_layer = dim_feature - 1  #to be consistent with the geometry

        #construct the weights
        self.weights = {}
        counter = 0
        for n in self.Geometry:
            self.weights[counter] = (np.random.rand(lower_layer + 1, n) - 0.5) / (lower_layer - 1)
            #define as inputs*weights = outputs
            counter = counter + 1
            lower_layer = n

        self.inputs = {}
        self.inputs[0] = data
        self.trans_table = list(np.unique(response)) #important, save all possible output

        node_response = np.array([self.trans_table.index(x) for x in response])

        for l in np.arange(len(self.Geometry)):
            info, w2 = self.pg.Train_SGD(self.inputs[l],node_response, self.weights[l],self.IO_agent,self.alpha,self.epoch_limit)
            #Now the weights[l] should already changed, due to directly passing by reference
            self.inputs[l + 1] = np.hstack([self.pg.Forward_1(self.inputs[l], self.weights[l]), np.ones([N, 1])])


        self.weight0prev=self.weights[0] #save the weights before tuning
        #Now to the final layer

        self.weights[counter] = (np.random.rand(lower_layer + 1, len(self.trans_table)) - 0.5)/\
                                (lower_layer+len(self.trans_table)+1)

        stool.AllNet_Train_SGD(data,node_response,self.weights,
                                   IO_agent = self.IO_agent,alpha = self.alpha_ft,epoch_limit=self.epoch_limit_ft,batch_size=self.batch_size)

    def score(self,output,response):
        node_response = np.array([self.trans_table.index(x) for x in response])
        node_predict = np.argmax(output,axis=1)
        N = output.shape[0]
        return sum(node_response == node_predict)/float(N)

    def predict(self, test_data):
        output = stool.Forward_all(test_data, self.weights)
        return output

    def get_score(self, data, response):
        output = self.predict(data)
        return self.score(output, response)

# A different network structure based on classes, seperating the training for the whole network
# do not confuse it with the by-class-greedy
# Not used in the paper

# class ByClassMethod(BaseMethod):
#     def train(self,data,response):
#         #in byclass method, each class will have the same amount of geometry
#         response_list = np.unique(response)
#         n_class = len(response_list)
#         for geom in self.Geometry:
#             assert geom%n_class == 0
#         N = data.shape[0]
#         dim_feature = data.shape[1]  #should be 256
#         lower_layer = dim_feature - 1  #to be consistent with the geometry

#         #construct the weights, differently
#         self.weights = {}
#         counter = 0
#         for n in self.Geometry:
#             self.weights[counter] = {}
#             for i in np.arange(n_class):
#                 self.weights[counter][i] = (np.random.rand(lower_layer + 1, n/n_class) - 0.5) / (lower_layer - 1)
#             #define as inputs*weights = outputs
#             counter = counter + 1
#             lower_layer = n/n_class

#         self.inputs = {}
#         index_list = groupby(response)
#         self.inputs[0] = [ data[index_list[x]] for x in response_list] #It will have the same order as it is
#         self.Training_step = []


#         for l in xrange(len(self.Geometry)):
#             self.inputs[l+1] = []
#             for i in xrange(n_class):
#                 inputs = self.inputs[l][i]

#                 if (self.opt_method == 'SGD'):
#                     info, w2 = self.pg.Train_SGD(inputs, self.weights[l][i],self.IO_agent,self.alpha,self.epoch_limit)
#                     #also can set tol and C_reg
#                 else:
#                     raise Exception('Undefined Method')
#                 #Now the weights[l] should already changed, due to directly passing by reference
#                 self.inputs[l + 1].append(self.pg.AddIntercept(self.pg.Forward_1(self.inputs[l][i], self.weights[l][i])))
#                 self.Training_step.append(info)


#         #Now to the final layer
#         output = self.predict(data)
#         self.LR = LogisticRegression(C=1.0, tol=0.0001)
#         self.LR.fit(output, response)

#     def predict(self, test_data):
#         output = [self.pg.AddIntercept(self.pg.Forward_1(test_data,x)) for x in self.weights[0]]

#         for l in xrange(1,len(self.weights)):
#             output = map(self.pg.Forward_1,output, self.weights[l])
#             output = [self.pg.AddIntercept(x) for x in output]

#         output = np.concatenate([x[:,:-1] for x in output],axis=1)
#         return output

#     def get_score(self, data, response):
#         output = self.predict(data)
#         return self.LR.score(output, response)



# #With K-mean clustering, not used for the paper, not sure if it is working.
# class ByClusterMethod(BaseMethod):

#     def train(self,data,response):
#         #in byclass method, each class will have the same amount of geometry
#         response_list = np.unique(response)
#         n_class = len(response_list)
#         n_expand = 1
#         n_cluster = n_expand*n_class
#         for geom in self.Geometry:
#             assert geom%(n_cluster) == 0
#         N = data.shape[0]
#         dim_feature = data.shape[1]  #should be 256
#         lower_layer = dim_feature - 1  #to be consistent with the geometry

#         #construct the weights, differently
#         self.weights = {}
#         counter = 0
#         for n in self.Geometry:
#             self.weights[counter] = []
#             rand_weights = (np.random.rand(lower_layer + 1, n/n_cluster) - 0.5) / (lower_layer - 1)
#             for i in np.arange(n_cluster):
#                 #self.weights[counter][i] = (np.random.rand(lower_layer + 1, n/n_cluster) - 0.5) / (lower_layer - 1)
#                 self.weights[counter].append(rand_weights)
#             #define as inputs*weights = outputs
#             counter = counter + 1
#             lower_layer = n/n_cluster

#         self.inputs = {}

#         from sklearn.cluster import KMeans
#         self.KMean = KMeans(n_clusters=n_cluster)

#         cluster_index = self.KMean.fit_predict(data[:,:-1])
#         index_list = groupby(cluster_index)
#         self.inputs[0] = [ data[index_list[x]] for x in xrange(n_cluster)] #It will have the same order as it is
#         self.Training_step = []


#         for l in xrange(len(self.Geometry)):
#             self.inputs[l+1] = []
#             for i in xrange(n_cluster):
#                 inputs = self.inputs[l][i]
#                 print i,inputs.shape,self.weights[l][i].shape

#                 if (self.opt_method == 'SGD'):
#                     info, w2 = self.pg.Train_SGD(inputs, self.weights[l][i],self.IO_agent,self.alpha,self.epoch_limit)
#                     #also can set tol and C_reg
#                 else:
#                     raise Exception('Undefined Method')
#                 #Now the weights[l] should already changed, due to directly passing by reference
#                 self.inputs[l + 1].append(self.pg.AddIntercept(self.pg.Forward_1(self.inputs[l][i], self.weights[l][i])))
#                 self.Training_step.append(info)


#         #Now to the final layer
#         output = self.predict(data)
#         self.LR = LogisticRegression(C=1.0, tol=0.0001)
#         self.LR.fit(output, response)

#     def predict(self, test_data):
#         output = [self.pg.AddIntercept(self.pg.Forward_1(test_data,x)) for x in self.weights[0]]

#         for l in xrange(1,len(self.weights)):
#             output = map(self.pg.Forward_1,output, self.weights[l])
#             output = [self.pg.AddIntercept(x) for x in output]

#         output = np.concatenate([x[:,:-1] for x in output],axis=1)
#         return output

#     def get_score(self, data, response):
#         output = self.predict(data)
#         return self.LR.score(output, response)

#     def Draw_Features(self):
#         from matplotlib import pylab as plt
#         #The first level features
#         num_f = self.Geometry[0]
#         weights_to_draw = np.concatenate(self.weights[0],axis=1)
#         for i in xrange(num_f):
#             if (i % 4 == 0):
#                 filename = str(i) + '-' + str(i + 3) + '_features.png'
#                 fig, axes = plt.subplots(2, 2)
#                 ax = axes.flat
#             im = ax[i % 4].imshow(weights_to_draw.T[i, :-1].reshape(16, 16), vmin=-0.3, vmax=0.3, interpolation='none')
#             if (i + 1 == num_f or i % 4 == 3):
#                 fig.subplots_adjust(right=0.8)
#                 cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#                 fig.colorbar(im, cax=cbar_ax)
#                 self.IO_agent.savefig(fig,filename )
