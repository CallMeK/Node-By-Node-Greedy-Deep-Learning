# The basic stuff that a model should do
#report training score
#report test score
#draw features

import numpy as np
import sys
from sklearn.linear_model import LogisticRegression

class Model:
    def __init__(self, data, io_agent):
        self.params = io_agent.get_params()
        Geometry = [int(x.strip()) for x in self.params['Geometry'].split(',')]
        #optional parameters: 
        if(not self.params.has_key('AF')):
          AF=1.0
        else:
          AF = self.params['AF']
        if(not self.params.has_key('alpha_ft')):
          alpha_ft = 0.1
        else:
          alpha_ft = float(self.params['alpha_ft'])
        if(not self.params.has_key('epoch_limit_ft')):
          epoch_limit_ft = 100
        else:
          epoch_limit_ft = int(self.params['epoch_limit_ft'])


        if(self.params['method'] == 'normal'):
            #new algorithms are also implemented in this one

            from Methods import NormalMethod
            self.method = NormalMethod(Geometry,
                                       self.params['opt_method'],
                                       io_agent,
                                       float(self.params['alpha']),
                                       int(self.params['epoch_limit']),
                                       alpha_ft,
                                       epoch_limit_ft,
                                       self.params['FT']=='Y',
                                       float(AF),
                                       int(self.params['batch_size']))
        elif(self.params['method'] == 'super'):
            #supervised pretraining, classic method
            from Methods import SuperMethod
            self.method = SuperMethod(Geometry,
                                       self.params['opt_method'],
                                       io_agent,
                                       float(self.params['alpha']),
                                       int(self.params['epoch_limit']),
                                       alpha_ft,
                                       epoch_limit_ft,
                                       self.params['FT']=='Y',
                                       int(self.params['batch_size']))
        elif(self.params['method'] == 'NN'):
            from Methods import BasicNN
            self.method = BasicNN(Geometry,
                                       self.params['opt_method'],
                                       io_agent,
                                       float(self.params['alpha']),
                                       int(self.params['epoch_limit']),
                                       alpha_ft,
                                       epoch_limit_ft,
                                       int(self.params['batch_size']))
        else:
            raise Exception("Unknown Method")
        self.data = data
        self.io_agent = io_agent
        self.io_agent.printlog("The Geometry was set as: ",Geometry)

    def train(self):
        self.io_agent.printlog("//////////////////////////////////////////////////////////////////////////////")
        self.io_agent.printlog("Starting the modeling...")
        self.io_agent.printlog("Number of data (N): ",self.data.length())
        self.method.train(self.data.inputs, self.data.response)

    def report_train(self):
        score = self.method.get_score(self.data.inputs, self.data.response)
        self.io_agent.printlog("Training score: ",score)

    def report_test(self, data_test):
        self.io_agent.printlog("Number of data in test: ", data_test.length())
        score = self.method.get_score(data_test.inputs, data_test.response)
        self.io_agent.printlog("Test score: ",score)

    def report_LRonly(self):
        self.io_agent.printlog('Start Logistic Regression Only...')
        self.LR_Only = LogisticRegression(C=1.0, tol=0.0001)
        self.LR_Only.fit(self.data.inputs[:,:-1], self.data.response)
        self.io_agent.printlog('Training Score: ',self.LR_Only.score(self.data.inputs[:,:-1], self.data.response))

    def report_LRonlytest(self,data_test):
        self.io_agent.printlog('Start Test on Logistic Regression...')
        self.io_agent.printlog('Test score: ',self.LR_Only.score(data_test.inputs[:,:-1], data_test.response))


    def Draw_Feature(self):
        self.method.Draw_Features()

    def Save_Weights(self):
        self.io_agent.saveweights(self.method.weight0prev,name='weight0prev.p')
        self.io_agent.saveweights(self.method.weights)