import numpy as np

class DataWrapper(object):
    def __init__(self,inputs,response,QuickTest = False,IO_agent=None):
        #method can be SGD or Batch
        if(IO_agent is None):
            #only for debug, print to stdout
            from IO_Wrapper import IO_Wrapper_stdout
            self.IO_agent = IO_Wrapper_stdout
            optionlist = None
        else:
            self.IO_agent = IO_agent
            if(self.IO_agent.get_params()['option_list'] is not None):
                optionlist = [x.strip() for x in self.IO_agent.get_params()['option_list'].split(',')]
            else:
                optionlist = None

        self.QuickTest = QuickTest
        self.n_quicktest = 100
        self.inputs = np.load(inputs)
        self.inputs = np.hstack([self.inputs,np.ones([len(self.inputs),1])]) #add the intercept 1's here
        self.response = np.load(response)
        if(self.QuickTest):
            self.IO_agent.printlog("QuickTest: N is set to be ",self.n_quicktest)
            self.inputs = self.inputs[:self.n_quicktest]
            self.response = self.response[:self.n_quicktest]

        if(optionlist is None):
            self.IO_agent.printlog("All data will be used")

        else:
            self.optionlist = optionlist
            self.IO_agent.printlog("Data will be used from classes: ",self.optionlist)
            self.option_mask = np.array([True if i in self.optionlist else False for i in self.response])
            self.inputs = self.inputs[self.option_mask]
            self.response = self.response[self.option_mask]
        self.IO_agent.printlog("Response list: ", np.unique(self.response))
        self.IO_agent.printlog("Number of data:",self.inputs.shape[0])
        self.IO_agent.printlog("Number of Features: ",self.inputs.shape[1]-1)

    def length(self):
        return len(self.inputs)

    def draw_pic(self,index):
        #only for USPS data, with the constant 1
        from matplotlib import pylab as plt
        fig, ax = plt.subplots()
        im = ax.imshow(self.inputs[index][:-1].reshape(16,16),vmin=-1,vmax=1,interpolation='none')
        fig.savefig("Fig: "+str(index))


if __name__ == '__main__':
    data_train = DataWrapper('digit_train_noise_x10.npy','digit_train_response.npy')
    data_train.draw_pic(1)
