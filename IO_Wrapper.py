__author__ = 'wuk3'

#All IO's should be here
#printlog, read_inputs and savefig

class IO_Wrapper(object):
    def __init__(self,input_file, output_file,foldername):
        self.output_file = output_file
        self.foldername = foldername
        self.inputfile = input_file
        self.result = {}
        with open(self.inputfile,'r') as f:
            lines = f.readlines()
            for line in lines:
                data = line.split(":")
                if(len(data[1].strip())!=0):
                    self.result[data[0].strip()] = data[1].strip()
                else:
                    self.result[data[0].strip()] = None
        print self.result

    def get_params(self):
        return self.result

    def printlog(self,*info):
        import os
        if not os.path.exists(self.foldername):
            os.mkdir(self.foldername)
        f=open(self.foldername+"/"+self.output_file,'a+')
        for item in info:
            if(type(item) is not str):
                f.write(str(item)+' ')
            else:
                f.write(item)
        f.write('\n')
        f.close()

    def printlog_loss(self,*info):
        import os
        if not os.path.exists(self.foldername):
            os.mkdir(self.foldername)
        f=open(self.foldername+"/loss.txt",'a+')
        for item in info:
            if(type(item) is not str):
                f.write(str(item)+' ')
            else:
                f.write(item)
        f.write('\n')
        f.close()

    def savefig(self,fig,filename):
        fig.savefig(self.foldername+"/"+filename)

    def saveweights(self,weights_dict,name='weights.p'):
        import cPickle as pickle
        pickle.dump(weights_dict,open(self.foldername+'/'+name,'w'))

class IO_Wrapper_stdout(object):
    @staticmethod
    def printlog(*info):
        for item in info:
            if(type(item) is not str):
                print str(item)+' ',
            else:
                print item+' ',
        print ''

    @staticmethod
    def save_fig(self,fig,filename):
        fig.savefig(filename)