#main.py

#exmaple

from DataWrapper import DataWrapper
from Model import Model
from IO_Wrapper import IO_Wrapper
import sys
import time

#IO_Wrapper defines the destination for all output
IO_agent = IO_Wrapper(sys.argv[1],'test.txt',sys.argv[1].split('.txt')[0])

#The argument QuickTest is used for debugging and quick test
data_train = DataWrapper('cancer_train_X.npy','cancer_train_resp.npy',QuickTest=False,
                         IO_agent = IO_agent)
data_test = DataWrapper('cancer_test_X.npy','cancer_test_resp.npy',QuickTest=False,
                        IO_agent = IO_agent)


DL = Model(data_train,IO_agent)


#Logistic Regression, only for comparison
DL.report_LRonly()
DL.report_LRonlytest(data_test)

start = time.clock()
start_t = time.time()
DL.train()
end = time.clock()
end_t = time.time()
DL.report_train()
DL.report_test(data_test)
IO_agent.printlog('CPU time (time.clock): ',end-start,'s')
IO_agent.printlog('Wall time (time.time): ', end_t-start_t,'s')
DL.Save_Weights()
