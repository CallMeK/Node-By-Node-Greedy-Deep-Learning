__author__ = 'wuk3'

import numpy as np
import sys

#This file contains functions for a normal multilayer neural network
#A final layer with softmax and other layers with tanh
#implemented with a SGD-minibatch


def groupby(classes):
    result = {}
    for i, x in enumerate(classes):
        if (x not in result):  # as key
            result[x] = []
        result[x].append(i)
    return result

#private

def AddIntercept(inputs):
    return np.hstack([inputs, np.ones([inputs.shape[0], 1])])


def AddIntercept_s(x):
    return np.append(x, 1)


def Forward_1(inputs, weights):
    return np.tanh(np.dot(inputs, weights))


def Forward_2(inputs, weights):
    result =  np.exp(np.dot(inputs, weights))
    if(len(result.shape) == 2):
        return result/result.sum(axis=1)[:,np.newaxis]
    elif(len(result.shape) == 1):
        return result/result.sum()

def Forward_dict_list(inputs,weights_dict):
    if(len(inputs.shape)==1):
        AddIntercept_ = AddIntercept_s
    else:
        AddIntercept_ = AddIntercept
    input_update = inputs
    result = [input_update]
    n_layer = len(weights_dict)
    for l in xrange(n_layer-1):
        weights = weights_dict[l]
        input_update = AddIntercept_(Forward_1(input_update, weights))
        result.append(input_update)
    return result

def Forward_all(inputs, weights_dict):
    n_layer = len(weights_dict)
    return Forward_2(Forward_dict_list(inputs, weights_dict)[-1], weights_dict[n_layer-1])

def Loss_Calc(inputs, weights_dict,response):
    n_layer = len(weights_dict)
    p = Forward_2(Forward_dict_list(inputs, weights_dict)[-1], weights_dict[n_layer-1])
    #response is vector indicate which node should be lit
    #here we assign the node number as 0-9 the same as the index
    #so the loss is simply the corresponding summation of the -t*log(p)
    p_correct = p[(np.arange(p.shape[0]),response)]
    return -np.sum(np.log(p_correct))

def Loss_Calc_LR(inputs, weights,response):
    p = Forward_2(inputs,weights)
    #response is vector indicate which node should be lit
    #here we assign the node number as 0-9 the same as the index
    #so the loss is simply the corresponding summation of the -t*log(p)
    p_correct = p[(np.arange(p.shape[0]),response)]
    return -np.sum(np.log(p_correct))



def AllNet_Train_SGD(inputs, response, weights_dict, IO_agent=None,alpha=0.001,epoch_limit=300,batch_size=20):
    #response is vector indicate which node should be lit, NOT the real response, which should be string


    if (IO_agent is None):
        from IO_Wrapper import IO_Wrapper_stdout
        printlog = IO_Wrapper_stdout.printlog
        printlog_loss = IO_Wrapper_stdout.printlog
    else:
        printlog = IO_agent.printlog
        printlog_loss = IO_agent.printlog_loss

    n_sample = inputs.shape[0]
    n_class = max(response) + 1
    n_layer = len(weights_dict)

    for i in xrange(1,n_layer):
        assert weights_dict[i-1].shape[1] == weights_dict[i].shape[0]-1

    #Setup the target matrix
    response_mat = np.zeros([n_sample,n_class])
    response_mat[(np.arange(n_sample),response)] = 1

    # now we do SGD
    epoch = 0
    printlog('FT Epoch_Limit: ', epoch_limit)

    n_batches = inputs.shape[0]/batch_size

    X_batches = np.array_split(inputs,n_batches)
    Y_batches = np.array_split(response_mat,n_batches)
    batch_indices = np.arange(n_batches)

    alphaX = alpha
    
    while (True):     
        np.random.shuffle(batch_indices)
        # if (epoch > 2000):
        #     alpha = alphaX / (np.log(epoch / 500.0))
        for indices in batch_indices:
            
            x = X_batches[indices]
            res = Y_batches[indices]
            bsize = len(x)
            output0_list = Forward_dict_list(x,weights_dict)

            output1 = Forward_2(output0_list[-1], weights_dict[n_layer-1]) 
            delta1 = (output1 - res)/bsize
            grad1 = np.dot(output0_list[-1].T, delta1)
            step1 = -alpha * grad1
            weights_dict[n_layer-1] += step1

            #The following part will not execute if it is logistic regression
            delta0 = delta1
            for l in np.arange(n_layer-1,0,-1):
                delta0 = np.multiply(np.dot(delta0, weights_dict[l][:-1].T),(1 - output0_list[l][:,:-1] ** 2))
                grad0 = np.dot(output0_list[l-1].T, delta0)

                step0 = -alpha * grad0
                weights_dict[l-1] += step0


        #print epoch,loss
        loss = Loss_Calc(inputs, weights_dict,response)
        printlog_loss(epoch, loss)
        #loss_prev = loss
        epoch = epoch + 1
        if (epoch == epoch_limit):
            break

    loss = Loss_Calc(inputs, weights_dict,response)
    printlog('Final Loss: ', loss)
    return epoch



def Logistic_Train_SGD(inputs, response, weights, IO_agent=None,alpha=0.001,epoch_limit=300,batch_size=20):
    if (IO_agent is None):
        from IO_Wrapper import IO_Wrapper_stdout
        printlog = IO_Wrapper_stdout.printlog
        printlog_loss = IO_Wrapper_stdout.printlog
    else:
        printlog = IO_agent.printlog
        printlog_loss = IO_agent.printlog_loss

    n_sample = inputs.shape[0]
    n_class = max(response) + 1

    #Setup the target matrix
    response_mat = np.zeros([n_sample,n_class])
    response_mat[(np.arange(n_sample),response)] = 1.0


    # now we do SGD
    epoch = 0
    printlog('Epoch_Limit: ', epoch_limit)

    n_batches = inputs.shape[0]/batch_size

    X_batches = np.array_split(inputs,n_batches)
    Y_batches = np.array_split(response_mat,n_batches)
    batch_indices = np.arange(n_batches)

    alphaX = alpha


    while (True):
        if (epoch > 100):
            alpha = alphaX / (1.0 + epoch/ 1000.0)
        np.random.shuffle(batch_indices)

        for indices in batch_indices:
            
            x = X_batches[indices]
            res = Y_batches[indices]
            bsize = len(x)

            output1 = Forward_2(x, weights)
            delta1 = output1 - res
            grad1 = np.dot(x.T, delta1) + 1.0/n_sample*np.append(weights[:-1],np.zeros([1,weights.shape[1]]),axis=0)*bsize
            step1 = -alpha * grad1/bsize
            weights += step1

        #print epoch,loss
        loss = Loss_Calc_LR(inputs, weights,response)
        printlog_loss(epoch, loss)
        #loss_prev = loss
        epoch = epoch + 1
        if (epoch == epoch_limit):
            break

    loss = Loss_Calc_LR(inputs, weights,response)
    printlog('Final Loss: ', loss)
    return epoch



if __name__ == '__main__':
    inputs = np.array([[-1.0, -0.5, 0.5, 1.0, 1]
        , [-0.5, -1.0, 1.0, 0.5, 1]
        , [0.5, 1.0, -0.5, -1.0, 1]
        , [1.0, 0.5, -1.0, -0.5, 1]])
    np.random.seed(33)
    weights0 = np.random.rand(5, 3) - 0.5
    weights1 = np.random.rand(4,2) - 0.5
    
    weights={}
    weights[0] = weights0.copy()
    weights[1] = weights1
    response = np.array([0,1,1,0])
    print weights
    # weights0 = np.zeros([4,5])
    crap = AllNet_Train_SGD(inputs, response,weights,alpha=0.1,epoch_limit=100,batch_size=2)
    print "after training"
    print weights
    print Forward_all(inputs,weights)
    weight_lr = np.random.rand(5, 2) - 0.5
    crap2 = Logistic_Train_SGD(inputs,response,weight_lr,alpha=0.1, epoch_limit=10,batch_size=2)
    print weights1
    print Forward_2(inputs,weight_lr)