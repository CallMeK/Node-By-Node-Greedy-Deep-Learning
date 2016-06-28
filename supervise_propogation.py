import numpy as np
import sys

##tanh for layer0 to layer1
##softmax for layer1 to layer2

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

def Forward_dict(inputs, weights_dict):
    input_update = inputs
    for weights in weights_dict.values():
        input_update = AddIntercept(Forward_1(input_update, weights))
    return input_update[:,:-1]


def Loss_Calc(inputs, w0, w1,response):
    p = Forward_2(AddIntercept(Forward_1(inputs, w0)), w1)
    #response is vector indicate which node should be lit
    #here we assign the node number as 0-9 the same as the index
    #so the loss is simply the corresponding summation of the -t*log(p)
    p_correct = p[(np.arange(p.shape[0]),response)]
    return -np.sum(np.log(p_correct))


def Train_SGD(inputs, response, weights0, IO_agent=None,alpha=0.001,epoch_limit=300):
    #response is vector indicate which node should be lit

    if (IO_agent is None):
        from IO_Wrapper import IO_Wrapper_stdout
        printlog = IO_Wrapper_stdout.printlog
        printlog_loss = IO_Wrapper_stdout.printlog
    else:
        printlog = IO_agent.printlog
        printlog_loss = IO_agent.printlog_loss

    # It is just a simple 3-layer neural network
    d_current = inputs.shape[1]  # with intercepts
    d_next = weights0.shape[1]
    d_output = max(response)+1 #it is also the n_class
    n_sample = inputs.shape[0]
    weights1 = (np.random.rand(d_next + 1, d_output) - 0.5) / (d_output)

    response_mat = np.zeros([n_sample,d_output])
    response_mat[(np.arange(n_sample),response)] = 1
    # now we do SGD
    epoch = 0
    printlog('Epoch_Limit: ', epoch_limit)

    alphaX = alpha
    while (True):
        if (epoch == epoch_limit):
            break
        if (epoch > 100):
            alpha = alphaX / (1.0 + epoch/1000.0)
        for (x,res) in zip(inputs,response_mat):
            output0 = Forward_1(x, weights0)
            output0_inpt = AddIntercept_s(output0)
            output1 = Forward_2(output0_inpt, weights1)
            delta1 = output1 - res
            delta0 = np.dot(delta1, weights1[:-1].T) * (1 - output0 ** 2)

            grad1 = np.outer(output0_inpt, delta1)
            grad0 = np.outer(x, delta0)

            '''
                print 'x',x
                print "weights0: ",weights0
                print "weights1: ",weights1
                print 'output0',output0
                print 'output1',output1

                print "grad0: ",grad0
                print "grad1: ",grad1
                '''

            step1 = -alpha * grad1
            step0 = -alpha * grad0
            weights0 += step0
            weights1 += step1




        # loss = Loss_Calc_dot(inputs, weights0, weights1)
        # if( abs(loss-loss_prev) / loss_prev < tol):
        # 	delay_counter += 1
        # elif(delay_counter != 0):
        # 	delay_counter = 0
        # if(delay_counter >= delay):
        # 	break

        #print epoch,loss
        loss = Loss_Calc(inputs, weights0, weights1,response)
        printlog_loss(epoch, loss)
        #loss_prev = loss
        epoch = epoch + 1


    loss = Loss_Calc(inputs, weights0, weights1,response)
    printlog('Final Loss: ', loss)
    return (epoch, weights1)

if __name__ == '__main__':
    inputs = np.array([[-1.0, -0.5, 0.5, 1.0, 1]
        , [-0.5, -1.0, 1.0, 0.5, 1]
        , [0.5, 1.0, -0.5, -1.0, 1]
        , [1.0, 0.5, -1.0, -0.5, 1]])
    np.random.seed(33)
    weights0 = np.random.rand(5, 4) - 0.5
    response = np.array([0,1,1,0])
    print weights0
    # weights0 = np.zeros([4,5])
    crap, w1 = Train_SGD(inputs, response,weights0,alpha=0.1)
    print "after training"
    print weights0
    print Forward_2(AddIntercept(Forward_1(inputs, weights0)), w1)


