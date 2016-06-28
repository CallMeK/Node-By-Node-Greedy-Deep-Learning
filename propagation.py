import numpy as np
import sys

#public-----------------------------------------------------------------------

def AddIntercept(inputs):
    return np.hstack([inputs, np.ones([inputs.shape[0], 1])])


def Forward_1(inputs, weights):
    return np.tanh(np.dot(inputs, weights))


def Forward_dict(inputs, weights_dict):
    input_update = inputs
    for weights in weights_dict.values():
        input_update = AddIntercept(Forward_1(input_update, weights))
    return input_update[:,:-1]


#private----------------------------------------------------------------------

def AddIntercept_s(x):
    return np.append(x, 1)


def Forward_2(inputs, weights):
    return np.dot(inputs, weights)


def Loss_Calc(inputs, w0, w1):
    y = Forward_2(AddIntercept(Forward_1(inputs, w0)), w1)
    n_sample = inputs.shape[0]
    return 0.5 * np.linalg.norm(y - inputs[:, :-1]) ** 2 / n_sample


def Train_SGD(inputs, response,weights0, IO_agent=None,alpha=0.001,epoch_limit=300,AF=None):

    #AF is only set for the consistency
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
    n_sample = inputs.shape[0]
    weights1 = (np.random.rand(d_next + 1, d_current - 1) - 0.5) / (d_current+d_next)

    # now we do SGD
    epoch = 0
    printlog('Epoch_Limit: ', epoch_limit)

    alphaX = alpha
    while (True):
        if (epoch == epoch_limit):
            break
        if (epoch > 100):
            alpha = alphaX / (1.0 + epoch/1000.0)
        for x in inputs:
            output0 = Forward_1(x, weights0)
            output0_inpt = AddIntercept_s(output0)
            output1 = Forward_2(output0_inpt, weights1)
            delta1 = output1 - x[:-1]
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

        #print epoch,loss, but only on d_next times
        if(epoch > epoch_limit-d_next-1):
            loss = Loss_Calc(inputs, weights0, weights1)
            printlog_loss(epoch, loss)
        #loss_prev = loss
        epoch = epoch + 1
        

    loss = Loss_Calc(inputs, weights0, weights1)
    printlog('Final Loss: ', loss)
    return (epoch, weights1)

if __name__ == '__main__':
    inputs = np.array([[-1.0, -0.5, 0.5, 1.0, 1]
        , [-0.5, -1.0, 1.0, 0.5, 1]
        , [0.5, 1.0, -0.5, -1.0, 1]
        , [1.0, 0.5, -1.0, -0.5, 1]])
    np.random.seed(33)
    weights0 = np.random.rand(5, 4) - 0.5
    print weights0
    # weights0 = np.zeros([4,5])
    crap, w1 = Train_SGD(inputs, weights0)
    print "after training"
    print weights0
    print Forward_2(AddIntercept(Forward_1(inputs, weights0)), w1)


