Ke (Kevin) Wu, PhD
Rensselaer Polytechnic Institute, 5.29.2016
wkcoke.work@gmail.com

This program implements four pretraining algorithms that have been studied in the paper
"Efficient Node-By-Node Greedy Deep Learning for Interpretable Feature Representation",
submitted to ECML06.

The four algorithms are supervised pretraining, unsupervised pretraining (autoencoder), greedy-by-node (GN)
and greedy-by-class-by-node (GCN).

This program is designed to solely deal with a cleaned dataset, that means the user is responsible for creating
the training X, training response, test X and test response. All the data file should in numpy format using
np.save(), please refer to the example file

A main file should be written by user. There is one example "main_cancer.py”

The parameters are set using a configuration file with flexible format.


//***************************************************************************//

For the parameter set:

method: Defines how the upper-level structure, for the current code, the value can be either "super"
(for supervised pretraining), or "normal" (for the other three methods)

opt_method : defines the optimization routine, the four algorithms have different way of training a single layer. "unsup" is for autoencoder, "super" is for supervised pretraining, "greedy" is for GN and "greedy_class" is for GCN. 
As for optimization algorithm, SGD is the only optimization method implemented. 



So the combinations are: 

method: 	super	normal	normal	normal
opt_method: 	super	unsup	greedy greedy_class


option_list is used for choosing a certain subset of classes. For example, one can train only on 7 and 9 by define 7,9. This function is enabled through the DataWrapper Module.

alpha : learning rate, please refer to the paper

epoch_limit : the epoches for pretraining.

Geometry : defines the internal network structure (no input and output layer)

FT: Y for turning on fine tuning, N for no output layer and fine-tuning.

AF: amnesia factor, default to be 1.0, please refer to the paper.

alpha_ft: learning rate for the fine-tuning and logistic regression layer (final layer)

epoch_limit_ft: epochs for the fine-tuning

batchsize: batch_size for fine-tuning and logistic regression layer (final layer)

There are examples in the folder, using the Wisconsin Cancer data.

//***************************************************************************//
Note: 

Based on a limit number of heuristic test, the total number of updates for a single-internal-node network should next be less than 4/alpha. If a large network is used for a small data set, user should increase the epochs.

//***************************************************************************//


 



