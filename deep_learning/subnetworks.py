import tensorflow as tf
from tensorflow import math
from deepchem.models.layers import Highway, DTNNEmbedding
from tensorflow.keras.layers import AlphaDropout, Activation, Dense, Dropout, BatchNormalization, Input, Conv1D, Lambda, Concatenate, add, LSTM, Bidirectional
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU, PReLU
from ast import literal_eval
from tensorflow.keras.regularizers import l1_l2
from inspect import getmembers
from tensorflow.keras.metrics import MeanSquaredError, RootMeanSquaredError
from spektral.layers import GCNConv, GlobalSumPool, GATConv



def dense_submodel(input_layer, hlayers_sizes='[10]', l1_regularization=0, l2_regularization=0,
                   hidden_activation='relu', input_dropout=0, hidden_dropout=0):
	"""
	Build a dense (fully-connected) subnetwork.
	
	Parameters
	----------
	input_layer : keras.layers.Layer
        Input layer of the subnetwork.
    hlayers_sizes : str
        String representation of a list of integers. Each integer represents the number of neurons in a hidden layer.
    l1_regularization : float
        L1 regularization factor.
    l2_regularization : float
        L2 regularization factor.
    hidden_activation : str
        Activation function to use in the hidden layers.
    input_dropout : float
        Dropout rate for the input layer.
    hidden_dropout : float
        Dropout rate for the hidden layers.
		
    Returns
    -------
    keras.layers.Layer
        Output layer of the subnetwork.
		"""
	hlayers_sizes = literal_eval(hlayers_sizes)  # because hlayers_sizes was passed as a string

	if hidden_activation == 'selu':
		# selu must be used with the LecunNormal initializer and AlphaDropout instead of normal Dropout
		initializer = 'lecun_normal'
		dropout = AlphaDropout
		batchnorm = False
	else:
		initializer = 'he_normal'
		dropout = Dropout
		batchnorm = True

	if input_dropout > 0:
		x = dropout(rate=input_dropout)(input_layer)
	else:
		x = input_layer

	for i in range(len(hlayers_sizes)):
		x = Dense(units=hlayers_sizes[i], kernel_initializer=initializer,
		          kernel_regularizer=l1_l2(l1=l1_regularization, l2=l2_regularization))(x)
		if hidden_activation.lower() == 'leakyrelu':
			x = LeakyReLU()(x)
		elif hidden_activation.lower() == 'prelu':
			x = PReLU()(x)
		else:
			x = Activation(hidden_activation)(x)

		if batchnorm:
			x = BatchNormalization()(x)

		if hidden_dropout > 0:
			x = dropout(rate=hidden_dropout)(x)

	return x



def lstm_submodel(input_layer, hlayers_sizes='[10]', l1_regularization=0, l2_regularization=0,
                   hidden_activation='relu', input_dropout=0, hidden_dropout=0):
	"""
	Build a dense (fully-connected) subnetwork.
	
	Parameters
	----------
	input_layer : keras.layers.Layer
        Input layer of the subnetwork.
    hlayers_sizes : str
        String representation of a list of integers. Each integer represents the number of neurons in a hidden layer.
    l1_regularization : float
        L1 regularization factor.
    l2_regularization : float
        L2 regularization factor.
    hidden_activation : str
        Activation function to use in the hidden layers.
    input_dropout : float
        Dropout rate for the input layer.
    hidden_dropout : float
        Dropout rate for the hidden layers.
		
    Returns
    -------
    keras.layers.Layer
        Output layer of the subnetwork.
		"""
	hlayers_sizes = literal_eval(hlayers_sizes)  # because hlayers_sizes was passed as a string

	if hidden_activation == 'selu':
		# selu must be used with the LecunNormal initializer and AlphaDropout instead of normal Dropout
		initializer = 'lecun_normal'
		dropout = AlphaDropout
		batchnorm = False
	else:
		initializer = 'he_normal'
		dropout = Dropout
		batchnorm = True

	if input_dropout > 0:
		x = dropout(rate=input_dropout)(input_layer)
	else:
		x = input_layer

	for i in range(len(hlayers_sizes)):
		return_sequences = (i < len(hlayers_sizes) - 1) 
		x = Bidirectional(LSTM(units=hlayers_sizes[i], kernel_initializer=initializer,
		          kernel_regularizer=l1_l2(l1=l1_regularization, l2=l2_regularization),
				  return_sequences= return_sequences))(x)
		if hidden_activation.lower() == 'leakyrelu':
			x = LeakyReLU()(x)
		elif hidden_activation.lower() == 'prelu':
			x = PReLU()(x)
		else:
			x = Activation(hidden_activation)(x)

		if batchnorm:
			x = BatchNormalization()(x)

		if hidden_dropout > 0:
			x = dropout(rate=hidden_dropout)(x)

	return x


def drugcnn_submodel(seq_length, char_dict, n_embedding=75,
                     kernel_sizes='[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]',
                     num_filters='[100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]',
                     l1=0, l2=0, dropout=0.25):
    
    """
    Build a CNN subnetwork for drug SMILES.
	
	Parameters
	----------
	seq_length : int
        Length of the SMILES sequences.
    char_dict : str
        String representation of a dictionary mapping each character to an integer.
    n_embedding : int
        Dimension of the embedding space.
    kernel_sizes : str
        String representation of a list of integers. Each integer represents the size of a kernel.
    num_filters : str
        String representation of a list of integers. Each integer represents the number of filters for a kernel.
    l1 : float
        L1 regularization factor.
    l2 : float
        L2 regularization factor.
    dropout : float
        Dropout rate.
        
    Returns
    -------
    keras.layers.Layer
        Output layer of the subnetwork.
        """
   
    input_layer = Input(shape=(seq_length,), dtype=tf.int32, name='drug')
    kernel_sizes = literal_eval(kernel_sizes)
    num_filters = literal_eval(num_filters)
    kernel_regularizer = l1_l2(l1=l1, l2=l2)
    # char_dict = dict(char_dict)
    char_dict = literal_eval(char_dict)

    embedding = DTNNEmbedding(
        n_embedding=n_embedding,
        periodic_table_length=len(char_dict.keys()) + 1)(input_layer)

    pooled_outputs = []
    conv_layers = []
	
    for filter_size, num_filter in zip(kernel_sizes, num_filters):
        # Multiple convolutional layers with different filter widths
        conv_layers.append(
            Conv1D(kernel_size=filter_size, filters=num_filter, padding='valid',
                kernel_regularizer=kernel_regularizer)(embedding))
        # Max-over-time pooling
        reduced = Lambda(lambda x: math.reduce_max(x, axis=1))(conv_layers[-1])
        pooled_outputs.append(reduced)
		
    # Concat features from all filters(one feature per filter)
    concat_outputs = Concatenate(axis=1)(pooled_outputs)
    dropout = Dropout(rate=dropout)(concat_outputs)
    dense = Dense(200, activation='relu')(dropout)
    # Highway layer from https://arxiv.org/pdf/1505.00387.pdf
    gather = Highway()(dense)

    submodel = tf.keras.Model(inputs=input_layer, outputs=gather, name='drugcnn_submodel')

    return submodel




def gcn_submodel(n_atom_features, gcn_layers='[64, 64]', residual=False, activation='relu',
                 dropout_rate=0.5, l2=0):
	"""Build a Graph Convolutional Network (GCN) (Kipf et al, 2017) submodel"""
	# I'm using a different kernel initializer though
	# GlobalSumPool was not part of the original GCN model, but is necessary to get a graph-level embedding
	gcn_layers = literal_eval(gcn_layers)
	regularizer = l1_l2(l1=0, l2=l2)
	# nodes_input = Input(shape=(max_n_atoms, drug_n_atom_features))
	# adjacency_input = Input(shape=(max_n_atoms, max_n_atoms))
	nodes_input = Input(shape=(None, n_atom_features))
	adjacency_input = Input(shape=(None, None))
	node_feats = nodes_input
	for n_channels in gcn_layers:
		x = GCNConv(n_channels, activation, kernel_initializer='he_normal',
		            kernel_regularizer=regularizer)([node_feats, adjacency_input])
		if residual:  # add a drug_residual_connection connection (as implemented in GCNPredictor from DGL LifeSci)
			res_feats = Dense(n_channels, activation=activation)(
				node_feats)  # the DGL LifeSci code does this first. I think it's needed because the shape of the original inputs is not the same as the outputs of GCNConv
			x = add([x, res_feats])  # drug_residual_connection connection
		x = Dropout(dropout_rate)(x)
		node_feats = x
	x = GlobalSumPool()(x)

	submodel = tf.keras.Model(inputs=[nodes_input, adjacency_input], outputs=[x], name='drug_gcn_submodel')

	return submodel


def gat_submodel(n_atom_features, gat_layers='[64, 64]', n_attention_heads=8, concat_heads=True, residual=False,
                 dropout_rate=0.5, l2=0):
	"""Build a Graph Attention Network (GAT) (Velickovic et al, 2018) submodel."""
	gat_layers = literal_eval(gat_layers)
	regularizer = l1_l2(l1=0, l2=l2)
	nodes_input = Input(shape=(None, n_atom_features))
	adjacency_input = Input(shape=(None, None))
	node_feats = nodes_input
	for n_channels in gat_layers:
		x = GATConv(n_channels, activation='elu', attn_heads=n_attention_heads, concat_heads=concat_heads,
		            dropout_rate=dropout_rate, kernel_initializer='he_normal',
		            kernel_regularizer=regularizer, attn_kernel_regularizer=regularizer,
		            bias_regularizer=regularizer)([node_feats, adjacency_input])
		if residual:  # add a drug_residual_connection connection (as in the implementation of GATPredictor in DGL LifeSci)
			if concat_heads:
				res_feats = Dense(n_channels * n_attention_heads)(
					node_feats)  # according to the implementation of residual connections for GATConv in DGL
			else:
				res_feats = Dense(n_channels)(node_feats)
			x = add([x, res_feats])  # drug_residual_connection connection
		x = Dropout(dropout_rate)(x)
		node_feats = x
	x = GlobalSumPool()(x)

	submodel = tf.keras.Model(inputs=[nodes_input, adjacency_input], outputs=[x], name='drug_gat_submodel')

	return submodel