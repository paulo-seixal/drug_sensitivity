from subnetworks import dense_submodel, gcn_submodel, drugcnn_submodel, gat_submodel, lstm_submodel
from ast import literal_eval
import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate, Dense
from tensorflow.keras import optimizers
from inspect import getmembers


class DenseModel:
    """
    Multi-input model for drug response prediction.
    Uses a dense subnetwork for both cell line and drug features.

    Parameters
    ----------
    expr_dim : int
        Dimension of the cell line feature space.
    drug_dim : int
        Dimension of the drug feature space.
    expr_hlayers_sizes : str
        String representation of a list of integers. Each integer represents the size of a hidden layer in the cell line subnetwork.
    drug_hlayers_sizes : str
        String representation of a list of integers. Each integer represents the size of a hidden layer in the drug subnetwork.
    predictor_hlayers_sizes : str
        String representation of a list of integers. Each integer represents the size of a hidden layer in the final network.
    initializer : str
        Initializer for the weights of the dense layers.
    hidden_activation : str
        Activation function for the hidden layers.
    l1 : float
        L1 regularization factor for the final network.
    l2 : float
        L2 regularization factor for the final network.
    input_dropout : float
        Dropout rate for the input layer.
    hidden_dropout : float
        Dropout rate for the hidden layers.
    optimizer : str
        Optimizer for the model.
    learn_rate : float
        Learning rate for the optimizer.

    Returns
    -------
    keras.Model
        Multi-input model for drug response prediction.
        
    """
    def __init__(self, expr_dim=None, drug_dim=None, expr_hlayers_sizes='[10]', drug_hlayers_sizes='[10]',
                          predictor_hlayers_sizes='[10]', initializer='he_normal', hidden_activation='relu', l1=0,
                          l2=0, input_dropout=0, hidden_dropout=0, optimizer='Adam', learn_rate=0.001):
        self.cell_line_feature_size = expr_dim
        self.drug_feature_size = drug_dim
        self.expr_hlayers_sizes = expr_hlayers_sizes
        self.drug_hlayers_sizes = drug_hlayers_sizes
        self.predictor_hlayers_sizes = predictor_hlayers_sizes
        self.initializer = initializer
        self.hidden_activation = hidden_activation
        self.l1 = l1
        self.l2 = l2
        self.input_dropout = input_dropout
        self.hidden_dropout = hidden_dropout
        self.optimizer = optimizer
        self.learn_rate = learn_rate
        self.model = self.build_model()

    def build_model(self):
        # Define input layers
        cell_line_input = Input(shape=(self.cell_line_feature_size,), name='cell_line_input')
        drug_input = Input(shape=(self.drug_feature_size,), name='drug_input')

        # Define subnetworks for cell line and drug
        cell_line_subnet = dense_submodel(cell_line_input)
        drug_subnet = dense_submodel(drug_input)

        # Concatenate the outputs of the attended subnetworks
        concat = Concatenate(name = 'input_layer_concat')([cell_line_subnet, drug_subnet])

        # Final network for IC50 prediction
        final_network = dense_submodel(concat, hlayers_sizes=self.predictor_hlayers_sizes,
	                             l1_regularization=self.l1, l2_regularization=self.l2,
	                             hidden_activation=self.hidden_activation, input_dropout=0,
	                             hidden_dropout=self.hidden_dropout)
        
        final_network = Dense(1, activation='linear', name='ic50_prediction_dense_output')(final_network)


        # Create the model
        model = tf.keras.Model(inputs=[cell_line_input, drug_input], outputs=final_network)

        # Define optimizer
        opt_class = dict(getmembers(optimizers))[self.optimizer]
        if self.optimizer == 'SGD':
            opt = opt_class(lr=self.learn_rate, decay=1e-6, momentum=0.9, nesterov=True)
        else:
            opt = opt_class(lr=self.learn_rate)

        # Compile the model
        model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mae'])

        return model

    def summary(self):
        self.model.summary()

    def train(self, cell_line_train_data, drug_train_data, train_labels, epochs, batch_size):
        self.history = self.model.fit(
            {'cell_line_input': cell_line_train_data, 'drug_input': drug_train_data},
            {'ic50_prediction_dense_output': train_labels},
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2
        )

    def evaluate(self, cell_line_test_data, drug_test_data, test_labels):
        return self.model.evaluate(
            {'cell_line_input': cell_line_test_data, 'drug_input': drug_test_data},
            {'ic50_prediction_dense_output': test_labels}
        )

    def predict(self, new_cell_line_data, new_drug_data):
        return self.model.predict({'cell_line_input': new_cell_line_data, 'drug_input': new_drug_data})
    



class LSTMModel:
    """
    Multi-input model for drug response prediction.
    Uses a lstm subnetwork for both cell line and drug features.

    Parameters
    ----------
    expr_dim : int
        Dimension of the cell line feature space.
    drug_dim : int
        Dimension of the drug feature space.
    expr_hlayers_sizes : str
        String representation of a list of integers. Each integer represents the size of a hidden layer in the cell line subnetwork.
    drug_hlayers_sizes : str
        String representation of a list of integers. Each integer represents the size of a hidden layer in the drug subnetwork.
    predictor_hlayers_sizes : str
        String representation of a list of integers. Each integer represents the size of a hidden layer in the final network.
    initializer : str
        Initializer for the weights of the dense layers.
    hidden_activation : str
        Activation function for the hidden layers.
    l1 : float
        L1 regularization factor for the final network.
    l2 : float
        L2 regularization factor for the final network.
    input_dropout : float
        Dropout rate for the input layer.
    hidden_dropout : float
        Dropout rate for the hidden layers.
    optimizer : str
        Optimizer for the model.
    learn_rate : float
        Learning rate for the optimizer.

    Returns
    -------
    keras.Model
        Multi-input model for drug response prediction.
        
    """
    def __init__(self, expr_dim=None, drug_dim=None, expr_hlayers_sizes='[10]', drug_hlayers_sizes='[10]',
                          predictor_hlayers_sizes='[10]', initializer='he_normal', hidden_activation='relu', l1=0,
                          l2=0, input_dropout=0, hidden_dropout=0, optimizer='Adam', learn_rate=0.001):
        self.cell_line_feature_size = expr_dim
        self.drug_feature_size = drug_dim
        self.expr_hlayers_sizes = expr_hlayers_sizes
        self.drug_hlayers_sizes = drug_hlayers_sizes
        self.predictor_hlayers_sizes = predictor_hlayers_sizes
        self.initializer = initializer
        self.hidden_activation = hidden_activation
        self.l1 = l1
        self.l2 = l2
        self.input_dropout = input_dropout
        self.hidden_dropout = hidden_dropout
        self.optimizer = optimizer
        self.learn_rate = learn_rate
        self.model = self.build_model()

    def build_model(self):
        # Define input layers
        cell_line_input = Input(shape=(self.cell_line_feature_size,), name='cell_line_input')
        drug_input = Input(shape=(self.drug_feature_size,), name='drug_input')

        # Define subnetworks for cell line and drug
        cell_line_subnet = dense_submodel(cell_line_input)
        drug_subnet = lstm_submodel(drug_input)

        # Concatenate the outputs of the attended subnetworks
        concat = Concatenate(name = 'input_layer_concat')([cell_line_subnet, drug_subnet])

        # Final network for IC50 prediction
        final_network = dense_submodel(concat, hlayers_sizes=self.predictor_hlayers_sizes,
	                             l1_regularization=self.l1, l2_regularization=self.l2,
	                             hidden_activation=self.hidden_activation, input_dropout=0,
	                             hidden_dropout=self.hidden_dropout)
        
        final_network = Dense(1, activation='linear', name='ic50_prediction_dense_output')(final_network)


        # Create the model
        model = tf.keras.Model(inputs=[cell_line_input, drug_input], outputs=final_network)

        # Define optimizer
        opt_class = dict(getmembers(optimizers))[self.optimizer]
        if self.optimizer == 'SGD':
            opt = opt_class(lr=self.learn_rate, decay=1e-6, momentum=0.9, nesterov=True)
        else:
            opt = opt_class(lr=self.learn_rate)

        # Compile the model
        model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mae'])

        return model

    def summary(self):
        self.model.summary()

    def train(self, cell_line_train_data, drug_train_data, train_labels, epochs, batch_size):
        self.history = self.model.fit(
            {'cell_line_input': cell_line_train_data, 'drug_input': drug_train_data},
            {'ic50_prediction_dense_output': train_labels},
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2
        )

    def evaluate(self, cell_line_test_data, drug_test_data, test_labels):
        return self.model.evaluate(
            {'cell_line_input': cell_line_test_data, 'drug_input': drug_test_data},
            {'ic50_prediction_dense_output': test_labels}
        )

    def predict(self, new_cell_line_data, new_drug_data):
        return self.model.predict({'cell_line_input': new_cell_line_data, 'drug_input': new_drug_data})
    



DEFAULT_CHAR_DICT_STR = str({'#': 1, '(': 2, ')': 3, '+': 4, '-': 5, '/': 6, '1': 7, '2': 8, '3': 9,
                             '4': 10, '5': 11, '6': 12, '7': 13, '8': 14,'=': 15, 'C': 16, 'F': 17,
                             'H': 18, 'I': 19, 'N': 20, 'O': 21, 'P': 22, 'S': 23, '[': 24, '\\': 25,
                             ']': 26, '_': 27, 'c': 28, 'Cl': 29, 'Br': 30, 'n': 31, 'o': 32, 's': 33,
                             '.': 34, 'Pt': 35, '@':36, 'B': 37, 'r': 38, 'l': 39, 'a': 40, 'i': 41, '9': 42})


class DrugCNNModel:
    """
    Multi-input model for drug response prediction.
    Uses a CNN subnetwork for drug SMILES and a dense subnetwork for cell line features.

    Parameters
    ----------
    expr_dim : int
        Dimension of the cell line feature space.
    drug_dim : int
        Dimension of the drug feature space.
    expr_hlayers_sizes : str
        String representation of a list of integers. Each integer represents the size of a hidden layer in the cell line subnetwork.
    predictor_hlayers_sizes : str
        String representation of a list of integers. Each integer represents the size of a hidden layer in the final network.
    initializer : str
        Initializer for the weights of the dense layers.
    hidden_activation : str
        Activation function for the hidden layers.
    drug_seq_length : int
        Length of the SMILES sequences.
    drug_n_embedding : int
        Dimension of the embedding space.
    drug_char_dict : str
        String representation of a dictionary mapping each character to an integer.
    drug_kernel_sizes : str
        String representation of a list of integers. Each integer represents the size of a kernel.
    drug_num_filters : str
        String representation of a list of integers. Each integer represents the number of filters for a kernel.
    drug_dropout : float
        Dropout rate for the drug subnetwork.
    drug_l1 : float
        L1 regularization factor for the drug subnetwork.
    drug_l2 : float
        L2 regularization factor for the drug subnetwork.
    l1 : float
        L1 regularization factor for the final network.
    l2 : float
        L2 regularization factor for the final network.
    input_dropout : float
        Dropout rate for the input layer.
    hidden_dropout : float
        Dropout rate for the hidden layers.
    optimizer : str
        Optimizer for the model.
    learn_rate : float
        Learning rate for the optimizer.
    
    Returns
    -------
    keras.Model
        Multi-input model for drug response prediction.
    """
    def __init__(self, expr_dim=None, drug_dim=None, expr_hlayers_sizes='[10]',
                        predictor_hlayers_sizes='[10]', initializer='he_normal', hidden_activation='relu',
                        drug_seq_length=0, drug_n_embedding=75,
                        drug_char_dict=DEFAULT_CHAR_DICT_STR,
                        drug_kernel_sizes='[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]',
                        drug_num_filters='[100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]',
                        drug_dropout=0.25, drug_l1=0, drug_l2=0, l1=0, l2=0,
                        input_dropout=0, hidden_dropout=0, optimizer='Adam', learn_rate=0.001):
        
        self.cell_line_feature_size = expr_dim
        self.drug_feature_size = drug_dim
        self.expr_hlayers_sizes = expr_hlayers_sizes
        self.predictor_hlayers_sizes = predictor_hlayers_sizes
        self.initializer = initializer
        self.hidden_activation = hidden_activation
        self.drug_seq_length = drug_seq_length
        self.drug_n_embedding = drug_n_embedding
        self.drug_char_dict = drug_char_dict
        self.drug_kernel_sizes = drug_kernel_sizes
        self.drug_num_filters = drug_num_filters
        self.drug_dropout = drug_dropout
        self.drug_l1 = drug_l1
        self.drug_l2 = drug_l2
        self.l1 = l1
        self.l2 = l2
        self.input_dropout = input_dropout
        self.hidden_dropout = hidden_dropout
        self.optimizer = optimizer
        self.learn_rate = learn_rate

        self.model = self.build_model()

    def build_model(self):
        # Define input layers
        cell_line_input = Input(shape=(self.cell_line_feature_size,), name='cell_line_input')
        drug_input = Input(shape=(self.drug_feature_size,), name='drug_input')

        # Define subnetworks for cell line and drug
        cell_line_subnet = dense_submodel(cell_line_input, hlayers_sizes=self.expr_hlayers_sizes, l1_regularization=self.l1, l2_regularization=self.l2,
                                          hidden_activation=self.hidden_activation, input_dropout=self.input_dropout, hidden_dropout=self.hidden_dropout)
        
        drug_subnet = drugcnn_submodel(seq_length=self.drug_seq_length, n_embedding=self.drug_n_embedding,
	                                 char_dict=self.drug_char_dict, kernel_sizes=self.drug_kernel_sizes,
	                                 num_filters=self.drug_num_filters, dropout=self.drug_dropout, l1=self.drug_l1, l2=self.drug_l2)
        
        drug = drug_subnet(drug_input)

        # Concatenate the outputs of the attended subnetworks
        concat = Concatenate(name = 'input_layer_concat')([cell_line_subnet, drug])

        # Final network for IC50 prediction
        final_network = dense_submodel(concat, hlayers_sizes=self.predictor_hlayers_sizes,
                                 l1_regularization=self.l1, l2_regularization=self.l2,
                                 hidden_activation=self.hidden_activation, input_dropout=0,
                                 hidden_dropout=self.hidden_dropout)
        
        final_network = Dense(1, activation='linear', name='ic50_prediction_dense_output')(final_network)


        # Create the model
        model = tf.keras.Model(inputs=[cell_line_input, drug_input], outputs=final_network)

        # Define optimizer
        opt_class = dict(getmembers(optimizers))[self.optimizer]
        if self.optimizer == 'SGD':
            opt = opt_class(lr=self.learn_rate, decay=1e-6, momentum=0.9, nesterov=True)
        else:
            opt = opt_class(lr=self.learn_rate)

        # Compile the model
        model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mae'])

        return model

    def summary(self):
        self.model.summary()

    def train(self, cell_line_train_data, drug_train_data, train_labels, epochs, batch_size):
        self.history = self.model.fit(
            {'cell_line_input': cell_line_train_data, 'drug_input': drug_train_data},
            {'ic50_prediction_dense_output': train_labels},
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2
        )

    def evaluate(self, cell_line_test_data, drug_test_data, test_labels):
        return self.model.evaluate(
            {'cell_line_input': cell_line_test_data, 'drug_input': drug_test_data},
            {'ic50_prediction_dense_output': test_labels}
        )

    def predict(self, new_cell_line_data, new_drug_data):
        return self.model.predict({'cell_line_input': new_cell_line_data, 'drug_input': new_drug_data})
    




class DrugGCNModel:
    """
    Multi-input model for drug response prediction.
    Uses a GCN subnetwork for drug SMILES and a dense subnetwork for cell line features.

    Parameters
    ----------
    expr_dim : int
        Dimension of the cell line feature space.
    drug_dim : int
        Dimension of the drug feature space.
    expr_hlayers_sizes : str
        String representation of a list of integers. Each integer represents the size of a hidden layer in the cell line subnetwork.
    predictor_hlayers_sizes : str
        String representation of a list of integers. Each integer represents the size of a hidden layer in the final network.
    initializer : str
        Initializer for the weights of the dense layers.
    hidden_activation : str
        Activation function for the hidden layers.
    drug_n_atom_features : int
        Number of atom features for the drug subnetwork.
    drug_gcn_layers : str
        String representation of a list of integers. Each integer represents the number of filters for a GCN layer.
    drug_residual_connection : bool
        Whether to use residual connections in the drug subnetwork.
    drug_dropout : float
        Dropout rate for the drug subnetwork.
    l1 : float
        L1 regularization factor for the final network.
    l2 : float
        L2 regularization factor for the final network.
    input_dropout : float
        Dropout rate for the input layer.
    hidden_dropout : float
        Dropout rate for the hidden layers.
    optimizer : str
        Optimizer for the model.
    learn_rate : float
        Learning rate for the optimizer.

    Returns
    -------
    keras.Model
        Multi-input model for drug response prediction.

    """
    def __init__(self, expr_dim=None, drug_dim=None, expr_hlayers_sizes='[10]', drug_n_atom_features=30,
                        drug_gcn_layers='[64, 64]', drug_residual_connection=True, drug_dropout=0.5,
                        predictor_hlayers_sizes='[10]', initializer='he_normal', hidden_activation='relu', l1=0, l2=0,
                        input_dropout=0, hidden_dropout=0, optimizer='Adam', learn_rate=0.001):
        
        self.cell_line_feature_size = expr_dim
        self.drug_feature_size = drug_dim
        self.expr_hlayers_sizes = expr_hlayers_sizes
        self.drug_n_atom_features = drug_n_atom_features
        self.drug_gcn_layers = drug_gcn_layers
        self.drug_residual_connection = drug_residual_connection
        self.drug_dropout = drug_dropout
        self.predictor_hlayers_sizes = predictor_hlayers_sizes
        self.initializer = initializer
        self.hidden_activation = hidden_activation
        self.l1 = l1
        self.l2 = l2
        self.input_dropout = input_dropout
        self.hidden_dropout = hidden_dropout
        self.optimizer = optimizer
        self.learn_rate = learn_rate

        self.model = self.build_model()

    def build_model(self):
        
        # define input layers
        cell_line_input = Input(shape=(self.cell_line_feature_size,), name='cell_line_input')
        drug_nodes_input = Input(shape=(None, self.drug_n_atom_features), name='drug_nodes_input')
        drug_edges_input = Input(shape=(None, None), name='drug_edges_input')

        # define subnetworks for cell line and drug
        cell_line_subnet = dense_submodel(cell_line_input, hlayers_sizes=self.expr_hlayers_sizes, l1_regularization=self.l1, l2_regularization=self.l2,
                                          hidden_activation=self.hidden_activation, input_dropout=self.input_dropout,
                                          hidden_dropout=self.hidden_dropout)
        
        drug_subnet = gcn_submodel(self.drug_n_atom_features, gcn_layers=self.drug_gcn_layers, residual=self.drug_residual_connection,
                                   dropout_rate=self.drug_dropout, l2=self.l2)


        drug = drug_subnet([drug_nodes_input, drug_edges_input])

        # Concatenate the outputs of the subnetworks
        concat = Concatenate(name = 'input_layer_concat')([cell_line_subnet, drug])

        # Final network for IC50 prediction
        final_network = dense_submodel(concat,hlayers_sizes=self.predictor_hlayers_sizes,
                                       l1_regularization=self.l1, l2_regularization=self.l2,
                                       hidden_activation=self.hidden_activation, input_dropout=0,
                                       hidden_dropout=self.hidden_dropout)
        
        final_network = Dense(1, activation='linear', name='ic50_prediction_dense_output')(final_network)


        # Create the model
        model = tf.keras.Model(inputs=[cell_line_input, drug_nodes_input, drug_edges_input], outputs=final_network)

        # Define optimizer
        opt_class = dict(getmembers(optimizers))[self.optimizer]
        if self.optimizer == 'SGD':
            opt = opt_class(lr=self.learn_rate, decay=1e-6, momentum=0.9, nesterov=True)
        else:
            opt = opt_class(lr=self.learn_rate)

        # Compile the model
        model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mae'])

        return model


    def summary(self):
        self.model.summary()

    def train(self, cell_line_train_data, drug_nodes_train_data, drug_edges_train_data ,train_labels, epochs, batch_size):
        self.history = self.model.fit(
            {'cell_line_input': cell_line_train_data, 'drug_nodes_input': drug_nodes_train_data, 'drug_edges_input': drug_edges_train_data},
            {'ic50_prediction_dense_output': train_labels},
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2
        )

    def evaluate(self, cell_line_test_data, drug_nodes_test_data, drug_edges_test_data, test_labels):
        return self.model.evaluate(
            {'cell_line_input': cell_line_test_data, 'drug_nodes_input': drug_nodes_test_data, 'drug_edges_input': drug_edges_test_data},
            {'ic50_prediction_dense_output': test_labels}
        )

    def predict(self, new_cell_line_data, new_drug_nodes_data, new_drug_edges_data):
        return self.model.predict({'cell_line_input': new_cell_line_data, 'drug_nodes_input': new_drug_nodes_data, 'drug_edges_input': new_drug_edges_data})
    




class DrugGATModel:

    def __init__(self, expr_dim=None, drug_dim=None, expr_hlayers_sizes='[10]', drug_n_atom_features=30,
                 drug_gat_layers='[64, 64]', drug_num_attention_heads=8, drug_concat_heads=True,
                 drug_residual_connection=True, drug_dropout=0.5, predictor_hlayers_sizes='[10]',
                 initializer='he_normal', hidden_activation='relu', l1=0, l2=0, input_dropout=0,
                 hidden_dropout=0, optimizer='Adam', learn_rate=0.001):
        
        self.cell_line_feature_size = expr_dim
        self.drug_feature_size = drug_dim
        self.expr_hlayers_sizes = expr_hlayers_sizes
        self.drug_n_atom_features = drug_n_atom_features
        self.drug_gat_layers = drug_gat_layers
        self.drug_num_attention_heads = drug_num_attention_heads
        self.drug_concat_heads = drug_concat_heads
        self.drug_residual_connection = drug_residual_connection
        self.drug_dropout = drug_dropout    
        self.predictor_hlayers_sizes = predictor_hlayers_sizes
        self.initializer = initializer
        self.hidden_activation = hidden_activation
        self.l1 = l1
        self.l2 = l2
        self.input_dropout = input_dropout
        self.hidden_dropout = hidden_dropout
        self.optimizer = optimizer
        self.learn_rate = learn_rate

        self.model = self.build_model()

    def build_model(self):
        
        # define input layers
        cell_line_input = Input(shape=(self.cell_line_feature_size,), name='cell_line_input')
        drug_nodes_input = Input(shape=(None, self.drug_n_atom_features), name='drug_nodes_input')
        drug_edges_input = Input(shape=(None, None), name='drug_edges_input')
        
        # define subnetworks for cell line and drug
        cell_line_subnet = dense_submodel(cell_line_input, hlayers_sizes=self.expr_hlayers_sizes, l1_regularization=self.l1, l2_regularization=self.l2,
                                          hidden_activation = self.hidden_activation, input_dropout = self.input_dropout,
                                          hidden_dropout = self.hidden_dropout)
        
        drug_subnet = gat_submodel(n_atom_features=self.drug_n_atom_features, gat_layers=self.drug_gat_layers,
                                   n_attention_heads=self.drug_num_attention_heads, concat_heads=self.drug_concat_heads,
                                   residual=self.drug_residual_connection, dropout_rate=self.drug_dropout)
        
        drug = drug_subnet([drug_nodes_input, drug_edges_input])

        # Concatenate the outputs of the subnetworks
        concat = Concatenate(name = 'input_layer_concat')([cell_line_subnet, drug])

        # Final network for IC50 prediction
        final_network = dense_submodel(concat,hlayers_sizes=self.predictor_hlayers_sizes,
                                       l1_regularization=self.l1, l2_regularization=self.l2,
                                       hidden_activation=self.hidden_activation, input_dropout=0,
                                       hidden_dropout=self.hidden_dropout)
        
        final_network = Dense(1, activation='linear', name='ic50_prediction_dense_output')(final_network)


        # Create the model
        model = tf.keras.Model(inputs=[cell_line_input, drug_nodes_input, drug_edges_input], outputs=final_network)

        # Define optimizer
        opt_class = dict(getmembers(optimizers))[self.optimizer]
        if self.optimizer == 'SGD':
            opt = opt_class(lr=self.learn_rate, decay=1e-6, momentum=0.9, nesterov=True)
        else:
            opt = opt_class(lr=self.learn_rate)

        # Compile the model
        model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mae'])

        return model


    def summary(self):
        self.model.summary()

    def train(self, cell_line_train_data, drug_nodes_train_data, drug_edges_train_data ,train_labels, epochs, batch_size):
        self.history = self.model.fit(
            {'cell_line_input': cell_line_train_data, 'drug_nodes_input': drug_nodes_train_data, 'drug_edges_input': drug_edges_train_data},
            {'ic50_prediction_dense_output': train_labels},
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2
        )

    def evaluate(self, cell_line_test_data, drug_nodes_test_data, drug_edges_test_data, test_labels):
        return self.model.evaluate(
            {'cell_line_input': cell_line_test_data, 'drug_nodes_input': drug_nodes_test_data, 'drug_edges_input': drug_edges_test_data},
            {'ic50_prediction_dense_output': test_labels}
        )

    def predict(self, new_cell_line_data, new_drug_nodes_data, new_drug_edges_data):
        return self.model.predict({'cell_line_input': new_cell_line_data, 'drug_nodes_input': new_drug_nodes_data, 'drug_edges_input': new_drug_edges_data})



