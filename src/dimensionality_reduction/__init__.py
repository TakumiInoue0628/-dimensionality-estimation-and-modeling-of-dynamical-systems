from os.path import dirname, abspath
import sys
import tensorflow as tf
from keras.api._v2 import keras
from sklearn.model_selection import train_test_split
### Move to parent directory
parent_dir = dirname(dirname(abspath(__file__)))
sys.path.append(parent_dir)
from util import set_seed
from dimensionality_reduction.function import three_layer_autoencoder, loss_history

class Autoencoder():

    def __init__(self, input_size, latent_size, activation_enc=None, activation_dec="sigmoid", seed=0):
        set_seed(seed)
        self.model = three_layer_autoencoder(input_size, latent_size, activation_enc, activation_dec)

    def fit(self, data, 
            parameters={"learning_rate": 5e-4,
                        "loss_function": "mse",
                        "train_size_rate": 0.8,
                        "batch_size": 128,
                        "epoch": 10000,
                        "early_stopping": True,
                        "early_stopping_round": 10,
                        "output_log": True}):
        self.train_parameters = parameters
        ### Set optimizers
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=parameters["learning_rate"]),
                           loss=parameters["loss_function"])
        ### Split data into train & valid sets
        train, valid = train_test_split(data, train_size=parameters["train_size_rate"], shuffle=False)
        train_dataset = tf.data.Dataset.from_tensor_slices((train, train)).batch(parameters["batch_size"])
        valid_dataset = tf.data.Dataset.from_tensor_slices((valid, valid)).batch(parameters["batch_size"])
        del train, valid
        ### Set callback
        callback_early_stop = keras.callbacks.EarlyStopping(monitor="val_loss",
                                                            min_delta=1e-5,
                                                            patience=parameters["early_stopping_round"],
                                                            verbose=int(parameters["output_log"]),
                                                            mode="min",
                                                            restore_best_weights=True)
        callback_loss_history = loss_history()
        if parameters["early_stopping"]: callback = [callback_early_stop, callback_loss_history]
        else: callback = [callback_loss_history]
        ### Train model
        self.model.fit(x=train_dataset,
                       validation_data=valid_dataset,
                       epochs=parameters["epoch"],
                       verbose=int(parameters["output_log"]),
                       callbacks=callback)
        del train_dataset, valid_dataset
        self.loss_train = callback_loss_history.loss
        self.loss_valid = callback_loss_history.val_loss

    def transform(self, data):
        return self.model.encoder(data).numpy()
    
    def inverse_transform(self, data):
        return self.model.decoder(data).numpy()