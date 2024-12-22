from keras.api._v2 import keras

class three_layer_autoencoder(keras.Model):

    def __init__(self, input_size, latent_size, activation_enc=None, activation_dec="sigmoid"):
        super(three_layer_autoencoder, self).__init__()
        ### Define encoder's layer
        self.encoder = keras.Sequential([keras.layers.Dense(latent_size, activation=activation_enc)])
        ### Define decoder's layer
        self.decoder = keras.Sequential([keras.layers.Dense(input_size, activation=activation_dec)])
        
    def call(self, x):
        return self.decoder(self.encoder(x))
    
class loss_history(keras.callbacks.Callback):
    
    def __init__(self):
        super().__init__()
        self.loss = []
        self.val_loss = []

    def on_epoch_end(self, epoch, logs=None):
        self.loss.append(logs['loss'])
        self.val_loss.append(logs['val_loss'])