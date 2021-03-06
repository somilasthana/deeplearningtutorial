import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs('acc') > 0.99:
            println("reaching accuracy 0.99 so cancelling training")
            self.model.stop_training = True
