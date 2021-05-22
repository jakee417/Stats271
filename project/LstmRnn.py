import layers
from layers import LocationScaleMixture
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np


class LstmRnn(tf.keras.Model):

    def __init__(self, num_features,
                 lstm_units=32,
                 t2v_units=None,
                 out_steps=24,
                 distribution=None):
        super().__init__()
        # Member attributes
        self.out_steps = out_steps
        self.lstm_units = lstm_units
        self.t2v_units = t2v_units
        self.num_features = num_features
        self.distribution = distribution
        # One LSTMCell for warmup, one for forecasting
        self.lstm_cell_warmup = tf.keras.layers.LSTMCell(self.lstm_units)
        self.lstm_cell = tf.keras.layers.LSTMCell(self.lstm_units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell_warmup, return_state=True)
        self.dense_1 = tf.keras.layers.Dense(self.lstm_units, activation='relu')
        if self.t2v_units:
            self.T2V = layers.T2V(self.t2v_units)
        if distribution == 'normal':
            params = 2
            self.dist_lambda = layers.normal
            #self.dist_lambda = distributions.variational_normal
        elif distribution == 'locationscalemix':
            # [(Normal, 2), (Student t, 3), (laplace, 2), (logits, 3)]
            params = 13
            self.dist_lambda = tfp.layers.DistributionLambda(
                lambda t: LocationScaleMixture()(t)
            )
        elif distribution == 'hiddenmarkovmodel':
            number_states = 30
            params = (
                2 * number_states
                + number_states
                + number_states**2
            )
            self.dense = tf.keras
        else:
            self.dense = tf.keras.layers.Dense(self.num_features)
        self.dense = tf.keras.layer.Dense(self.num_features * params)

    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        if self.t2v_units:
            x = self.T2V(inputs)
            x, *state = self.lstm_rnn(x)
        else:
            x, *state = self.lstm_rnn(inputs)
        # predictions.shape => (batch, features)
        for _ in range(2):
            x = self.dense_1(x)
        prediction = self.dense(x)
        return prediction, state

    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the lstm state
        prediction, state = self.warmup(inputs)

        # Insert the first prediction
        predictions.append(prediction)

        # TODO: Add variational layer

        # Run the rest of the prediction steps
        for n in range(1, self.out_steps):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            x, state = self.lstm_cell(x,
                                      states=state,
                                      training=training)
            # Convert the lstm output to a prediction.
            for _ in range(2):
                x = self.dense_1(x)
            prediction = self.dense(x)

            # Add the prediction to the output
            predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        # convert rates to distribution layer
        if self.distribution:
            predictions = self.dist_lambda(predictions)
        return predictions

    def compile_and_fit(self,
                        model,
                        window,
                        checkpoint_path=None,
                        save_path=None,
                        patience=2,
                        max_epochs=20):
        cp = []

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=patience,
                                                          mode='min')
        cp.append(early_stopping)

        if checkpoint_path:
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                             save_weights_only=True,
                                                             verbose=1)
            cp.append(cp_callback)

        if self.distribution:
            loss = lambda y_true, y_hat: -y_hat.log_prob(y_true)
        else:
            loss = tf.losses.MeanSquaredError()

        model.compile(loss=loss,
                      optimizer=tf.optimizers.Adam(),
                      metrics=[tf.metrics.MeanAbsoluteError()])

        history = model.fit(window.train,
                            epochs=max_epochs,
                            validation_data=window.val,
                            callbacks=cp,
                            verbose=1)

        if save_path and not self.distribution:
            model.save(save_path)

        return history
