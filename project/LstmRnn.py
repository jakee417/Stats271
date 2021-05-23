import layers
from layers import LocationScaleMixture
from layers import HiddenMarkovModel
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

"""
List of resources used:
https://www.tensorflow.org/tutorials/structured_data/time_series
https://keras.io/examples/generative/vae/
https://github.com/francois-meyer/time2vec
"""

class LstmRnn(tf.keras.Model):
    """Implements a Probabilistic Lstm Rnn with embeddings and variational layers"""
    def __init__(self, num_features,
                 lstm_units=32,
                 t2v_units=None,
                 out_steps=24,
                 dense_cells=1,
                 latent_dim=2,
                 distribution=None):
        super().__init__()
        # Member attributes
        self.out_steps = out_steps
        self.lstm_units = lstm_units
        self.t2v_units = t2v_units
        self.num_features = num_features
        self.distribution = distribution
        self.dense_cells = dense_cells
        self.params = None
        self.latent_dim = latent_dim

        # Metrics
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

        # TF Layers
        self.lstm_cell_warmup = tf.keras.layers.LSTMCell(self.lstm_units)
        self.lstm_cell = tf.keras.layers.LSTMCell(self.lstm_units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell_warmup, return_state=True)
        self.dense_1 = tf.keras.layers.Dense(self.lstm_units, activation='relu')
        if self.t2v_units:
            self.T2V = layers.T2V(self.t2v_units)
        if self.latent_dim:
            self.z_mean = None
            self.z_log_var = None
            self.sampling = layers.Sampling()
            self.dense_latent = tf.keras.layers.Dense(self.latent_dim)

        # TFP Layers
        if distribution == 'normal':
            self.params = 2
            self.dist_lambda = layers.normal
            # self.dist_lambda = distributions.variational_normal
        elif distribution == 'locationscalemix':
            # [(Normal, 2), (Student t, 3), (laplace, 2), (logits, 3)]
            self.params = 13
            self.dist_lambda = tfp.layers.DistributionLambda(
                lambda t: LocationScaleMixture()(t)
            )
        elif distribution == 'hiddenmarkovmodel':
            # FixMe: output shape not working
            number_states = 15
            self.params = (
                    2 * number_states
                    + number_states
                    + number_states ** 2
            )
            self.dist_lambda = tfp.layers.DistributionLambda(
                lambda t: HiddenMarkovModel(number_states=number_states,
                                            forecast_length=out_steps)(t)
            )
        self.dense = tf.keras.layers.Dense(self.num_features * self.params)
        if latent_dim:
            self.dense_unbottleneck = tf.keras.layers.Dense(self.num_features * self.params)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    @staticmethod
    def negative_log_likelihood(y_pred, y_true):
        # TODO: Put prior on reconstruction loss
        # (tf.reduce_sum(rate_prior.log_prob(tf.math.exp(trainable_log_rates))) + hmm.log_prob(obs_data))
        return -y_pred.log_prob(y_true)

    def warmup(self, inputs):
        """Encodes a time sequence as an Lstm Rnn"""
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        if self.t2v_units:
            x = self.T2V(inputs)
            x, *state = self.lstm_rnn(x)
        else:
            x, *state = self.lstm_rnn(inputs)
        # predictions.shape => (batch, features)
        for _ in range(self.dense_cells):
            x = self.dense_1(x)
        prediction = self.dense(x)
        return prediction, state

    def call(self, inputs, training=None, encoding=False):
        # encoder
        prediction, state = self.warmup(inputs)
        if self.latent_dim:
            self.z_mean = self.dense_latent(prediction)
            self.z_log_var = self.dense_latent(prediction)
            prediction = self.sampling([self.z_mean, self.z_log_var])
            if encoding:
                return prediction
        # transform dimensions back to size of output
        prediction = self.dense_unbottleneck(prediction)

        # decoder
        predictions = [prediction]
        # Run the rest of the prediction steps
        for n in range(1, self.out_steps):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            x, state = self.lstm_cell(
                x,
                states=state,
                training=training
            )
            # Convert the lstm output to a prediction.
            for _ in range(self.dense_cells):
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
            # predictions.shape => (batch, time, params)
            if self.distribution == 'hiddenmarkovmodel':
                # predictions.shape => (batch, time)
                predictions = predictions[:, 0, :]
            predictions = self.dist_lambda(predictions)
        return predictions

    def train_step(self, data):
        # Unpack the data
        x, y = data
        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self(x, training=True)
            # Compute the loss value
            reconstruction = self.negative_log_likelihood(y_pred, y)
            # https://keras.io/examples/generative/vae/
            kl_loss = -0.5 * (1 + self.z_log_var
                              - tf.square(self.z_mean)
                              - tf.exp(self.z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction + kl_loss

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction)
        self.kl_loss_tracker.update_state(kl_loss)
        # Return a dict mapping metric names to current value
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        reconstruction = self.negative_log_likelihood(y_pred, y)
        # https://keras.io/examples/generative/vae/
        kl_loss = -0.5 * (1 + self.z_log_var
                          - tf.square(self.z_mean)
                          - tf.exp(self.z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction + kl_loss
        self.compiled_metrics.update_state(y, y_pred)
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction)
        self.kl_loss_tracker.update_state(kl_loss)
        # Return a dict mapping metric names to current value
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def compile_and_fit(self,
                        model,
                        window,
                        checkpoint_path=None,
                        save_path=None,
                        patience=2,
                        max_epochs=20):
        cp = [tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                mode='min'
            )]
        if checkpoint_path:
            cp.append(tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                save_weights_only=True,
                verbose=1
            ))
        model.compile(optimizer=tf.optimizers.Adam())
        history = model.fit(window.train,
                            epochs=max_epochs,
                            validation_data=window.val,
                            callbacks=cp,
                            verbose=1)
        if save_path and not self.distribution:
            model.save(save_path)
        return history
