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
    def __init__(self, params):
        super().__init__()
        # Member attributes
        self.out_steps = params['label_width']
        self.lstm_units = params['lstm_units']
        self.t2v_units = params['t2v_units']
        self.distribution = params['distribution']
        self.dense_cells = params['dense_cells']
        # These will be parameters for our TFP layer
        self.params = None
        self.latent_dim = params['latent_dim']
        # http://www.matthey.me/pdf/betavae_iclr_2017.pdf
        self.beta = params['beta']
        self.time_index = params['time_index']
        self.l2_reg = tf.keras.regularizers.l2(params['regularization']) if params['regularization'] else None
        self.initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.01)

        # Metrics
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.regularization_tracker = tf.keras.metrics.Mean(name='regularization_loss')

        # TF Layers for RNN
        self.lstm_cell_warmup = tf.keras.layers.LSTMCell(self.lstm_units,
                                                         kernel_regularizer=self.l2_reg,
                                                         recurrent_regularizer=self.l2_reg,
                                                         name='lstm_warmup')
        self.lstm_cell = tf.keras.layers.LSTMCell(self.lstm_units,
                                                  kernel_regularizer=self.l2_reg,
                                                  recurrent_regularizer=self.l2_reg,
                                                  name='lstm_forecast')
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell_warmup, return_state=True, name='rnn')

        # T2V and Latent Dimension
        if self.t2v_units:
            self.T2V = layers.T2V(self.t2v_units)
        if self.latent_dim:
            self.z_mean = None
            self.z_log_var = None
            self.sampling = layers.Sampling()
            self.dense_mean_latent = tf.keras.layers.Dense(self.latent_dim, name='dense_z_mean')
            self.dense_var_latent = tf.keras.layers.Dense(self.latent_dim, name='dense_z_log_var')

        # TFP Layers
        if self.distribution == 'normal':
            self.params = 2
            self.dist_lambda = layers.normal
            # self.dist_lambda = distributions.variational_normal
        elif self.distribution == 'poisson_approx':
            self.params = 1
            self.dist_lambda = layers.poisson_approximation
        elif self.distribution == 'student_t':
            self.params = 3
            self.min_df = params['min_df'] if params['min_df'] else 1.0
            self.dist_lambda = layers.StudentT(self.min_df, name='student_t')
        elif self.distribution == 'laplace':
            self.params = 2
            self.dist_lambda = layers.laplace
        elif self.distribution == 'mix':
            # [(Normal, 2), (Student t, 3), (Laplace, 2), (logits, 3)]
            self.params = 11
            self.min_df = params['min_df'] if params['min_df'] else 1.0
            self.dist_lambda = layers.LocationScaleMixture(self.min_df, name='mix')
        elif self.distribution == 'hmm':
            self.number_states = params['number_states']
            self.params = (
                    2 * self.number_states
                    + self.number_states
                    + self.number_states ** 2
            )
            self.dist_lambda = layers.HiddenMarkovModel(number_states=self.number_states,
                                                        forecast_length=self.out_steps,
                                                        name='hmm')
        else:
            raise Exception('Distribution not correct.')

        # Dense TF Layers
        self.dense_extras = tf.keras.layers.Dense(self.lstm_units, activation='relu',
                                                  kernel_regularizer=self.l2_reg,
                                                  kernel_initializer=self.initializer,
                                                  name='dense_extra')

        self.dense = tf.keras.layers.Dense(self.params,
                                           kernel_regularizer=self.l2_reg,
                                           kernel_initializer=self.initializer,
                                           name='dense_main')
        if self.latent_dim:
            self.dense_unbottleneck = tf.keras.layers.Dense(self.params, activation='relu',
                                                            kernel_regularizer=self.l2_reg,
                                                            kernel_initializer=self.initializer,
                                                            name='dense_unbottleneck')

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def negative_log_likelihood(self, y_pred, y_true):
        if self.distribution == 'hmm':
            return -y_pred.log_prob(y_true)
        else:
            return -tf.reduce_mean(
                tf.reduce_sum(
                    y_pred.log_prob(y_true), axis=1
                )
            )

    @staticmethod
    def kl_loss(z_log_var, z_mean, beta):
        kl_loss = -0.5 * (1 + z_log_var
                          - tf.square(z_mean)
                          - tf.exp(z_log_var))
        return beta * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

    def warmup(self, inputs):
        """Encodes a time sequence as an Lstm Rnn"""
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        if self.t2v_units:
            # parse time feature for T2V
            t = self.T2V(inputs[..., self.time_index:self.time_index + 1])
            x = tf.keras.layers.Concatenate()([inputs, t])
            # x.shape => (batch, time, features + self.t2v_units)
            x, *state = self.lstm_rnn(x)
        else:
            x, *state = self.lstm_rnn(inputs)
        # predictions.shape => (batch, features)
        if self.dense_cells:
            for _ in range(self.dense_cells):
                x = self.dense_extras(x)
        prediction = self.dense(x)
        return prediction, state

    def call(self, inputs, training=None, encoding=False):
        # encoder
        prediction, state = self.warmup(inputs)

        if encoding and not self.latent_dim:
            return None

        if self.latent_dim:
            self.z_mean = self.dense_mean_latent(prediction)
            self.z_log_var = self.dense_var_latent(prediction)
            prediction = self.sampling([self.z_mean, self.z_log_var])
            # shortstop encoding and return
            if encoding:
                return prediction
            # transform dimensions back to size of self.params
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
            if self.dense_cells:
                for _ in range(self.dense_cells):
                    x = self.dense_extras(x)
            prediction = self.dense(x)

            # Add the prediction to the output
            predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        # convert rates to distribution layer
        if self.distribution:
            # predictions.shape => (batch, time, params) or (batch, time)
            predictions = predictions[:, 0, :] if self.distribution == 'hmm' else predictions
            predictions = self.dist_lambda(predictions)
        return predictions

    def train_step(self, data):
        # Unpack the data
        x, y = data

        # ensure y.shape => (batch, time) for hmm
        y = y[..., 0] if self.distribution == 'hmm' else y

        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self(x, training=True)
            regularization_loss = tf.add_n(self.losses) if self.l2_reg else 0
            # Compute the loss value
            reconstruction = self.negative_log_likelihood(y_pred, y)
            # https://keras.io/examples/generative/vae/
            # https://arxiv.org/pdf/1312.6114.pdf
            kl_loss = self.kl_loss(self.z_log_var, self.z_mean, self.beta) if self.latent_dim else 0
            total_loss = reconstruction + regularization_loss + kl_loss
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction)
        self.regularization_tracker.update_state(regularization_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        # Return a dict mapping metric names to current value
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "regularization_loss": self.regularization_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
        }

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # ensure y.shape => (batch, time) for hmm
        y = y[..., 0] if self.distribution == 'hmm' else y

        # Compute predictions
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        regularization_loss = tf.add_n(self.losses) if self.l2_reg else 0
        reconstruction = self.negative_log_likelihood(y_pred, y)
        # https://keras.io/examples/generative/vae/
        kl_loss = self.kl_loss(self.z_log_var, self.z_mean, self.beta) if self.latent_dim else 0
        total_loss = reconstruction + regularization_loss + kl_loss
        # Update metrics (includes the metric that tracks the loss)
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction)
        self.regularization_tracker.update_state(regularization_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        # Return a dict mapping metric names to current value
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "regularization_loss": self.regularization_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
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
            patience=patience * 2,
            mode='min',
            restore_best_weights=True,
            verbose=1
        ), tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, patience=patience, verbose=1,
            mode='auto', min_delta=0.1, cooldown=0, min_lr=0
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
