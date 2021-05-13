import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


class LstmRnn(tf.keras.Model):

    def __init__(self, num_features, units=32, out_steps=24, distribution=None):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.num_features = num_features
        self.distribution = distribution
        # TODO: Add Time2Vec https://towardsdatascience.com/time2vec-for-time-series-features-encoding-a03a4f3f937e
        self.lstm_cell_warmup = tf.keras.layers.LSTMCell(units)
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell_warmup, return_state=True)
        if distribution == 'poisson':
            self.dense = tf.keras.layers.Dense(self.num_features)
            self.dist_lambda = tfp.layers.DistributionLambda(
                lambda t: tfd.Poisson(
                    #rate=1e-3 + tf.math.softplus(0.05 * t)
                    rate=1e-3 + tf.exp(t)
                )
            )
        elif distribution == 'negative_binomial':
            self.dense = tf.keras.layers.Dense(self.num_features * 2)
            self.dist_lambda = tfp.layers.DistributionLambda(
                lambda t: tfd.NegativeBinomial(
                    # FixMe: results in NaNs
                    total_count=tf.math.round(1e-3 + tf.math.softplus(0.05 * t[..., 1:])),
                    probs=tf.math.minimum(1e-3 + tf.math.softplus(0.05 * t[..., 1:]), tf.constant([1.]))
                )
            )
        elif distribution == 'normal':
            self.dense = tf.keras.layers.Dense(self.num_features * 2)
            self.dist_lambda = tfp.layers.DistributionLambda(
                lambda t: tfd.Normal(
                    loc=t[..., :1],
                    scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:])
                )
            )
        elif distribution == 'poisson_approximation':
            # https://en.wikipedia.org/wiki/Poisson_distribution#Related_distributions
            self.dense = tf.keras.layers.Dense(self.num_features)
            self.dist_lambda = tfp.layers.DistributionLambda(
                lambda t: tfd.Normal(
                    loc=1e-3 + tf.math.softplus(0.05 * t),
                    scale=tf.math.sqrt(1e-3 + tf.math.softplus(0.05 * t))
                )
            )
        else:
            self.dense = tf.keras.layers.Dense(self.num_features)

    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *state = self.lstm_rnn(inputs)

        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state

    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the lstm state
        prediction, state = self.warmup(inputs)

        # Insert the first prediction
        predictions.append(prediction)

        # Run the rest of the prediction steps
        for n in range(1, self.out_steps):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            x, state = self.lstm_cell(x,
                                      states=state,
                                      training=training)
            # Convert the lstm output to a prediction.
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

    @property
    def model(self):
        return self.keras_model

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
