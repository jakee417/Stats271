import layers
from layers import LocationScaleMixture
from layers import HiddenMarkovModel
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

def encoder(num_features,
            params,
            shape=(32, 30, 1),
            t2v_units=128,
            lstm_units=100,
            dense_cells=2):
    # inputs.shape => (batch, time, features)
    # x.shape => (batch, lstm_units)
    lstm_cell_warmup = tf.keras.layers.LSTMCell(lstm_units)
    # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
    encoder_inputs = tf.keras.Input(shape=shape)
    if t2v_units:
        x = layers.T2V(t2v_units)(encoder_inputs)
        x, *state = tf.keras.layers.RNN(lstm_cell_warmup, return_state=True)(x)
    else:
        x, *state = tf.keras.layers.RNN(lstm_cell_warmup, return_state=True)(encoder_inputs)
    # predictions.shape => (batch, features)
    for _ in range(dense_cells):
        x = tf.keras.layers.Dense(lstm_units, activation='relu')(x)
    prediction = tf.keras.layers.Dense(num_features * params)(x)
    # TODO: Add variational layer
    return tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")


def decoder(num_features,
            params,
            prediction,
            state,
            out_steps,
            distribution,
            dist_lambda,
            lstm_units=100,
            training=None,
            dense_cells=2):
    # Use a TensorArray to capture dynamically unrolled outputs.
    predictions = []

    # Insert the first prediction
    predictions.append(prediction)

    # Run the rest of the prediction steps
    for n in range(1, out_steps):
        # Use the last prediction as input.
        x = prediction
        # Execute one lstm step.
        x, state = tf.keras.layers.LSTMCell(lstm_units)(
            x,
            states=state,
            training=training
        )
        # Convert the lstm output to a prediction.
        for _ in range(dense_cells):
            x = tf.keras.layers.Dense(lstm_units, activation='relu')(x)
        prediction = tf.keras.layers.Dense(num_features * params)(x)

        # Add the prediction to the output
        predictions.append(prediction)

    # predictions.shape => (time, batch, features)
    predictions = tf.stack(predictions)
    # predictions.shape => (batch, time, features)
    predictions = tf.transpose(predictions, [1, 0, 2])
    # convert rates to distribution layer
    if distribution:
        # predictions.shape => (batch, time, params)
        if distribution == 'hiddenmarkovmodel':
            predictions = predictions[:, 0, :]
        predictions = dist_lambda(predictions)
        # predictions.shape => (batch, time)
    return predictions


class LstmRnn(tf.keras.Model):
    def __init__(self, num_features,
                 window,
                 max_epochs,
                 lstm_units=32,
                 t2v_units=None,
                 out_steps=24,
                 dense_cells=1,
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

        self.max_epochs = max_epochs
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.optimizer = tf.optimizers.Adam()
        self.window = window

        # One LSTMCell for warmup, one for forecasting
        self.lstm_cell_warmup = tf.keras.layers.LSTMCell(self.lstm_units)
        self.lstm_cell = tf.keras.layers.LSTMCell(self.lstm_units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell_warmup, return_state=True)
        self.dense_1 = tf.keras.layers.Dense(self.lstm_units, activation='relu')
        if self.t2v_units:
            self.T2V = layers.T2V(self.t2v_units)
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
        else:
            self.dense = tf.keras.layers.Dense(self.num_features)
        self.dense = tf.keras.layers.Dense(self.num_features * self.params)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker
        ]

    def loss(self, y_true, y_hat):
        return -y_hat.log_prob(y_true)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            encoded, state = encoder(data,
                                     self.num_features,
                                     self.params,
                                     shape=(32, 30, 1),
                                     t2v_units=self.t2v_units,
                                     lstm_units=self.lstm_units,
                                     dense_cells=self.dense_cells)
            encoded(data)

            decoded = decoder(self.num_features,
                              self.params,
                              encoded,
                              state,
                              self.out_steps,
                              self.distribution,
                              self.dist_lambda,
                              self.lstm_units,
                              self.training,
                              self.dense_cells)



        grads = tape.gradient(self.loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.total_loss_tracker.update_state(total_loss)
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

        history = model.fit(window.train,
                            epochs=max_epochs,
                            validation_data=window.val,
                            callbacks=cp,
                            verbose=1)

        return history
