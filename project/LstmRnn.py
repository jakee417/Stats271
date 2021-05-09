import tensorflow as tf
from WindowGenerator import WindowGenerator

bitcoin = 'data/bitcoin_query.csv'
OUT_STEPS = 24
multi_window = WindowGenerator(bitcoin,
                               input_width=72,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS,
                               label_columns=['num_transactions'])
num_features = multi_window.num_features


class LstmRnn(tf.keras.Model):

    def __init__(self, units=32, out_steps=24):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(num_features)

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
            x, state = self.lstm_cell(x, states=state,
                                      training=training)
            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output
            predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions

    @staticmethod
    def compile_and_fit(model, window, patience=2, max_epochs=20):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=patience,
                                                          mode='min')

        model.compile(loss=tf.losses.MeanSquaredError(),
                      optimizer=tf.optimizers.Adam(),
                      metrics=[tf.metrics.MeanAbsoluteError()])

        history = model.fit(window.train, epochs=max_epochs,
                            validation_data=window.val,
                            callbacks=[early_stopping])
        return history


feedback_model = LstmRnn(units=32, out_steps=OUT_STEPS)
print('Output shape (batch, time, features): ', feedback_model(multi_window.example[0]).shape)

history = feedback_model.compile_and_fit(feedback_model, multi_window)

multi_val_performance = dict()
multi_performance = dict()

multi_val_performance['AR LSTM'] = feedback_model.evaluate(multi_window.val)
multi_performance['AR LSTM'] = feedback_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(feedback_model)
