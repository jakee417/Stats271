from WindowGenerator import WindowGenerator
import tensorflow as tf
from LstmRnn import LstmRnn
import pytest
import json

checkpoint_path = 'checkpoints/cp.ckpt'
save_path = 'saved/LstmRnn'
bitcoin = 'data/bitcoin_query.csv'
save_img = 'figures/test_lstm_rnn.jpg'
OUT_STEPS = 24


@pytest.mark.skip(reason='too long')
@pytest.mark.parametrize("fname", [bitcoin])
def test_train(fname):
    multi_window = WindowGenerator(fname,
                                   input_width=72,
                                   label_width=OUT_STEPS,
                                   shift=OUT_STEPS,
                                   label_columns=['num_transactions'])

    num_features = multi_window.num_features

    feedback_model = LstmRnn(num_features, units=32, out_steps=OUT_STEPS)
    print('Output shape (batch, time, features): ', feedback_model(multi_window.example[0]).shape)

    history = feedback_model.compile_and_fit(model=feedback_model,
                                             window=multi_window,
                                             checkpoint_path=checkpoint_path,
                                             save_path=save_path)

    performance = dict()
    performance['train'] = feedback_model.evaluate(multi_window.train)
    performance['val'] = feedback_model.evaluate(multi_window.val)
    performance['test'] = feedback_model.evaluate(multi_window.test, verbose=0)
    print(performance)
    out_file = open("metrics/test.json", "w")
    json.dump(performance, out_file, indent=6)
    out_file.close()
    multi_window.plot(feedback_model, save_path=save_img)

@pytest.mark.parametrize("fname", [bitcoin])
@pytest.mark.parametrize("model_path", ['saved/LstmRnn'])
def test_load(model_path, fname):
    multi_window = WindowGenerator(fname,
                                   input_width=72,
                                   label_width=OUT_STEPS,
                                   shift=OUT_STEPS,
                                   label_columns=['num_transactions'])

    new_model = tf.keras.models.load_model(model_path)

    # Check its architecture
    print(new_model.summary())

    performance = dict()
    performance['train'] = new_model.evaluate(multi_window.train)
    performance['val'] = new_model.evaluate(multi_window.val)
    performance['test'] = new_model.evaluate(multi_window.test, verbose=0)
    print(performance)
    out_file = open("metrics/test.json", "w")
    json.dump(performance, out_file, indent=6)
    out_file.close()