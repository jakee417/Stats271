from WindowGenerator import WindowGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
from LstmRnn import LstmRnn
import pytest
import json
import time

bitcoin = 'data/bitcoin_query.csv'

# normal, negative_binomial, poisson, poisson_approximation
#@pytest.mark.skip(reason='skipping for now')
@pytest.mark.parametrize("fname", [bitcoin])
@pytest.mark.parametrize("distribution", [None])
@pytest.mark.parametrize("hidden_units", [100])
@pytest.mark.parametrize("resample", ['6H'])
@pytest.mark.parametrize("input_width", [90])
@pytest.mark.parametrize("out_steps", [30])
def test_train_distribution(fname,
                            distribution,
                            hidden_units,
                            resample,
                            input_width,
                            out_steps):
    tic = time.time()
    checkpoint_path = f'checkpoints/{distribution}.ckpt'
    save_path = f'saved/LstmRnn{distribution}'
    save_img = f'figures/{distribution}_test_lstm_rnn.jpg'
    loss_img = f'figures/{distribution}_loss.jpg'
    multi_window = WindowGenerator(fname,
                                   input_width=input_width,
                                   label_width=out_steps,
                                   shift=out_steps,
                                   label_columns=['num_transactions'],
                                   resample_frequency=resample,
                                   standardize=True)

    num_features = multi_window.num_features
    # multi_window.plot_splits(feedback_model)
    print(multi_window)

    feedback_model = LstmRnn(num_features,
                             units=hidden_units,
                             out_steps=out_steps,
                             distribution=distribution)

    history = feedback_model.compile_and_fit(model=feedback_model,
                                             window=multi_window,
                                             checkpoint_path=checkpoint_path,
                                             save_path=save_path,
                                             max_epochs=50,
                                             patience=5)

    # plot history of loss and val_loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.xlabel('Epochs')
    plt.savefig(loss_img)
    plt.show()

    # record evaluation metrics
    performance = dict()
    performance['time'] = time.time() - tic
    performance['input_width'] = input_width
    performance['out_steps'] = out_steps
    performance['resample'] = resample
    performance['hidden_units'] = hidden_units
    performance['train_size'] = len(multi_window.train)
    performance['val_size'] = len(multi_window.val)
    performance['test_size'] = len(multi_window.test)
    performance['distribution'] = distribution
    performance['num_features'] = multi_window.num_features
    performance['train'] = feedback_model.evaluate(multi_window.train)
    performance['val'] = feedback_model.evaluate(multi_window.val)
    performance['test'] = feedback_model.evaluate(multi_window.test, verbose=0)
    print(performance)
    out_file = open(f'metrics/{distribution}.json', "w")
    json.dump(performance, out_file, indent=6)
    out_file.close()

    samples = None
    if distribution:
        samples = 500

    multi_window.plot(feedback_model,
                      save_path=save_img,
                      max_subplots=6,
                      samples=samples,
                      mode='train')

    multi_window.plot(feedback_model,
                      save_path=save_img,
                      max_subplots=6,
                      samples=samples,
                      mode='test')



@pytest.mark.skip(reason='skipping for now')
@pytest.mark.parametrize("fname", [bitcoin])
@pytest.mark.parametrize("model_path", ['saved/LstmRnn'])
def test_load_save(model_path, fname):
    out_steps = 6
    multi_window = WindowGenerator(fname,
                                   input_width=72,
                                   label_width=out_steps,
                                   shift=out_steps,
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