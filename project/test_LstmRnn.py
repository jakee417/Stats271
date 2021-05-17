from WindowGenerator import WindowGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
from LstmRnn import LstmRnn
import pytest
import json
import time

# TODO: Add other coins
bitcoin = 'data/bitcoin_query.csv'


# normal, negative_binomial, poisson, poisson_approximation
# @pytest.mark.skip(reason='skipping for now')
@pytest.mark.parametrize("fname", [bitcoin])
@pytest.mark.parametrize("distribution", ['locationscalemix'])
@pytest.mark.parametrize("hidden_units", [100])
@pytest.mark.parametrize("t2v_units", [128])
@pytest.mark.parametrize("resample", ['6H'])
@pytest.mark.parametrize("input_width", [90])
@pytest.mark.parametrize("out_steps", [30])
@pytest.mark.parametrize("max_epochs", [20])
@pytest.mark.parametrize("patience", [3])
def test_train_distribution(fname,
                            distribution,
                            hidden_units,
                            t2v_units,
                            resample,
                            input_width,
                            out_steps,
                            max_epochs,
                            patience):
    tic = time.time()
    checkpoint_path = f'checkpoints/{distribution}.ckpt'
    save_path = f'saved/LstmRnn{distribution}'
    train_save_img = f'figures/{distribution}_train_lstm_rnn.jpg'
    test_save_img = f'figures/{distribution}_test_lstm_rnn.jpg'
    global_save_img = f'figures/{distribution}_global_lstm_rnn.jpg'
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
                             lstm_units=hidden_units,
                             t2v_units=t2v_units,
                             out_steps=out_steps,
                             distribution=distribution)

    history = feedback_model.compile_and_fit(model=feedback_model,
                                             window=multi_window,
                                             checkpoint_path=checkpoint_path,
                                             save_path=save_path,
                                             max_epochs=max_epochs,
                                             patience=patience)

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
    performance['t2v_units'] = t2v_units
    performance['train_size'] = len(multi_window.train)
    performance['val_size'] = len(multi_window.val)
    performance['test_size'] = len(multi_window.test)
    performance['total_data'] = len(multi_window.df)
    performance['distribution'] = distribution
    performance['num_features'] = multi_window.num_features
    performance['max_epochs'] = max_epochs
    performance['patience'] = patience
    performance['train'] = feedback_model.evaluate(multi_window.train)
    performance['val'] = feedback_model.evaluate(multi_window.val)
    performance['test'] = feedback_model.evaluate(multi_window.test, verbose=0)
    print(performance)

    # cache performance
    out_file = open(f'metrics/{distribution}.json', "w")
    json.dump(performance, out_file, indent=6)
    out_file.close()

    # init samples
    samples = None
    if distribution:
        samples = 500

    # plot forecast plots
    multi_window.plot(feedback_model,
                      save_path=train_save_img,
                      max_subplots=3,
                      samples=samples,
                      mode='train')

    multi_window.plot(feedback_model,
                      save_path=test_save_img,
                      max_subplots=3,
                      samples=samples,
                      mode='test')

    multi_window.plot_global_forecast(model=feedback_model,
                                      save_path=global_save_img,
                                      breaklines=True,
                                      dataset_name='test')

    multi_window.plot_global_forecast(model=feedback_model,
                                      save_path=global_save_img,
                                      breaklines=True,
                                      dataset_name='train')


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
