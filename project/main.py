from WindowGenerator import WindowGenerator
import matplotlib.pyplot as plt
from LstmRnn import LstmRnn
import json
import time
import pprint
pp = pprint.PrettyPrinter(indent=4)


def run(fname,
        distribution,
        hidden_units,
        t2v_units,
        dense_cells,
        latent_dim,
        beta,
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
    post_check_img = f'figures/{distribution}_post_check.jpg'
    latent_img = f'figures/{distribution}_correlation'
    multi_window = WindowGenerator(fname,
                                   input_width=input_width,
                                   label_width=out_steps,
                                   shift=out_steps,
                                   label_columns=['num_transactions'],
                                   resample_frequency=resample,
                                   standardize=True)

    # multi_window.plot_splits(feedback_model)
    print(multi_window)
    time_index = multi_window.column_indices['timestamp']

    feedback_model = LstmRnn(lstm_units=hidden_units,
                             t2v_units=t2v_units,
                             out_steps=out_steps,
                             dense_cells=dense_cells,
                             distribution=distribution,
                             latent_dim=latent_dim,
                             beta=beta,
                             time_index=time_index)

    history = feedback_model.compile_and_fit(model=feedback_model,
                                             window=multi_window,
                                             checkpoint_path=checkpoint_path,
                                             save_path=save_path,
                                             max_epochs=max_epochs,
                                             patience=patience)
    toc = time.time()

    # plot history of loss and val_loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.xlabel('Epochs')
    plt.savefig(loss_img)
    plt.show()

    # init samples
    samples = None
    if distribution:
        samples = 500

    # plot forecast plots
    '''
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
    '''

    train_forecast = multi_window.forecast(
        model=feedback_model,
        dataset_name='train',
        samples=200
    )

    val_forecast = multi_window.forecast(
        model=feedback_model,
        dataset_name='val',
        samples=200
    )

    test_forecast = multi_window.forecast(
        model=feedback_model,
        dataset_name='test',
        samples=200
    )



    forecasts = [train_forecast, test_forecast, val_forecast]
    post_checks = multi_window.plot_posterior_predictive_check(forecasts,
                                                               post_check_img)

    '''
    multi_window.plot_correlations(train_forecast, latent_img)
    multi_window.plot_correlations(test_forecast, latent_img)
    '''

    multi_window.plot_global_forecast(
        train_forecast,
        save_path=global_save_img
    )

    multi_window.plot_global_forecast(
        val_forecast,
        save_path=global_save_img
    )

    multi_window.plot_global_forecast(
        test_forecast,
        save_path=global_save_img
    )

    # record evaluation metrics
    performance = dict()
    performance['time'] = toc - tic
    performance['input_width'] = input_width
    performance['out_steps'] = out_steps
    performance['resample'] = resample
    performance['hidden_units'] = hidden_units
    performance['t2v_units'] = t2v_units
    performance['dense_cells'] = dense_cells
    performance['latent_dim'] = latent_dim
    performance['beta'] = beta
    performance['train_size'] = len(multi_window.train)
    performance['val_size'] = len(multi_window.val)
    performance['test_size'] = len(multi_window.test)
    performance['total_data'] = len(multi_window.df)
    performance['distribution'] = distribution
    performance['max_epochs'] = max_epochs
    performance['patience'] = patience
    performance['train'] = feedback_model.evaluate(multi_window.train)
    performance['val'] = feedback_model.evaluate(multi_window.val)
    performance['test'] = feedback_model.evaluate(multi_window.test, verbose=0)
    performance.update(post_checks)
    pp.pprint(performance)

    # cache performance
    out_file = open(f'metrics/{distribution}.json', "w")
    json.dump(performance, out_file, indent=6)
    out_file.close()


if __name__ == '__main__':
    bitcoin = 'data/bitcoin_query.csv'
    fname = bitcoin
    distribution = 'normal'
    hidden_units = 8
    t2v_units = 8
    dense_cells = 1
    resample = None
    input_width = 96
    out_steps = 24
    max_epochs = 40
    patience = 3
    latent_dim = 2
    beta = 2

    run(fname,
        distribution,
        hidden_units,
        t2v_units,
        dense_cells,
        latent_dim,
        beta,
        resample,
        input_width,
        out_steps,
        max_epochs,
        patience)
