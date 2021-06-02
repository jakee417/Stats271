from WindowGenerator import WindowGenerator
import matplotlib.pyplot as plt
from LstmRnn import LstmRnn
from datetime import datetime
import json
import time
import pprint

pp = pprint.PrettyPrinter(indent=4)


def run(params):
    now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    tic = time.time()
    distribution = params['distribution']
    checkpoint_path = f'checkpoints/{distribution}.ckpt'
    save_path = f'saved/LstmRnn{distribution}'
    train_save_img = f'figures/{now}_{distribution}_train_lstm_rnn.jpg'
    val_save_img = f'figures/{now}_{distribution}_val_lstm_rnn.jpg'
    test_save_img = f'figures/{now}_{distribution}_test_lstm_rnn.jpg'
    global_train_img = f'figures/{now}_train_{distribution}_global_lstm_rnn.jpg'
    global_val_img = f'figures/{now}_val_{distribution}_global_lstm_rnn.jpg'
    global_test_img = f'figures/{now}_test_{distribution}_global_lstm_rnn.jpg'

    loss_img = f'figures/{now}_{distribution}_loss.jpg'
    post_check_img = f'figures/{now}_{distribution}_post_check.jpg'
    train_correlation_img = f'figures/{now}_train_{distribution}_correlation'
    test_correlation_img = f'figures/{now}_test_{distribution}_correlation'

    multi_window = WindowGenerator(params['fname'],
                                   input_width=params['input_width'],
                                   label_width=params['label_width'],
                                   shift=params['shift'],
                                   label_columns=['num_transactions'],
                                   resample_frequency=params['resample'],
                                   standardize=True,
                                   batch_size=params['batch_size'])

    # multi_window.plot_splits(feedback_model)
    print(multi_window)
    time_index = multi_window.column_indices['timestamp']

    params.update({'time_index': time_index})
    feedback_model = LstmRnn(params)
    feedback_model(multi_window.train_example[0])
    print(feedback_model.summary())
    history = feedback_model.compile_and_fit(model=feedback_model,
                                             window=multi_window,
                                             checkpoint_path=checkpoint_path,
                                             save_path=save_path,
                                             max_epochs=params['max_epochs'],
                                             patience=params['patience'])
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
        samples = 200

    # plot forecast plots
    multi_window.plot(feedback_model,
                      save_path=train_save_img,
                      max_subplots=3,
                      samples=samples,
                      mode='train')
    #
    # multi_window.plot(feedback_model,
    #                   save_path=cal_save_img,
    #                   max_subplots=3,
    #                   samples=samples,
    #                   mode='val')
    #
    # multi_window.plot(feedback_model,
    #                   save_path=test_save_img,
    #                   max_subplots=3,
    #                   samples=samples,
    #                   mode='test')

    forecasts = []
    train_forecast = multi_window.forecast(
        model=feedback_model,
        dataset_name='train',
        samples=samples
    )
    forecasts.append(train_forecast)

    val_forecast = multi_window.forecast(
        model=feedback_model,
        dataset_name='val',
        samples=samples
    )
    forecasts.append(val_forecast)

    test_forecast = multi_window.forecast(
        model=feedback_model,
        dataset_name='test',
        samples=samples
    )
    forecasts.append(test_forecast)

    post_checks = multi_window.plot_posterior_predictive_check(forecasts,
                                                               post_check_img)

    # multi_window.plot_correlations(train_forecast, train_correlation_img)
    # multi_window.plot_correlations(test_forecast, test_correlation_img)


    multi_window.plot_global_forecast(
        train_forecast,
        save_path=global_train_img
    )

    multi_window.plot_global_forecast(
        val_forecast,
        save_path=global_val_img
    )

    multi_window.plot_global_forecast(
        test_forecast,
        save_path=global_test_img
    )

    # record evaluation metrics
    performance = dict()
    performance['time'] = toc - tic
    performance['now'] = now
    performance.update(params)
    performance['train_size'] = len(multi_window.train)
    performance['val_size'] = len(multi_window.val)
    performance['test_size'] = len(multi_window.test)
    performance['total_data'] = len(multi_window.df)
    performance['train'] = feedback_model.evaluate(multi_window.train)
    performance['val'] = feedback_model.evaluate(multi_window.val)
    performance['test'] = feedback_model.evaluate(multi_window.test)
    performance.update(post_checks)
    pp.pprint(performance)

    # cache performance
    out_file = open(f'metrics/{now}_{distribution}.json', "w")
    json.dump(performance, out_file, indent=6)
    out_file.close()


if __name__ == '__main__':
    bitcoin = 'data/bitcoin_query.csv'
    etherium = 'data/ethereum.csv'
    params = dict(
        fname=bitcoin,
        distribution='normal',
        lstm_units=32,
        t2v_units=8,
        dense_cells=None,
        resample=None,
        input_width=96,
        label_width=24,
        shift=24,
        max_epochs=40,
        patience=2,
        latent_dim=2,
        beta=1,
        min_df=2.0,
        number_states=10,
        batch_size=256,
        regularization=0.01
    )

    run(params)
