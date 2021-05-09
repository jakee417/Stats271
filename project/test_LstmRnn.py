from WindowGenerator import WindowGenerator
from LstmRnn import LstmRnn


checkpoint_path = 'checkpoints/cp.ckpt'
save_path = 'saved/LstmRnn'
bitcoin = 'data/bitcoin_query.csv'
OUT_STEPS = 24
multi_window = WindowGenerator(bitcoin,
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

multi_val_performance = dict()
multi_performance = dict()

multi_val_performance['AR LSTM'] = feedback_model.evaluate(multi_window.val)
multi_performance['AR LSTM'] = feedback_model.evaluate(multi_window.test, verbose=0)
print(multi_val_performance)
print(multi_performance)
multi_window.plot(feedback_model)
