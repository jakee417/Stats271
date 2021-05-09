from WindowGenerator import WindowGenerator

fname = 'bitcoin_query.csv'
w1 = WindowGenerator(fname,
                     shift=5,
                     label_width=5,
                     label_columns=['num_transactions'])

for example_inputs, example_labels in w1.train.take(1):
  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  print(f'Labels shape (batch, time, features): {example_labels.shape}')

w1.plot()


OUT_STEPS = 24
multi_window = WindowGenerator(fname,
                               input_width=24,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS)

multi_window.plot()
multi_window