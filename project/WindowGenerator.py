#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


class WindowGenerator():
    def __init__(self,
                 fname,
                 input_width=24,
                 label_width=1,
                 shift=1,
                 label_columns=None,
                 resample_frequency='60T',
                 standardize=True):
        # Member attributes
        self.label_columns = label_columns
        self.covariate_columns = None
        self.resample = resample_frequency
        self.standardize = standardize

        # Read in data.
        self.fname = fname
        self._read_data()
        self._split()
        if self.standardize:
            self._standardize()

        # Work out the label column indices.
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(self.train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def _read_data(self):
        self.df = pd.read_csv(self.fname)
        time_cats = ['hour', 'month', 'day', 'year']
        self.df.index = pd.to_datetime(self.df[time_cats])
        self.df = self.df.drop(labels=time_cats, axis=1)
        self.df = self.df[self.label_columns]
        # TODO: add log variable option
        if self.resample:
            self.df = self.df.resample(self.resample).apply(self._aggregate)
        # self.df['timestamp'] = self.df.index.map(datetime.datetime.timestamp)

    @staticmethod
    def _aggregate(x):
        return np.sum(x) if len(x) > 0 else 0

    @staticmethod
    def rolling_window(a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def _split(self):
        self.column_indices = {name: i for i, name in enumerate(self.df.columns)}
        self.n = len(self.df)
        self.train_df = self.df[0:int(self.n * 0.7)]
        self.val_df = self.df[int(self.n * 0.7):int(self.n * 0.9)]
        self.test_df = self.df[int(self.n * 0.9):]
        self.num_features = self.df.shape[1]

    def _standardize(self):
        self.train_mean = self.train_df.mean()
        self.train_std = self.train_df.std()
        self.train_df = (self.train_df - self.train_mean) / self.train_std
        self.val_df = (self.val_df - self.train_mean) / self.train_std
        self.test_df = (self.test_df - self.train_mean) / self.train_std

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1
            )
        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels

    def make_dataset(self, data, shuffle=True):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=shuffle,
            # setting this to self.label_width simplifies forecast
            batch_size=32, )
        ds = ds.map(self.split_window)
        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def test_example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_test_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.test))
            # And cache it for next time
            self._test_example = result
        return result

    @property
    def train_example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_train_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._train_example = result
        return result

    def rescale(self, x):
        return (x.numpy() * self.train_std[0]) + self.train_mean[0]

    def plot(self, model=None,
             plot_col='num_transactions',
             max_subplots=3,
             save_path=None,
             mode='train',
             samples=None):
        if mode == 'test':
            inputs, labels = self.test_example
        else:
            inputs, labels = self.train_example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        ax1 = None
        for n in range(max_n):
            ax = plt.subplot(max_n, 1, n + 1,
                             sharex=ax1 if ax1 else None,
                             sharey=ax1 if ax1 else None)
            true_inputs = self.rescale(inputs[n, :, plot_col_index])
            plt.plot(self.input_indices, true_inputs,
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            true_labels = self.rescale(labels[n, :, label_col_index])
            plt.scatter(self.label_indices, true_labels,
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                if samples is not None:
                    predictions = model(inputs).sample(1)[0, ...]
                else:
                    predictions = model(inputs)
                predictions = self.rescale(predictions)
                plt.scatter(self.label_indices,
                            predictions[n, :, label_col_index],
                            marker='X',
                            edgecolors='k',
                            label='Predictions',
                            c='#ff7f0e', s=64)
            if samples:
                res_samples = model(inputs).sample(samples)
                res_samples = res_samples[:, n, :, label_col_index]
                res_samples = self.rescale(res_samples)
                upper = np.percentile(res_samples, 95, axis=0)
                lower = np.percentile(res_samples, 5, axis=0)
                mean = np.mean(res_samples, axis=0)
                plt.fill_between(x=self.label_indices,
                                 y1=lower,
                                 y2=upper,
                                 alpha=0.5,
                                 label='90% CR')
                plt.plot(self.label_indices,
                         mean,
                         'r--',
                         label='Predicted Mean',
                         alpha=0.5)

            if n == 0:
                ax1 = ax
                plt.legend()
                plt.title(f'Forecast on {mode}')
                ax1.set_ylabel(f'{plot_col} [normed]')
                ax1.set_xlabel(f'Time [{self.resample}]')

        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_global_forecast(self,
                             model,
                             dataset_name='test',
                             samples=500,
                             save_path=None,
                             breaklines=False,
                             plot_col='num_transactions'):

        if dataset_name == 'train':
            dataset = self.train_df
        elif dataset_name == 'val':
            dataset = self.val_df
        else:
            dataset = self.test_df

        original = (dataset[self.label_columns[0]]
                    * self.train_std[0]
                    + self.train_mean[0])

        if model:
            # Get indices of our forecast windows that are similar to how we created our data
            ind = self.rolling_window(np.arange(len(dataset.index)), self.total_window_size)

            # This is the overlapping code
            ind_overlapping = ind[:, self.input_width:].flatten()
            ind_overlapping_unique = np.unique(ind_overlapping)
            ind_overlapping_index = dataset.index[ind_overlapping_unique]

            # This is the non-overlapping code
            '''
            ind = ind[::self.label_width, self.input_width:]  # slice only non-overlapping forecast windows
            starts = ind[:, 0]
            starts = dataset.index[starts]
            ind = ind.flatten()
            # Convert indices to dates
            ind = dataset.index[ind]
            '''

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)

            # Make forecasts for windows from unshuffled dataset
            unshuffled = self.make_dataset(data=dataset, shuffle=False)


            res_samples_l = []

            # Loop through datasets making forecasts
            # and sample from each overlapping window's forecast
            for element in unshuffled.enumerate(start=0):
                data = element[1][0]
                res_samples = model(data).sample(samples)
                # res_samples => (samples, batch, time, features)
                res_samples = res_samples[..., label_col_index]
                res_samples = self.rescale(res_samples)
                res_samples_l.append(res_samples)

            # Concatenate and seperate (samples) from (time, features)
            all_samples = np.concatenate(res_samples_l, axis=1)
            all_samples = all_samples.reshape(samples, -1)

            upper_l = []
            lower_l = []
            mean_l = []
            upper_50_l = []
            lower_50_l = []

            # TODO: Vectorize me
            # TODO: Vectorize this over many percentiles
            # TODO: Add weighted forecast by sample variance?
            for index in ind_overlapping_unique:
                sub_samples = all_samples[:, ind_overlapping == index]
                upper_l.append(np.percentile(sub_samples, 95))
                lower_l.append(np.percentile(sub_samples, 5))
                upper_50_l.append(np.percentile(sub_samples, 75))
                lower_50_l.append(np.percentile(sub_samples, 25))
                mean_l.append(np.mean(sub_samples))

            mean = np.array(mean_l)
            upper = np.array(upper_l)
            lower = np.array(lower_l)
            upper_50 = np.array(upper_50_l)
            lower_50 = np.array(lower_50_l)

            '''
            # Take non-overlapping slices of self.label_width and then flatten result
            mean = np.concatenate(mean_l)[::self.label_width, :].flatten()
            upper = np.concatenate(upper_l)[::self.label_width, :].flatten()
            lower = np.concatenate(lower_l)[::self.label_width, :].flatten()
            '''

            # Start the plotting!
            if breaklines:
                # plot breaklines and the warmup section
                '''
                plt.vlines(starts,
                           ymin=original.min(),
                           ymax=original.max(),
                           linestyles='--',
                           color='black',
                           label='Forecast Start Lines',
                           alpha=0.1)
                '''

                plt.fill_between(x=dataset.index[dataset.index <= ind_overlapping_index[0]],
                                 y1=original.min(),
                                 y2=original.max(),
                                 color='yellow',
                                 alpha=0.2,
                                 label=f'Warmup Period')

            # Plot resulting credible intervals
            plt.fill_between(x=ind_overlapping_index,
                             y1=lower,
                             y2=upper,
                             alpha=0.5,
                             color='cornflowerblue',
                             label=f'90% CR')

            plt.fill_between(x=ind_overlapping_index,
                             y1=lower_50,
                             y2=upper_50,
                             alpha=0.5,
                             color='royalblue',
                             label=f'50% CR')

            # Compute and plot anomalies and not anomalies
            anomaly_index = np.logical_or(original[ind_overlapping_index] > upper,
                                          original[ind_overlapping_index] < lower)
            anomalies = original[ind_overlapping_index][anomaly_index]
            not_anomalies = original[ind_overlapping_index][-anomaly_index]

            anomaly_index_50 = np.logical_or(original[ind_overlapping_index] > upper_50,
                                             original[ind_overlapping_index] < lower_50)
            anomalies_50 = original[ind_overlapping_index][anomaly_index_50]
            not_anomalies_50 = original[ind_overlapping_index][-anomaly_index_50]

            # Plot original points, good points, and anomalies
            plt.scatter(x=original.index,
                        y=original,
                        edgecolors='k',
                        label='All Labels',
                        c='black',
                        s=4)

            plt.plot(not_anomalies,
                     linestyle='--',
                     linewidth=0.3,
                     color='green')

            plt.scatter(x=not_anomalies.index,
                        y=not_anomalies,
                        edgecolors='k',
                        label='Labels within 90%',
                        c='green',
                        s=5)

            plt.scatter(x=anomalies.index,
                        y=anomalies,
                        edgecolors='k',
                        label='Anomalies outside 90%',
                        c='red',
                        s=10)

            # Plot anomaly ratio
            anomaly_fraction = np.sum(not_anomalies) / (np.sum(anomalies) + np.sum(not_anomalies))
            print(f'Not Anomaly (90%) to Total ratio for {dataset_name}: {anomaly_fraction}')
            anomaly_fraction_50 = np.sum(not_anomalies_50) / (np.sum(anomalies_50) + np.sum(not_anomalies_50))
            print(f'Not Anomaly (50%) to Total ratio for {dataset_name}: {anomaly_fraction_50}')

            # Finally plot the mean
            plt.plot(ind_overlapping_index,
                     mean,
                     '--',
                     color='black',
                     alpha=0.8,
                     label=f'Predicted Mean')

            # Clip plot in case of wild means
            plt.ylim(original.min(), original.max())

        plt.legend()
        if save_path:
            plt.savefig(save_path)
        plt.show()

        return anomaly_fraction, anomaly_fraction_50

    # TODO: Add Posterior Predictive check plot
    def plot_posterior_predictive_check(self):
        pass

    def plot_splits(self, save_path=None):
        plt.plot(self.train_df[self.label_columns[0]], label='train')
        plt.plot(self.val_df[self.label_columns[0]], label='val')
        plt.plot(self.test_df[self.label_columns[0]], label='test')
        plt.legend()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Shift (Input) -> (Label): {self.shift}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}',
            f'Total time series: {self.n}'])
