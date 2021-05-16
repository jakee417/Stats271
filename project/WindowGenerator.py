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
        # TODO: Incorporate time
        self.df = self.df.drop(labels=time_cats, axis=1)
        self.df = self.df[self.label_columns]
        if self.resample:
            self.df = self.df.resample(self.resample).apply(self._aggregate)
        #self.df['timestamp'] = self.df.index.map(datetime.datetime.timestamp)

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

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32, )
        ds = ds.map(self.split_window)
        return ds

    def make_dataset_unshuffled(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=32, )
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
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f'{plot_col} [normed]')
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
            # TODO: Remove this function and include ancestral sampling
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
                plt.legend()

        plt.xlabel(f'Time [{self.resample}]')
        plt.title(f'Forecast on {mode}')
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_global_forecast(self, model=None, samples=500, save_path=None, plot_col='num_transactions'):
        plt.plot(self.test_df[self.label_columns[0]] * self.train_std[0] + self.train_mean[0], label='test')

        if model:
            unshuffled = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=np.array(self.test_df, dtype=np.float32),
                targets=None,
                sequence_length=self.total_window_size,
                sequence_stride=1,
                shuffle=False,
                batch_size=32, )

            ind = self.rolling_window(np.arange(len(self.test_df.index)), self.label_width) \
                      .flatten()[:-self.label_width * self.input_width]
            ind = np.arange(len(self.test_df.index))[self.total_window_size-1:]
            ind = self.test_df.index[ind]

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)

            upper_l = []
            lower_l = []
            mean_l = []

            for element in unshuffled.enumerate(start=0):
                data = element[1]
                res_samples = model(data).sample(samples)
                # res_samples => (samples, batch, time, features)
                res_samples = res_samples[:, :, 0, label_col_index]
                res_samples = self.rescale(res_samples)
                upper_l.append(np.percentile(res_samples, 95, axis=0).flatten())
                lower_l.append(np.percentile(res_samples, 5, axis=0).flatten())
                mean_l.append(np.mean(res_samples, axis=0).flatten())

            mean = np.concatenate(mean_l)
            upper = np.concatenate(upper_l)
            lower = np.concatenate(lower_l)

            plt.fill_between(x=ind,
                             y1=lower,
                             y2=upper,
                             alpha=0.5,
                             label=f'90% CR')
            plt.plot(ind,
                     mean,
                     'r--',
                     label=f'Predicted Mean',
                     alpha=0.5)

        plt.legend()
        if save_path:
            plt.savefig(save_path)
        plt.show()


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
