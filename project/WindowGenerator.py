#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import itertools


class WindowGenerator():
    """Creates a Window object consisting of time series data"""
    def __init__(self, fname, input_width=24, label_width=1,
                 shift=1, label_columns=None, resample_frequency='60T',
                 standardize=True, batch_size=256):
        # Member attributes
        self.label_columns = label_columns
        self.covariate_columns = None
        self.resample = resample_frequency
        self.standardize = standardize
        self.batch_size = batch_size

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
        """Read in raw data and preprocess"""
        time_cats = ['hour', 'month', 'day', 'year']
        self.df = pd.read_csv(self.fname)
        self.df.index = pd.to_datetime(self.df[time_cats])
        self.df = self.df.drop(labels=time_cats, axis=1)
        self.df = self.df[self.label_columns]
        if self.resample:
            self.df = self.df.resample(self.resample).apply(self._aggregate)
        self.df['timestamp'] = self.df.index.map(datetime.datetime.timestamp)

    @staticmethod
    def _aggregate(x):
        """Helper function to aggregate data"""
        return np.sum(x) if len(x) > 0 else 0

    @staticmethod
    def rolling_window(a, window):
        """Helper function to find indices of a rolling window"""
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def _split(self):
        """Helper function to split train, val, and test sets"""
        self.column_indices = {name: i for i, name in enumerate(self.df.columns)}
        self.n = len(self.df)
        self.train_df = self.df[0:int(self.n * 0.7)]
        self.val_df = self.df[int(self.n * 0.7):int(self.n * 0.9)]
        self.test_df = self.df[int(self.n * 0.9):]
        self.num_features = self.df.shape[1]

    def _standardize(self):
        """Helper function to standardize data"""
        self.train_mean = self.train_df.mean()
        self.train_std = self.train_df.std()
        self.train_df = (self.train_df - self.train_mean) / self.train_std
        self.val_df = (self.val_df - self.train_mean) / self.train_std
        self.test_df = (self.test_df - self.train_mean) / self.train_std

    def split_window(self, features):
        """Helper function to split a window into inputs and labels"""
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
        """Create a time series dataset from array"""
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=shuffle,
            batch_size=self.batch_size, )
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

    @property
    def fig_size(self):
        return (12, 8)

    def rescale(self, x):
        return (x.numpy() * self.train_std[0]) + self.train_mean[0]

    def plot(self,
             model=None,
             plot_col='num_transactions',
             max_subplots=3,
             save_path=None,
             mode='train',
             samples=None):
        """Plot individual batches of data with or without predictions"""
        if mode == 'test':
            inputs, labels = self.test_example
        else:
            inputs, labels = self.train_example
        plt.figure(figsize=self.fig_size)
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

    def forecast(self, model, dataset_name='test',
                 samples=500, plot_col='num_transactions'):
        """Forecast future values using a trained model over an entire dataset"""
        if dataset_name == 'train':
            dataset = self.train_df
        elif dataset_name == 'val':
            dataset = self.val_df
        elif dataset_name == 'test':
            dataset = self.test_df
        else:
            raise Exception('dataset_name is either train, val, or test.')

        original = (dataset[self.label_columns[0]]
                    * self.train_std[0]
                    + self.train_mean[0])

        # Get indices of our forecast windows that are similar to how we created our data
        ind = self.rolling_window(np.arange(len(dataset.index)), self.total_window_size)

        # slice just the forecasted indices
        ind_overlapping = ind[:, -self.label_width:].flatten()
        ind_overlapping_unique = np.unique(ind_overlapping)
        ind_overlapping_index = dataset.index[ind_overlapping_unique]

        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)

        # Make forecasts for windows from unshuffled dataset
        unshuffled = self.make_dataset(data=dataset, shuffle=False)

        print(f'Sampling forecast values: {dataset_name}')
        tic = time.time()
        res_samples_l = []
        zs = []

        # Loop through datasets making forecasts
        # and sample from each overlapping window's forecast
        for element in unshuffled.enumerate(start=0):
            data = element[1][0]
            batch_size = data.shape[0]

            # make forecasts
            res_samples = model(data).sample(samples)
            if res_samples.shape != (samples, batch_size, self.label_width, 1):
                res_samples = res_samples[..., None]
            assert res_samples.shape == (samples, batch_size, self.label_width, 1)

            # extract z's
            zs.append(model(data, encoding=True))

            # res_samples => (samples, batch, time, features)
            res_samples = res_samples[..., label_col_index]
            res_samples = self.rescale(res_samples)
            res_samples_l.append(res_samples)

        print(f'Sampling complete: {time.time() - tic}')

        # Concatenate and separate (samples) from (time, features)
        all_samples = np.concatenate(res_samples_l, axis=1)
        all_samples = all_samples.reshape(samples, -1)
        try:
            zs = np.concatenate(zs)
        except:
            zs = None

        upper_l = []
        lower_l = []
        mean_l = []

        # grid to compute model checking
        uppers = np.arange(55, 96, 1)
        lowers = np.arange(45, 4, -1)

        # sorted_indices = np.argsort(ind_overlapping)
        # sorted_ind_overlapping = np.sort(ind_overlapping)
        # sorted_all_samples = all_samples[:, sorted_indices]
        # changepoints = np.where(sorted_ind_overlapping[:-1] != sorted_ind_overlapping[1:])[0] + 1
        # split_list = np.array_split(sorted_all_samples, changepoints, axis=1)
        # split_list_filled = np.array(list(itertools.zip_longest(*split_list, fillvalue=np.nan))).T
        # mean = np.nanmean(split_list_filled, axis=1)
        # upper = np.nanpercentile(split_list_filled, uppers, axis=1)
        # lower = np.nanmean(split_list_filled, lowers, axis=1)

        # TODO: rewrite this
        for index in ind_overlapping_unique:
            sub_samples = all_samples[:, ind_overlapping == index]
            upper_l.append(np.percentile(sub_samples, uppers))
            lower_l.append(np.percentile(sub_samples, lowers))
            mean_l.append(np.mean(sub_samples))
        mean = np.array(mean_l)
        upper = np.array(upper_l)
        lower = np.array(lower_l)

        print(f'Assign samples time index: {time.time() - tic}')

        # extract 90 percentiles for plotting
        upper_90 = upper[:, -1]
        lower_90 = lower[:, -1]

        # Compute and plot anomalies and not anomalies
        anomaly_index = np.logical_or(original[ind_overlapping_index] > upper_90,
                                      original[ind_overlapping_index] < lower_90)
        anomalies = original[ind_overlapping_index][anomaly_index]
        not_anomalies = original[ind_overlapping_index][-anomaly_index]

        return {
            'dataset': dataset,
            'mean': mean,
            'upper': upper,
            'lower': lower,
            'upper_90': upper_90,
            'lower_90': lower_90,
            'original': original,
            'ind_overlapping_index': ind_overlapping_index,
            'anomalies': anomalies,
            'not_anomalies': not_anomalies,
            'dataset_name': dataset_name,
            'zs': zs
        }

    def plot_global_forecast(self, forecast,
                             save_path=None):
        """Plot a global forecast given forecasted values"""
        anomalies = forecast['anomalies']
        not_anomalies = forecast['not_anomalies']
        mean = forecast['mean']
        upper_90 = forecast['upper_90']
        lower_90 = forecast['lower_90']
        dataset = forecast['dataset']
        dataset_name = forecast['dataset_name']
        ind_overlapping_index = forecast['ind_overlapping_index']
        original = forecast['original']
        plt.figure(figsize=self.fig_size)
        plt.fill_between(x=dataset.index[dataset.index <= ind_overlapping_index[0]],
                         y1=original.min(),
                         y2=original.max(),
                         color='yellow',
                         alpha=0.2,
                         label=f'Warmup Period')

        # Plot resulting credible intervals
        plt.fill_between(x=ind_overlapping_index,
                         y1=lower_90,
                         y2=upper_90,
                         alpha=0.5,
                         color='cornflowerblue',
                         label=f'90% CR')

        # Plot original points, good points, and anomalies
        plt.scatter(x=original.index,
                    y=original,
                    label='All Labels',
                    c='black',
                    s=2)


        plt.plot(not_anomalies,
                 linestyle='-',
                 label='Non-Anomalies Inside 90%',
                 linewidth=0.1,
                 color='green')

        # plt.scatter(x=not_anomalies.index,
        #             y=not_anomalies,
        #             label='Labels within 90%',
        #             c='green',
        #             s=3)

        plt.scatter(x=anomalies.index,
                    y=anomalies,
                    label='Anomalies outside 90%',
                    c='red',
                    s=3)

        # Finally plot the mean
        plt.plot(ind_overlapping_index,
                 mean,
                 '--',
                 color='black',
                 alpha=0.9,
                 label=f'Predicted Mean')

        # Clip plot in case of wild means
        plt.ylim(original.min(), original.max())

        plt.legend()
        plt.title(f'Dataset: {dataset_name}')
        plt.ylabel('Number of Transactions')
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_posterior_predictive_check(self, forecasts,
                                        save_path):
        """Plot a posterior predictive check given forecasts"""
        uppers = np.arange(55, 96, 1)
        lowers = np.arange(45, 4, -1)
        ideal = ((100 - uppers) + lowers) / 100
        plt.figure(figsize=self.fig_size)
        plt.plot(uppers - lowers,
                 ideal / 2,
                 color='red',
                 linestyle='--',
                 label='Ideal Upper and Lower Tails')
        res = {}
        colors = {'train': 'blue', 'val': 'orange', 'test': 'green'}
        for i, forecast in enumerate(forecasts):
            name = forecast['dataset_name']
            data = np.array(forecast['original'][forecast['ind_overlapping_index']])
            data = data[..., None]
            upper = forecast['upper']
            lower = forecast['lower']

            # Compute T(Y, T') and Err
            outside_above = (data > upper).sum(axis=0) / len(data)
            outside_below = (data < lower).sum(axis=0) / len(data)

            # Compute T(Err)
            area_above = np.sum(np.abs(outside_above - ideal / 2))
            area_below = np.sum(np.abs(outside_below - ideal / 2))

            # Plot results
            plt.plot(uppers - lowers, outside_above, label=f'Upper Tails of {name}',
                     marker='^', lw=1, markersize=10, color=colors[name])
            plt.plot(uppers - lowers, outside_below, label=f'Lower Tails of {name}',
                     marker='v', lw=1, markersize=10, color=colors[name])
            res[name + '_area_above'] = area_above
            res[name + '_area_below'] = area_below
            res[name + '_total_area'] = round(area_above + area_below, 2)
        plt.title('Posterior Predictive Check')
        plt.ylabel('Observed Proportion Falling Outside Credible Interval')
        plt.xlabel('Credible Interval Level')
        plt.xticks((uppers - lowers)[::5])
        plt.yticks(np.arange(.05, .6, .05))
        plt.grid()
        plt.legend()
        if save_path:
            plt.savefig(save_path)
        plt.show()
        return res

    def plot_correlations(self, forecast,
                          save_path=None):
        zs = forecast['zs']
        # FixMe: bug
        if zs:
            n = len(zs)
            plt.figure(figsize=self.fig_size)
            fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True, sharey=True)
            ax1.acorr(zs[:, 0], usevlines=True, maxlags=None, normed=True, lw=1,
                      label='Autocorrelation of 1st Latent Dimension', alpha=.3)
            ax2.acorr(zs[:, 1], usevlines=True, maxlags=None, normed=True, lw=1,
                      label='Autocorrelation of 2nd Latent Dimension', alpha=.3, color='orange')
            ax1.axhline(y=2 / np.sqrt(n), color='red', linestyle='--')
            ax1.axhline(y=-2 / np.sqrt(n), color='red', linestyle='--')
            ax2.axhline(y=2 / np.sqrt(n), color='red', linestyle='--', label=r'$\pm 2/n^{1/2}$')
            ax2.axhline(y=-2 / np.sqrt(n), color='red', linestyle='--')
            plt.xlim(left=1)
            plt.ylim(-3 / np.sqrt(n), 3 / np.sqrt(n))
            ax1.set(title='Autocorrelation of Latent Dimensions', ylabel='Sample Correlation')
            ax2.set(ylabel='Sample Correlation')
            plt.xlabel('Lag')
            handles, labels = [(a + b) for a, b in
                               zip(ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
            fig.legend(handles, labels, loc='upper right')
            if save_path:
                plt.savefig(save_path + '_latent_dims.jpg')
            plt.show()

        original = forecast['original']
        mean = forecast['mean']
        n = len(mean)
        plt.figure(figsize=self.fig_size)
        fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True)
        ax1.acorr(mean, usevlines=True, maxlags=None, normed=True, lw=1,
                  label='Autocorrelation of Predictions', alpha=.3)
        ax2.acorr(original, usevlines=True, maxlags=None, normed=True, lw=1,
                  label='Autocorrelation of Observed', alpha=.3, color='orange')
        ax1.axhline(y=2 / np.sqrt(n), color='red', linestyle='--')
        ax1.axhline(y=-2 / np.sqrt(n), color='red', linestyle='--')
        ax2.axhline(y=2 / np.sqrt(n), color='red', linestyle='--', label=r'$\pm 2/n^{1/2}$')
        ax2.axhline(y=-2 / np.sqrt(n), color='red', linestyle='--')
        plt.xlim(left=1)
        ax1.set(title='Autocorrelation of Predicted Mean vs Observed Series', ylabel='Sample Correlation')
        ax2.set(ylabel='Sample Correlation')
        plt.xlabel('Lag')
        handles, labels = [(a + b) for a, b in zip(ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
        fig.legend(handles, labels, loc='upper right')
        if save_path:
            plt.savefig(save_path + '_obs_predicted.jpg')
        plt.show()

    def plot_splits(self, save_path=None):
        """Plot train, val, and test sets"""
        plt.figure(figsize=self.fig_size)
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
