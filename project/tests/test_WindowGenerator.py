from ..WindowGenerator import WindowGenerator
import pytest

bitcoin = 'data/bitcoin_query.csv'
bitcoincash = 'data/bitcoincash_query.csv'
dash = 'data/dash_query.csv'
dogecoin = 'data/dogecoin_query.csv'
litecoin = 'data/litecoin_query.csv'
zcash = 'data/zcash_query.csv'


@pytest.mark.parametrize("fname", [bitcoin])
def test_basic_loading(fname):
    w1 = WindowGenerator(fname,
                         input_width=24,
                         shift=24,
                         label_width=24,
                         label_columns=['num_transactions'],
                         standardize=True)
    w1.plot()
    w1.plot_splits()
    print(w1)
    print(w1.train_mean)
    print(w1.train_std)

@pytest.mark.skip(reason='skipping for now')
@pytest.mark.parametrize("fname", [bitcoin])
def test_multi_window(fname):
    OUT_STEPS = 24
    multi_window = WindowGenerator(fname,
                                   input_width=24,
                                   label_width=OUT_STEPS,
                                   shift=OUT_STEPS,
                                   label_columns=['num_transactions'],
                                   resample_frequency='D')
    multi_window.plot_splits('figures/test_splits.jpg')
    multi_window.plot(save_path='../figures/test_multi_window.jpg')
    multi_window