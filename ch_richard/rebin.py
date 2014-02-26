
import numpy as np

import dataset


def rebin_time(dset, ntrebin=1000):
    """Rebin data in timeby a given factor.

    Parameters
    ----------
    dset : MPIDataset
        Dataset to rebin.
    ntrebin : integer
        Number of time samples to rebin into.

    Returns
    -------
    new_dset : MPIDataset
        Rebinned dataset.
    """

    nt = dset.ntime / ntrebin

    tshape = dset.data.shape[:-1] + (nt, ntrebin)
    rebin_data = dset.data[..., :(nt * ntrebin)].reshape(tshape).sum(axis=-1)
    rebin_mask = dset.mask[..., :(nt * ntrebin)].reshape(tshape).mean(axis=-1, dtype=np.float64)
    rebin_mask = (rebin_mask > 0).astype(np.int8)

    rebin_tstamps = dset.timestamp[:(nt * ntrebin)].reshape(nt, ntrebin).mean(axis=-1)

    # Construct an MPIDataset to hold the rebinned data.
    rebin_dset = dataset.MPIDataset()

    rebin_dset.global_timestamp = rebin_tstamps
    rebin_dset.time_start = 0
    rebin_dset.time_end = nt

    rebin_dset.freq_start = dset.freq_start
    rebin_dset.freq_end = dset.freq_end
    rebin_dset.global_freq = dset.global_freq
    rebin_dset.global_nfreq = dset.global_nfreq

    rebin_dset.data = rebin_data
    rebin_dset.mask = rebin_mask


    return rebin_dset
