
import numpy as np

import dataset


def dm_delays(freq, dm, f_ref=400.0):
    """Provides dispersion delays as a function of frequency.

    Parameters
    ----------
    freq : np.ndarray
        Array of frequencies.
    dm : float
        Dispersion measure in pc/cm**3
    f_ref : float
        Reference frequency for time delays

    Returns
    -------
    Vector of delays in seconds as a function of frequency
    """

    return 4.148808e3 * dm * (freq**(-2) - f_ref**(-2))


def fold_pulsar(dset, p0, dm, ngate=32, ntrebin=1000):
    """Folds pulsar into nbins after dedispersing it.

    Parameters
    ----------
    dset : MPIDataset
        Dataset to fold.
    p0 : float
        Pulsar period in seconds.
    dm : float
        Dispersion measure in pc/cm**3
    ngate : integer
        Number of bins to fold pulsar into.
    ntrebin : integer
        Number of time samples to rebin into.

    Returns
    -------
    profile : MPIDataset[nfreq, nprod, ngate, ntime]
        Folded pulse profile of length nbins
    """

    nt = dset.ntime / ntrebin

    fshape = dset.data.shape[:-1] + (ngate, nt)

    fold_data = np.zeros(fshape, dtype=np.complex128)
    fold_mask = np.zeros(fshape, dtype=np.int8)

    delays = dm_delays(dset.freq, dm)

    tstamps_all = dset.timestamp[:(nt * ntrebin)].reshape(nt, ntrebin)

    tshape = dset.data.shape[:-1] + (nt, ntrebin)
    trim_dset = dset.data[..., :(nt * ntrebin)].reshape(tshape)
    trim_mask = dset.mask[..., :(nt * ntrebin)].reshape(tshape)

    ## Loop over new timebins, freq and prod and bin into gates
    for ti in range(nt):

        tstamps = tstamps_all[ti]

        for fi in range(dset.nfreq):

            tstamp_f = tstamps - delays[fi]
            bin = (((tstamp_f / p0) % 1.0) * ngate).astype(np.int)

            for pi in range(dset.data.shape[1]):
                ds = trim_dset[fi, pi, ti].copy()
                mask = trim_mask[fi, pi, ti]

                #dset /= running_mean(dset)

                # for gi in range(ngate):
                #     vals = dset[bin == gi]
                #     m1 = mask[bin == gi]
                #     v = vals[m1].mean()
                #     fold_data[fi, pi, gi, ti] = v

                data_fold_r = np.bincount(bin, weights=ds.real, minlength=ngate)
                data_fold_i = np.bincount(bin, weights=ds.imag, minlength=ngate)
                mask_fold = np.bincount(bin, weights=mask, minlength=ngate)

                # Remove invalid entries
                maskf = np.where(mask_fold > 0.0, 1.0 / mask_fold, np.zeros_like(mask_fold))

                gate_data = (data_fold_r + 1.0J * data_fold_i) * maskf

                fold_data[fi, pi, :, ti] = gate_data
                fold_mask[fi, pi, :, ti] = (mask_fold > 0)


    averaged_ts = tstamps_all.mean(axis=1)

    # Construct an MPIDataset to hold the folded data.
    folded_dset = dataset.MPIDataset()

    folded_dset.global_timestamp = averaged_ts
    folded_dset.time_start = 0
    folded_dset.time_end = nt

    folded_dset.freq_start = dset.freq_start
    folded_dset.freq_end = dset.freq_end
    folded_dset.global_freq = dset.global_freq
    folded_dset.global_nfreq = dset.global_nfreq

    folded_dset.data = fold_data
    folded_dset.mask = fold_mask


    return folded_dset


# def running_mean(arr, radius=50):
#     """                                                                                                                
#     Not clear why this works. Need to think more about it.                                                                             
#     """
#     arr = abs(arr)
#     n = radius*2+1
#     padded = np.concatenate((arr[1:radius+1][::-1], arr, arr[-radius-1:-1][::-1]), axis=0)
#     ret = np.cumsum(padded, axis=0, dtype=float)
#     ret[n:] = ret[n:] - ret[:-n]
    
#     return ret[n-1:] / n
