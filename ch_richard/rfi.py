
import numpy as np

rfi_lines = np.array([[ 111,  138],
                      #[ 889,  893],
                      [ 856,  860],
                      [ 873,  877],
                      #[ 552,  570],  # Stuff in between (merge?)
                      #[ 583,  600],
                      [ 552,  600],
                      [ 630,  645],
                      [ 645,  661],  # Weak spill over?
                      [ 676,  692],
                      [ 753,  768],
                      [ 783,  799],
                      [ 895,  896],
                      [ 897,  898],
                      [ 882,  884],
                      [ 808,  811],
                      [ 512,  513],
                      [ 707,  709],
                      [ 268,  269],
                      [ 273,  274],
                      [ 384,  385],
                      [ 600,  601],
                      [ 611,  612],
                      [ 396,  397],
                      [ 846,  847],
                      [ 890,  892],
                      [ 991, 1024],  # End of the band. Masking out anyway.
                      [   0,    1],  # Start of band is also odd.
#                      [ 798,  799],
                      ])

# always_cut = range(111,138) + range(889,893) + range(856,860) + \
#         range(873,877) + range(583,600) + range(552,568) + range(630,645) +\
#         range(678,690) + range(753, 768) + range(783, 798)


def mask_rfi(dset):
    """Apply an RFI mask.

    This routine modifies only the mask part of the dset.

    Parameters
    ----------
    dset : dataset.MPIDataset
        The data we want to clean.

    Returns
    -------
    cleaned_dset : dataset.MPIDataset
    """

    dset = mask_known_lines(dset)

    return dset


def mask_known_lines(dset):
    """Apply an RFI mask to remove known RFI lines.

    This routine modifies only the mask part of the dset.

    Parameters
    ----------
    dset : dataset.MPIDataset
        The data we want to clean.

    Returns
    -------
    cleaned_dset : dataset.MPIDataset
    """

    # Find which lines are actually in the section on this process
    line_mask = np.logical_and(rfi_lines[:, 1] > dset.freq_start,
                               rfi_lines[:, 0] < dset.freq_end)

    # Extract those lines
    lines_in_section = rfi_lines[np.where(line_mask)]

    # Ensure line endpoints are within the file.
    lines_in_section = np.maximum(lines_in_section, dset.freq_start)
    lines_in_section = np.minimum(lines_in_section, dset.freq_end)

    # Remove the start frequency to get bin indices
    line_indices = lines_in_section - dset.freq_start

    # Iteratre over lines and mask them out.
    for line_start, line_end in line_indices:
        dset.mask[line_start:line_end] = 0.0

    return dset
