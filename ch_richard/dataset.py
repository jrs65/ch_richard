import numpy as np
from mpi4py import MPI

import h5py

from drift.util import mpiutil

from ch_util import andata


class MPIDataset(object):

    data = None
    mask = None

    timestamp = None
    time_start = None
    time_end = None
    global_timestamp = None

    freq_start = None
    freq_end = None
    global_nfreq = None
    global_freq = None

    # Independent in case we have a mixed type. ?!
    freq_split = True
    time_split = False

    @property
    def masked_data(self):
        """Dataset as a numpy masked array. Data is a view, but mask is a copy
        and won't be updated."""

        np_mask = ~self.mask.astype(np.bool)
        return np.ma.array(self.data, mask=np_mask, copy=False)

    @property
    def timestamp(self):
        """Timestamps local to this process."""
        return self.global_timestamp[self.time_start:self.time_end]

    @property
    def freq(self):
        """Frequencies local to this process."""
        return self.global_freq[self.freq_start:self.freq_end]

    @property
    def ntime(self):
        """Number of local timestamps."""
        return (self.time_end - self.time_start)

    @property
    def nfreq(self):
        """Number of local frequencies."""
        return (self.freq_end - self.freq_start)

    # @property
    # def nprod(self):
    #     """Number of correlation products."""
    #     return self.data.shape[1]

    @property
    def global_ntime(self):
        """Global number of timestamps."""
        return self.global_timestamp.shape[0]

    @property
    def global_shape(self):
        """Shape of the global array."""
        return (self.global_nfreq,) + self.data.shape[1:-1] + (self.global_ntime,)

    @classmethod
    def from_files(cls, filelist, acq=True):
        """Load from a set of CHIME data files.

        Parameters
        ----------
        filelist : list
            List of filenames to load.
        acq : boolean, optional
            Are the files analysis files or acquisition files (default)?

        Returns
        -------
        mpi_dset : MPIDataset
            Distributed dataset type.
        """
        mpi_dset = cls()

        filelist.sort()

        # Global number of files
        ngfiles = len(filelist)

        # Split into local set of files.
        lf, sf, ef = mpiutil.split_local(ngfiles)
        local_files = filelist[sf:ef]

        fshape = None

        # Set file loading routine depending on whether files are acq or
        # analysis.
        _load_file = andata.AnData.from_acq_h5 if acq else andata.AnData.from_file

        # Rank 0 should open file and check the shape.
        if mpiutil.rank0:
            d0 = _load_file(local_files[0])
            fshape = d0.datasets['vis'].shape

        # Broadcast the shape to all other ranks
        fshape = mpiutil.world.bcast(fshape, root=0)

        # Unpack to get the individual lengths
        nfreq, nprod, ntime = fshape

        # This will be the local shape, file ordered.
        lshape = (lf, ntime, nprod, nfreq)

        local_array = np.zeros(lshape, dtype=np.complex128)

        # Timestamps
        timestamps = []

        for li, lfile in enumerate(local_files):

            print "Rank %i reading %s" % (mpiutil.rank, lfile)
            # Load file
            df = _load_file(lfile)

            # Copy data into local dataset
            dset = df.datasets['vis']

            if dset.shape != fshape:
                raise Exception("Data from %s is not the right shape" % lfile)

            local_array[li] = dset.T

            # Get timestamps
            timestamps.append((li + sf, df.timestamp))


        ## Merge timestamps
        tslist = mpiutil.world.allgather(timestamps)
        tsflat = [ts for proclist in tslist for ts in proclist]  # Flatten list

        # Add timestamps into array
        timestamp_array = np.zeros((ngfiles, ntime), dtype=np.float64)
        for ind, tstamps in tsflat:
            timestamp_array[ind] = tstamps
        timestamp_array = timestamp_array.reshape(ngfiles * ntime)

        if mpiutil.rank0:
            print "Starting transpose...",

        data_by_freq = mpiutil.transpose_blocks(local_array, (ngfiles, ntime, nprod, nfreq))

        if mpiutil.rank0:
            print " done."
            
        # Get local frequencies
        lfreq, sfreq, efreq = mpiutil.split_local(nfreq)

        data_by_freq = data_by_freq.reshape((ngfiles * ntime, nprod, lfreq)).T

        # Set dataset
        mpi_dset.data = data_by_freq
        mpi_dset.mask = np.ones_like(mpi_dset.data, dtype=np.int8)

        # Set time properties
        mpi_dset.global_timestamp = timestamp_array
        mpi_dset.time_start = 0
        mpi_dset.time_end = mpi_dset.timestamp.shape[0]

        # Set frequency properties
        mpi_dset.freq_start = sfreq
        mpi_dset.freq_end = efreq
        mpi_dset.global_nfreq = nfreq

        return mpi_dset


    @classmethod
    def load(self, filename, freq_split=True):
        """Load a large dataset from a single file on disk.

        Parameters
        ----------
        filename : string
            File to load.
        freq_split : boolean
            Split file across nodes by frequency (default) or by time.

        Returns
        -------
        mpi_dset : MPIDataset
        """

        mpi_dset = MPIDataset()

        with h5py.File(filename, 'r') as f:

            mpi_dset.global_timestamp = f['timestamp'][:]
            mpi_dset.global_nfreq = f['vis'].shape[0]

            if freq_split:
                st = 0
                et = mpi_dset.global_timestamp.shape[0]
                nf, sf, ef = mpiutil.split_local(mpi_dset.global_nfreq)
            else:
                # Reset flags of which split we are in
                mpi_dset.freq_split = False
                mpi_dset.time_split = True

                sf = 0
                ef = mpi_dset.global_nfreq
                nt, st, et = mpiutil.split_local(mpi_dset.global_timestamp.shape[0])

            # Copy correct section of data from file.
            mpi_dset.data = f['vis'][sf:ef, ..., st:et][:]

            # Load mask if required
            if 'mask' in f:
                mpi_dset.mask = f['mask'][sf:ef, ..., st:et][:]
            else:
                mpi_dset.mask = np.ones_like(mpi_dset.data, dtype=np.int8)

            mpi_dset.time_start = st
            mpi_dset.time_end = et
            mpi_dset.freq_start = sf
            mpi_dset.freq_end = ef

        return mpi_dset


    def save(self, filename):
        """Save large dataset into a single file.

        Parameters
        ----------
        filename : string
            Name of file to save into. If it exists already it is destroyed.
        """
        if mpiutil.rank0:

            with h5py.File(filename, 'w') as f:
                f.create_dataset('timestamp', data=self.global_timestamp)

                f.create_dataset('vis', self.global_shape, dtype=np.complex128)
                f.create_dataset('mask', self.global_shape, dtype=np.int8)

        for pi in range(mpiutil.size):
            mpiutil.world.Barrier()

            if mpiutil.rank == pi:
                print "Rank %i writing." % mpiutil.rank

                with h5py.File(filename, 'a') as f:
                    f['vis'][self.freq_start:self.freq_end, ..., self.time_start:self.time_end] = self.data
                    f['mask'][self.freq_start:self.freq_end, ..., self.time_start:self.time_end] = self.mask

        mpiutil.world.Barrier()


    def _save_separate(self, filename):

        for pi in range(mpiutil.size):

            if mpiutil.rank == pi:

                fname = filename + ('.r_%i' % mpiutil.rank)

                print "Rank %i writing into: %s" % (mpiutil.rank, fname)

                with h5py.File(fname, 'w') as f:
                    f.create_dataset('timestamp', data=self.timestamp)
                    f.create_dataset('vis', data=self.data)

        mpiutil.world.Barrier()


    def to_freq_split(self):
        """Transform from a time split dataset to a frequency split one.

        Returns
        -------
        mpi_dset : MPIDataset
            A dataset which is now distributed along the freq axis.
        """
        if self.freq_split:
            return self

        if not self.time_split:
            raise Exception("Can't transform from mixed splitting.")

        fsplit_dset = MPIDataset()

        # Set global properties
        fsplit_dset.global_timestamp = self.global_timestamp
        fsplit_dset.global_nfreq = self.global_nfreq

        # Set local frequency properties
        fsplit_dset.time_start = 0
        fsplit_dset.time_end = self.global_timestamp.size


        # Determine local time properties
        lf, sf, ef = mpiutil.split_local(self.global_nfreq)
        fsplit_dset.freq_start = sf
        fsplit_dset.freq_end = ef

        fs_data = mpiutil.transpose_blocks(self.data.T, self.global_shape[::-1])
        fs_mask = mpiutil.transpose_blocks(self.mask.T, self.global_shape[::-1])

        fsplit_dset.data = fs_data.T
        fsplit_dset.mask = fs_mask.T

        return fsplit_dset


    def to_time_split(self):
        """Transform from a frequency split dataset to a time split one.

        Returns
        -------
        mpi_dset : MPIDataset
            A dataset which is now distributed along the time axis.
        """

        if self.time_split:
            return self

        if not self.freq_split:
            raise Exception("Can't transform from mixed splitting.")

        tsplit_dset = MPIDataset()

        # Set global properties
        tsplit_dset.global_timestamp = self.global_timestamp
        tsplit_dset.global_nfreq = self.global_nfreq

        # Set local frequency properties
        tsplit_dset.freq_start = 0
        tsplit_dset.freq_end = self.global_nfreq

        # Determine local time properties
        gtime = self.global_timestamp.size
        lt, st, et = mpiutil.split_local(gtime)
        tsplit_dset.time_start = st
        tsplit_dset.time_end = et

        # MPI transpose the data
        ts_data = mpiutil.transpose_blocks(self.data, self.global_shape)
        ts_mask = mpiutil.transpose_blocks(self.mask, self.global_shape)
        tsplit_dset.data = ts_data
        tsplit_dset.mask = ts_mask

        tsplit_dset.freq_split = False
        tsplit_dset.time_split = True

        return tsplit_dset

