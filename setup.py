from setuptools import setup, find_packages

setup(
    name = 'ch_richard',
    version = 0.1,

    packages = find_packages(),
    requires = ['numpy', 'scipy', 'healpy', 'h5py', 'cora'],
    scripts = ['scripts/foldpulsar'],

    # metadata for upload to PyPI
    author = "J. Richard Shaw",
    author_email = "jrs65@cita.utoronto.ca",
    description = "Misc routines. A staging post for developing things.",
    license = "GPL v3.0",
    url = "http://github.com/jrs65/ch_richard"
)
