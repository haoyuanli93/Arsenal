from setuptools import setup

setup(
    name='arsenal',
    version='dev',
    description=(
        'This package contains functions Haoyuan has written for SPI data processing. '),
    long_description="I save all my useful functions in this package.",
    author='Haoyuan Li',
    author_email='hyli16@stanford.edu',
    maintainer='Haoyuan Li',
    maintainer_email='hyli16@stanford.edu',
    license='BSD License',
    packages=["arsenal", ],
    install_requires=['numpy',
                      'matplotlib',
                      'h5py',
                      'numba',
                      'holoviews',
                      'scipy',
                      'mpi4py',
                      'scikit-image',
                      'imageio'],
    platforms=["Linux"],
    url='https://github.com/haoyuanli93/DiffusionMap'
)
