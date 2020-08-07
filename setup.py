from distutils.core import setup
setup(
  name = 'NeatMS',
  packages = ['NeatMS'],
  version = 'v0.6',
  license='MIT', 
  description = 'NeatMS is an open source python package for untargeted LCMS deep learning peak curation', 
  author = 'Yoann Gloaguen', 
  author_email = 'yoann.gloaguen@mdc-berlin.de', 
  url = 'https://github.com/bihealth/NeatMS', 
  download_url = 'https://github.com/bihealth/NeatMS/archive/v0.6.tar.gz',
  keywords = ['LCMS', 'Classifier', 'Peak', 'Neural network'], 
  install_requires=[ 
          'pymzml',
          'numpy',
          'pandas',
          'scikit-learn',
          'tensorflow',
          'pillow',
          'h5py',
          'keras'
      ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',   
    'Topic :: Scientific/Engineering :: Bio-Informatics',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7'
  ],
)