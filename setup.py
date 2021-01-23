from setuptools import setup, find_packages

setup(
  name = 'dc-gan-pytorch',
  packages = find_packages(),
  version = '0.0.1',
  license='MIT',
  description = 'DC-GAN - Pytorch',
  author = 'Shauray',
  author_email = 'shauray9@gmail.com',
  url = 'https://github.com/shauray8/DCGAN-pytorch',
  keywords = [
    'artificial intelligence',
    'generative adversarial network'
  ],
  install_requires=[
    'einops>=0.3',
    'torch>=1.7.1+cu101'
  ],
  classifiers=[
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.8',
  ],
)
