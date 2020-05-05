# -*- coding: UTF-8 -*-
""""
Created on 23.12.19

:author:     Martin Dočekal
"""
from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

setup_args = dict(
    name='windPyTransformers',
    version='1.0.0',
    description='Utils for transformers models.',
    long_description_content_type="text/markdown",
    long_description=README,
    license='The Unlicense',
    packages=find_packages(),
    author='Martin Dočekal',
    keywords=['utils', 'PyTorch', 'transformers', 'general usage'],
    url='https://github.com/windionleaf/windPyTransformers',
    install_requires=[
        'windpyutils @ https://github.com/windionleaf/windPyUtils/archive/master.zip#egg=windpyutils',
        'windpytorchutils @ https://github.com/windionleaf/windPyTorchUtils/archive/master.zip#egg=windpytorchutils',
        'tqdm>=4.41.1',
        'torch>=1.3.1',
        'transformers>=2.3.0'
    ]
)

if __name__ == '__main__':
    setup(**setup_args)