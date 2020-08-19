import setuptools
from setuptools import find_packages
import signaturizer

setuptools.setup(
    name='signaturizer',
    version=signaturizer.__version__,
    description='Generate Chemical Checker signatures from molecules SMILES.',
    long_description=open('README.md').read().strip(),
    long_description_content_type="text/markdown",
    author='Martino Bertoni',
    author_email='martino.bertoni@irbarcelona.org',
    url='http://gitlabsbnb.irbbarcelona.org/packages/signaturizer',
    py_modules=find_packages(),
    packages=find_packages(),
    install_requires=[
        'tensorflow<2',
        'tensorflow_hub',
        'numpy',
        'h5py'],
    license='MIT License',
    zip_safe=False,
    keywords='signaturizer bioactivity signatures chemicalchecker',
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ])
