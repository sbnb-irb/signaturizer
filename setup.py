from setuptools import setup, find_packages

__author__ = """Martino Bertoni"""
__email__ = 'martino.bertoni@irbbarcelona.org'
__version__ = '1.1.9'

setup(
    name='signaturizer',
    version=__version__,
    author=__author__,
    author_email=__email__,
    description='Generate Chemical Checker signatures from molecules SMILES.',
    long_description=open('README.md').read().strip(),
    long_description_content_type="text/markdown",
    url='http://gitlabsbnb.irbbarcelona.org/packages/signaturizer',
    py_modules=find_packages(),
    packages=find_packages(),
    install_requires=[
        'tensorflow<2.4',
        'tensorflow_hub',
        'numpy',
        'h5py',
        'tqdm'],
    zip_safe=False,
    license='MIT License',
    keywords='signaturizer bioactivity signatures chemicalchecker chemoinformatics',
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ])
