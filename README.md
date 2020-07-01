# Signaturizer

![alt text](http://gitlabsbnb.irbbarcelona.org/packages/signaturizer/raw/master/images/cc_signatures.jpg "Molecule Signaturization")

Bioactivity signatures are multi-dimensional vectors that capture biological
traits of the molecule (for example, its target profile) in a numerical vector
format that is akin to the structural descriptors or fingerprints used in the
field of chemoinformatics.

Our **signaturizers** relate to bioactivities of 25 different types (including
target profiles, cellular response and clinical outcomes) and can be used as
drop-in replacements for chemical descriptors in day-to-day chemoinformatics
tasks.

For and overview of the different bioctivity descriptors available please check
the original Chemical Checker 
[paper](https://www.nature.com/articles/s41587-020-0502-7) or 
[website](https://chemicalchecker.com/)


# Installation


## from PyPI

```bash
pip install signaturizer
```

## from Git repository

```bash
pip install git+http://gitlabsbnb.irbbarcelona.org/packages/signaturizer.git
```



# Usage


## Generating Bioactivity Signatures

```python
from signaturizer import Signaturizer
# load the predictor for B1 space (representing the Mode of Action)
sign = Signaturizer('/aloy/web_checker/exported_smilespreds/B1')
# prepare a list of SMILES strings
smiles = ['C', 'CCC']
# run prediction
results = sign.predict(smiles)
print(results.signature)
# [[-0.05777782  0.09858645 -0.09854423 ... -0.04505355  0.09859559
#    0.09859559]
#  [ 0.03842233  0.10035036 -0.10023173 ... -0.07104399  0.10035563
#    0.10035574]
print(results.signature.shape)
# (2, 128)
# or save results as H5 file if you have many molecules
results = sign.predict(smiles, 'destination.h5')
```


## Generating Multiple Bioactivity Signatures
```python
from signaturizer import Signaturizer
# load the bioactivity space predictor for all space
models = ['/aloy/web_checker/exported_smilespreds/%s%s' % (y,x) for y in 'ABCDE' for x in '12345']
sign = Signaturizer(models)
# prepare a list of SMILES strings
smiles = ['C', 'CCC']
# run prediction
results = sign.predict(smiles)
print(results.signature.shape)
# (2, 3200)
```
