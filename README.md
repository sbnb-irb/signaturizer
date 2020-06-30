# Signaturizer
Generate Chemical Checker signatures from molecules SMILES.

# Install

```
pip install signaturizer
```

# Example
```python
from signaturizer import Signaturizer
# load the bioactivity space predictor
sign = Signaturizer('/aloy/web_checker/exported_smilespreds/A1')
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