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
sign = Signaturizer('A1')
# prepare a list of SMILES strings
smiles = ['C', 'CCC']
# run prediction
results = sign.predict(smiles)
# or save results as H5 file
results = sign.predict(smiles, 'destination.h5')
```