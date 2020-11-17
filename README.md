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


## Installation

The only strong dependency for this resource is [RDKit](https://www.rdkit.org/docs/Install.html)
which can be installed in a local conda environment.
The installation procedure takes less than 5 minutes.

### Conda environment

```bash
conda create --no-default-packages -n sign -y python=3.7
conda activate sign
conda install -c conda-forge -y rdkit
```
> **_NOTE:_**  On some system with previous tensorflow installation, the following might be needed:
> ```bash
> conda create --no-default-packages -n sign -y python=3.7 tensorflow=1.14.0
> ```

### from PyPI

```bash
pip install signaturizer
```

### from Git repository

```bash
pip install git+http://gitlabsbnb.irbbarcelona.org/packages/signaturizer.git
```


## Basic Usage

### Generating Bioactivity Signatures

```python
from signaturizer import Signaturizer
# load the predictor for B1 space (representing the Mode of Action)
sign = Signaturizer('B1')
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

## Speed

Generating 1000 signatures for one bioactivity spaces takes less than 4 seconds on an average machine with 4 cores.

## Advanced Usage

For an exemplary application please check the ipython [notebook](http://gitlabsbnb.irbbarcelona.org/packages/signaturizer/blob/master/notebook/signaturizer.ipynb) in the `notebook` directory (you can download it and run on [Google Colab](https://colab.research.google.com/notebooks/))


## Citing

If you use this resource in the course of your research, please consider citing 
these papers:


> Bertoni M, et al<br>
> "**Bioactivity descriptors for uncharacterized compounds.**"<br>
> BioaRXiv (2020) [[link]](https://biorxiv.org/cgi/content/short/2020.07.21.214197v1)

> Duran-Frigola M, et al<br>
> "**Extending the small-molecule similarity principle to all levels of biology with the Chemical Checker.**"<br>
> Nature Biotechnology (2020) [[link]](https://www.nature.com/articles/s41587-020-0502-7)



You can use these bibtex entries:

```
@article {Bertoni2020,
    author = {Bertoni, Martino and Duran-Frigola, Miquel and Badia-i-Mompel, Pau and Orozco-Ruiz, Modesto and Guitart-Pla, Oriol and Aloy, Patrick},
    title = {Bioactivity descriptors for uncharacterized compounds},
    elocation-id = {2020.07.21.214197},
    year = {2020},
    doi = {10.1101/2020.07.21.214197},
    publisher = {Cold Spring Harbor Laboratory},
    abstract = {Chemical descriptors encode the physicochemical and structural properties of small molecules, and they are at the core of chemoinformatics. The broad release of bioactivity data has prompted enriched representations of compounds, reaching beyond chemical structures and capturing their known biological properties. Unfortunately, bioactivity descriptors are not available for most small molecules, which limits their applicability to a few thousand well characterized compounds. Here we present a collection of deep neural networks able to infer bioactivity signatures for any compound of interest, even when little or no experimental information is available for them. Our signaturizers relate to bioactivities of 25 different types (including target profiles, cellular response and clinical outcomes) and can be used as drop-in replacements for chemical descriptors in day-to-day chemoinformatics tasks. Indeed, we illustrate how inferred bioactivity signatures are useful to navigate the chemical space in a biologically relevant manner, and unveil higher-order organization in drugs and natural product collections. Moreover, we implement a battery of signature-activity relationship (SigAR) models and show a substantial improvement in performance, with respect to chemistry-based classifiers, across a series of biophysics and physiology activity prediction benchmarks.Competing Interest StatementThe authors have declared no competing interest.},
    URL = {https://www.biorxiv.org/content/early/2020/07/21/2020.07.21.214197},
    eprint = {https://www.biorxiv.org/content/early/2020/07/21/2020.07.21.214197.full.pdf},
    journal = {bioRxiv}
}
```

```
﻿@Article{Duran-Frigola2020,
    author={Duran-Frigola, Miquel and Pauls, Eduardo and Guitart-Pla, Oriol and Bertoni, Martino and Alcalde, V{\'i}ctor and Amat, David and Juan-Blanco, Teresa and Aloy, Patrick},
    title={Extending the small-molecule similarity principle to all levels of biology with the Chemical Checker},
    journal={Nature Biotechnology},
    year={2020},
    month={May},
    day={18},
    abstract={Small molecules are usually compared by their chemical structure, but there is no unified analytic framework for representing and comparing their biological activity. We present the Chemical Checker (CC), which provides processed, harmonized and integrated bioactivity data on {\textasciitilde}800,000 small molecules. The CC divides data into five levels of increasing complexity, from the chemical properties of compounds to their clinical outcomes. In between, it includes targets, off-targets, networks and cell-level information, such as omics data, growth inhibition and morphology. Bioactivity data are expressed in a vector format, extending the concept of chemical similarity to similarity between bioactivity signatures. We show how CC signatures can aid drug discovery tasks, including target identification and library characterization. We also demonstrate the discovery of compounds that reverse and mimic biological signatures of disease models and genetic perturbations in cases that could not be addressed using chemical information alone. Overall, the CC signatures facilitate the conversion of bioactivity data to a format that is readily amenable to machine learning methods.},
    issn={1546-1696},
    doi={10.1038/s41587-020-0502-7},
    url={https://doi.org/10.1038/s41587-020-0502-7}
}
```