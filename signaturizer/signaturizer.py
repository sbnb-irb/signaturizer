# using the module
import os
import h5py
import itertools
import numpy as np
from tqdm import tqdm
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    raise ImportError("requires RDKit " +
                      "https://www.rdkit.org/docs/Install.html")


class Signaturizer():
    """Class loading TF-hub module and performing predictions."""

    def __init__(self, model_name, verbose=True,
                 base_url="https://chemicalchecker.org/signaturizer/"):
        """Initialize the Signaturizer.

        Args:
            model(str): The model to load. Possible values:
                - the model name (the bioactivity space (e.g. "B1") )
                - the model path (the directory containing 'saved_model.pb')
                - a list of models names or paths (e.g. ["B1", "B2", "E5"])
                - 'GLOBAL' to get the global (i.e. horizontally stacked)
                    bioactivity signature.
            base_url(str): The ChemicalChecker getModel API URL.
        """
        self.verbose = verbose
        if isinstance(model_name, list):
            models = model_name
        else:
            if model_name.upper() == 'GLOBAL':
                models = list(itertools.product("ABCDE", "12345"))
            else:
                models = [model_name]
        # load modules
        self.modules = list()
        self.graph = tf.Graph()
        with self.graph.as_default():
            for model in models:
                if os.path.isdir(model):
                    spec = hub.create_module_spec_from_saved_model(model)
                    module = hub.Module(spec, tags=['serve'])
                else:
                    module = hub.Module(base_url + model, tags=['serve'])
                self.modules.append(module)

    def predict(self, smiles, destination=None, chunk_size=1000):
        """Predict signatures for given SMILES.

        Args:
            smiles(list): List of SMILES strings.
            chunk_size(int): Perform prediction on chunks of this size.
            destination(str): File path where to save predictions.
        Returns:
            results: `SignaturizerResult` class.
        """
        with self.graph.as_default():
            with tf.Session() as sess:
                sess.run(tf.tables_initializer())
                sess.run(tf.global_variables_initializer())
                # Prepare result object
                features = len(self.modules) * 128
                results = SignaturizerResult(len(smiles), destination,
                                             features)
                # predict by chunk
                all_chunks = range(0, len(smiles), chunk_size)
                for i in tqdm(all_chunks, disable=self.verbose):
                    chunk = slice(i, i + chunk_size)
                    sign0s = list()
                    failed = list()
                    for idx, mol_smiles in enumerate(smiles[chunk]):
                        try:
                            # read SMILES as molecules
                            mol = Chem.MolFromSmiles(mol_smiles)
                            if mol is None:
                                raise Exception(
                                    "Cannot get molecule from smiles.")
                            info = {}
                            fp = AllChem.GetMorganFingerprintAsBitVect(
                                mol, 2, nBits=2048, bitInfo=info)
                            bin_s0 = [fp.GetBit(i) for i in range(
                                fp.GetNumBits())]
                            calc_s0 = np.array(bin_s0).astype(np.float32)
                        except Exception as err:
                            # in case of failure save idx to fill NaNs
                            print("SKIPPING %s: %s", mol_smiles, str(err))
                            failed.append(idx)
                            calc_s0 = np.full((2048, ),  np.nan)
                        finally:
                            sign0s.append(calc_s0)
                    # stack input fingerprints and run predictor
                    sign0s = np.vstack(sign0s)
                    for idx, module in enumerate(self.modules):
                        pred = module(sign0s, signature='serving_default')
                        preds = sess.run(pred)
                        # add NaN where SMILES conversion failed
                        if failed:
                            preds[np.array(failed)] = np.full((128, ),  np.nan)
                        # save chunk to results dictionary
                        mdl_cols = slice(idx * 128, (idx + 1) * 128)
                        results.signature[chunk, mdl_cols] = preds
        results.close()
        return results


class SignaturizerResult():
    """Class storing result of the prediction.

    Results are stored in the following numpy vector:
        signatures: 128 float32 defining the molecule signature.

    If a destination is specified the result are saved in an HDF5 file with
    the same vector available as HDF5 datasets.
    """

    def __init__(self, size, destination, features=128):
        """Initialize the result containers.

        Args:
            size(int): The number of molecules being signaturized.
            destination(str): Path to HDF5 file where prediction results will
                be saved.
        """
        self.dst = destination
        if self.dst is None:
            # simple numpy arrays
            self.h5 = None
            self.signature = np.zeros((size, features), dtype=np.float32)
        else:
            # check if the file exists already
            if os.path.isfile(self.dst):
                print('HDF5 file %s exists, opening in read-only.' % self.dst)
                # this avoid overwriting by mistake
                self.h5 = h5py.File(self.dst, 'r')
            else:
                # create the datasets
                self.h5 = h5py.File(self.dst, 'w')
                self.h5.create_dataset(
                    'signature', (size, features), dtype=np.float32)
            # expose the datasets
            self.signature = self.h5['signature']

    def close(self):
        if self.h5 is None:
            return
        self.h5.close()
        # leave it open for reading
        self.h5 = h5py.File(self.dst, 'r')
        # expose the datasets
        self.signature = self.h5['signature']
