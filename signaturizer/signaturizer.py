# using the module
import os
import h5py
import shutil
import tempfile
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow_hub as hub
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    raise ImportError("requires RDKit " +
                      "https://www.rdkit.org/docs/Install.html")


class Signaturizer():
    """Class loading TF-hub module and performing predictions."""

    def __init__(self, model_name, verbose=True, compressed=True, local=False,
                 cc_url="https://dynbench3d.irbbarcelona.org/.well-known/acme-challenge/"):
        """Initialize the Signaturizer.

        Args:
            model_name(str): The model name, i.e. the bioactivity space of
                interest (e.g. "A1")
            cc_url(str): The ChemicalChecker getModel API URL.
        """
        # Model url
        if local is False:
            model_url = cc_url + model_name
        else:
            model_url = model_name
        if compressed:
            model_url += '.tar.gz'
        self.verbose = verbose
        # load Module
        print('model_url', model_url)
        spec = hub.create_module_spec_from_saved_model(model_url)
        self.module = hub.Module(spec, tags=['serve'])

    @staticmethod
    def _export_smilespred_as_module(smilespred_path, module_destination, tmp_path=None):
        from keras import backend as K
        from chemicalchecker.tool.smilespred import Smilespred
        smilespred = Smilespred(smilespred_path)
        smilespred.build_model(load=True)
        model = smilespred.model
        signature = tf.saved_model.signature_def_utils.predict_signature_def(
            inputs={'default': model.input}, outputs={'default': model.output})
        if tmp_path is None:
            tmp_path = tempfile.mkdtemp()
        print("_export_smilespred_as_module", tmp_path)
        builder = tf.saved_model.builder.SavedModelBuilder(tmp_path)
        builder.add_meta_graph_and_variables(
            sess=K.get_session(),
            tags=['serve'],
            signature_def_map={'serving_default': signature
                               })
        builder.save()
        Signaturizer._export_savedmodel_as_module(tmp_path, module_destination)
        # clean temporary folder
        # shutil.rmtree(tmp_path)

    @staticmethod
    def _export_savedmodel_as_module(savedmodel_path, module_destination, tmp_path=None):
        """Export tensorflow SavedModel to the TF-hub module format."""
        # Create ModuleSpec
        spec = hub.create_module_spec_from_saved_model(
            savedmodel_path, drop_collections=['saved_model_train_op'])
        # Initialize Graph and export session to temporary folder
        if tmp_path is None:
            tmp_path = tempfile.mkdtemp()
        print("_export_savedmodel_as_module", tmp_path)
        with tf.Graph().as_default():
            module = hub.Module(spec, tags=['serve'])
            with tf.Session() as session:
                session.run(tf.tables_initializer())
                session.run(tf.global_variables_initializer())
                module.export(tmp_path, session)
        # compress the exported files
        os.system("tar -cz -f %s --owner=0 --group=0 -C %s ." %
                  (module_destination, tmp_path))
        # clean temporary folder
        # shutil.rmtree(tmp_path)

    def predict(self, smiles, destination=None, chunk_size=1000):
        """Predict signatures for given SMILES.

        Args:
            smiles(list): A list of SMILES strings.
            chunk_size(int): Perform prediction on chuncks of this size.
            destination(str): Path to H5 file where prediction results will
                be saved.
        Returns:
            results: `SignaturizerResult` class.
        """
        # Init TF session
        with tf.Session() as session:
            # Init Graph ariables
            session.run(tf.tables_initializer())
            session.run(tf.global_variables_initializer())
            # Prepare result object
            results = SignaturizerResult(len(smiles), destination)
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
                            raise Exception("Cannot get molecule from smiles.")
                        info = {}
                        fp = AllChem.GetMorganFingerprintAsBitVect(
                            mol, 2, nBits=2048, bitInfo=info)
                        bin_s0 = [fp.GetBit(i) for i in range(fp.GetNumBits())]
                        calc_s0 = np.array(bin_s0).astype(np.float32)
                    except Exception as err:
                        # in case of failure save idx to later append NaNs
                        print("SKIPPING %s: %s", mol_smiles, str(err))
                        failed.append(idx)
                        calc_s0 = np.full((2048, ),  np.nan)
                    finally:
                        sign0s.append(calc_s0)
                # stack input fingerprints and run predictor
                sign0s = np.vstack(sign0s)
                pred = self.module(sign0s, signature='serving_default')
                preds = session.run(pred)
                print('preds', preds)
                # add NaN where SMILES conversion failed
                if failed:
                    preds[np.array(failed)] = np.full((131, ),  np.nan)
                # save chunk to results dictionary
                results.signature[chunk] = preds[:, :128]
        results.close()
        return results


class SignaturizerResult():
    """Class storing result of the prediction.

    Results are stored in the following numpy vectors:
        signatures: 128 float32 defining the moleule signature.
        stddev_norm: standard deviation of the signature.
        intensity_norm: intensity of the consensus.
        confidence: signature confidence.

    If a destination is specified the result are saved in an H5 file with
    the same vector available as H5 datasets.
    """

    def __init__(self, size, destination):
        """Initialize the result containers.

        Args:
            size(int): The number of molecules being signaturized.
            destination(str): Path to H5 file where prediction results will
                be saved.
        """
        self.dst = destination
        if self.dst is None:
            # simply numpy arrays
            self.h5 = None
            self.signature = np.zeros((size, 128), dtype=np.float32)
        else:
            # check if the file exists already
            if os.path.isfile(self.dst):
                print('H5 file %s exists, opening in read-only.' % self.dst)
                # this avoid overwriting by mistake
                self.h5 = h5py.File(self.dst, 'r')
            else:
                # create the datasets
                self.h5 = h5py.File(self.dst, 'w')
                self.h5.create_dataset(
                    'signature', (size, 128), dtype=np.float32)
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


# UNIT TEST
from chemicalchecker import ChemicalChecker
from chemicalchecker.core.signature_data import DataSignature
test_smiles = ['CCC', 'C']

cc = ChemicalChecker()
s3 = cc.signature('B1.001', 'sign3')
s3.predict_from_smiles(test_smiles, './tmp.h5')
pred1 = DataSignature('./tmp_pred1.h5')

a = Signaturizer('/tmp/moduledir/', compressed=False, local=True)
module_destination = './tmp_dest'
Signaturizer._export_smilespred_as_module(
        os.path.join(s3.module_path, 'smiles_final'),
        module_destination, tmp_path='./conv_k2tf')
module2 = Signaturizer('./conv_k2tf', compressed=False, local=True)
pred2 = module2.predict(test_smiles)
Signaturizer._export_savedmodel_as_module(
        os.path.join(s3.module_path, 'smiles_final'),
        module_destination, tmp_path='./conv_tf2hub')
module3 = Signaturizer('./conv_tf2hub', compressed=False, local=True)
pred3 = module3.predict(test_smiles)
assert(pred1 == pred2)
assert(pred1 == pred3)
