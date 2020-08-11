# using the module
import os
import h5py
import shutil
import numpy as np
from tqdm import tqdm
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import tensorflow.compat.v1 as tf
    import tensorflow_hub as hub
    from tensorflow.compat.v1.keras.models import Model
    from tensorflow.compat.v1.keras import Input
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    raise ImportError("requires RDKit " +
                      "https://www.rdkit.org/docs/Install.html")

tf.logging.set_verbosity(tf.logging.ERROR)


class Signaturizer(object):
    """Class loading TF-hub module and performing predictions."""

    def __init__(self, model_name,
                 base_url="http://chemicalchecker.com/api/db/getSignaturizer/",
                 version='v1.1', local=False, tf_version='1', verbose=False,
                 applicability=True):
        """Initialize the Signaturizer.

        Args:
            model(str): The model to load. Possible values:
                - the model name (the bioactivity space (e.g. "B1") )
                - the model path (the directory containing 'saved_model.pb')
                - a list of models names or paths (e.g. ["B1", "B2", "E5"])
                - 'GLOBAL' to get the global (i.e. horizontally stacked)
                    bioactivity signature.
            base_url(str): The ChemicalChecker getModel API URL.
            version(int): Signaturizer version.
            local(bool): Wethere the specified model_name shoudl be
                interpreted as a path to a local model.
            tf_version(int): The Tesorflow version.
            verbose(bool): If True some more information will be printed.
            applicability(bool): Wether to also compute the applicability of
                each prediction.
        """
        self.verbose = verbose
        self.applicability = applicability
        if not isinstance(model_name, list):
            if model_name.upper() == 'GLOBAL':
                self.model_names = [y + x for y in 'ABCDE' for x in '12345']
            else:
                self.model_names = [model_name]
        else:
            self.model_names = model_name
        # load modules as layer to compose a new model
        main_input = Input(shape=(2048,), dtype=tf.float32, name='main_input')
        sign_output = list()
        app_output = list()
        as_dict = False
        output_key = 'default'
        if len(self.model_names) == 1:
            as_dict = True
            output_key = None
        for name in self.model_names:
            # build module spec
            if local:
                if os.path.isdir(name):
                    url = name
                    if self.verbose:
                        print('LOADING local:', url)
                else:
                    raise Exception('Module path not found!')
            else:
                url = base_url + '%s/%s' % (version, name)
                if self.verbose:
                    print('LOADING remote:', url)

            sign_layer = hub.KerasLayer(url, signature='serving_default',
                                        trainable=False,  tags=['serve'],
                                        output_key=output_key,
                                        signature_outputs_as_dict=as_dict)
            sign_layer._is_hub_module_v1 = True
            sign_output.append(sign_layer(main_input))

            if self.applicability:
                try:
                    app_layer = hub.KerasLayer(
                        url, signature='applicability',
                        trainable=False, tags=['serve'],
                        output_key=output_key,
                        signature_outputs_as_dict=as_dict)
                    app_layer._is_hub_module_v1 = True
                    app_output.append(app_layer(main_input))
                except Exception as ex:
                    print('WARNING: applicability predictions not available. '
                          + str(ex))
                    self.applicability = False
        # join signature output and prepare model
        if len(sign_output) > 1:
            sign_output = tf.keras.layers.concatenate(sign_output)
        self.model = Model(inputs=main_input, outputs=sign_output)
        # same for applicability
        if self.applicability:
            if len(app_output) > 1:
                app_output = tf.keras.layers.concatenate(app_output)
            self.app_model = Model(inputs=main_input, outputs=app_output)

    def predict(self, smiles, destination=None, chunk_size=1000):
        """Predict signatures for given SMILES.

        Args:
            smiles(list): List of SMILES strings.
            destination(str): File path where to save predictions.
            chunk_size(int): Perform prediction on chunks of this size.
        Returns:
            results: `SignaturizerResult` class.
        """
        # Prepare result object
        features = len(self.model_names) * 128
        results = SignaturizerResult(len(smiles), destination, features)
        results.dataset[:] = self.model_names
        if results.readonly:
            raise Exception(
                'Destination file already exists, ' +
                'delete or rename to proceed.')

        # predict by chunk
        all_chunks = range(0, len(smiles), chunk_size)
        for i in tqdm(all_chunks, disable=not self.verbose):
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
                    print("SKIPPING %s: %s" % (mol_smiles, str(err)))
                    failed.append(idx)
                    calc_s0 = np.full((2048, ),  np.nan)
                finally:
                    sign0s.append(calc_s0)
            # stack input fingerprints and run signature predictor
            sign0s = np.vstack(sign0s)
            preds = self.model.predict(sign0s)
            # add NaN where SMILES conversion failed
            if failed:
                preds[np.array(failed)] = np.full(features,  np.nan)
            results.signature[chunk] = preds
            # run applicability predictor
            if self.applicability:
                apreds = self.app_model.predict(sign0s)
                if failed:
                    apreds[np.array(failed)] = np.full(features,  np.nan)
                results.applicability[chunk] = apreds
        results.close()
        if self.verbose:
            print('PREDICTION complete!')
        if failed:
            print('The following SMILES could not be recognized,'
                  ' the corresponding signatures are NaN')
            for idx in failed:
                print(smiles[idx])
        return results

    @staticmethod
    def _clear_tfhub_cache():
        cache_dir = os.getenv('TFHUB_CACHE_DIR')
        if cache_dir is None:
            cache_dir = '/tmp/tfhub_modules/'
        if not os.path.isdir(cache_dir):
            raise Exception('Cannot find tfhub cache directory, ' +
                            'please set TFHUB_CACHE_DIR variable')
        shutil.rmtree(cache_dir)
        os.mkdir(cache_dir)


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
        self.readonly = False
        if self.dst is None:
            # simple numpy arrays
            self.h5 = None
            self.signature = np.full((size, features), np.nan, order='F',
                                     dtype=np.float32)
            self.applicability = np.full(
                (size, int(np.ceil(features / 128))), np.nan, dtype=np.float32)
            self.dataset = np.full((int(np.ceil(features / 128)),), np.nan,
                                   dtype=h5py.special_dtype(vlen=str))
        else:
            # check if the file exists already
            if os.path.isfile(self.dst):
                print('HDF5 file %s exists, opening in read-only.' % self.dst)
                # this avoid overwriting by mistake
                self.h5 = h5py.File(self.dst, 'r')
                self.readonly = True
            else:
                # create the datasets
                self.h5 = h5py.File(self.dst, 'w')
                self.h5.create_dataset(
                    'signature', (size, features), dtype=np.float32)
                self.h5.create_dataset(
                    'applicability', (size, int(np.ceil(features / 128))),
                    dtype=np.float32)
                self.h5.create_dataset(
                    'dataset', (int(np.ceil(features / 128)),),
                    dtype=h5py.special_dtype(vlen=str))
            # expose the datasets
            self.signature = self.h5['signature']
            self.applicability = self.h5['applicability']
            self.dataset = self.h5['dataset']

    def close(self):
        if self.h5 is None:
            return
        self.h5.close()
        # leave it open for reading
        self.h5 = h5py.File(self.dst, 'r')
        # expose the datasets
        self.signature = self.h5['signature']
        self.applicability = self.h5['applicability']
        self.dataset = self.h5['dataset']
