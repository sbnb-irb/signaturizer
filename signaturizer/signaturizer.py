"""Signaturize molecules."""
import os
import h5py
import shutil
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import Model
from tensorflow.keras import Input

try:
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem import AllChem
except ImportError:
    raise ImportError("requires RDKit " +
                      "https://www.rdkit.org/docs/Install.html")


class Signaturizer(object):
    """Signaturizer Class.

    Loads TF-hub module, compose a single model, handle verbosity.
    """

    def __init__(self, model_name,
                 base_url="http://chemicalchecker.com/api/db/getSignaturizer/",
                 version='2020_02', local=False, verbose=False,
                 applicability=True):
        """Initialize a Signaturizer instance.

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
            if model_name == ['GLOBAL']:
                self.model_names = [y + x for y in 'ABCDE' for x in '12345']
            else:
                if 'GLOBAL' in model_name:
                    raise Exception('"GLOBAL" model can only be used alone.')
                self.model_names = model_name
        # load modules as layer to compose a new model
        main_input = Input(shape=(2048,), dtype=tf.float32, name='main_input')
        sign_output = list()
        app_output = list()
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

            sign_layer = hub.KerasLayer(url, signature='signature',
                                        trainable=False, tags=['serve'],
                                        output_key='signature',
                                        signature_outputs_as_dict=False)
            sign_output.append(sign_layer(main_input))

            if self.applicability:
                try:
                    app_layer = hub.KerasLayer(
                        url, signature='applicability',
                        trainable=False, tags=['serve'],
                        output_key='applicability',
                        signature_outputs_as_dict=False)
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
        # set rdKit verbosity
        if self.verbose:
            RDLogger.EnableLog('rdApp.*')
        else:
            RDLogger.DisableLog('rdApp.*')

    def predict(self, molecules, destination=None, keytype='SMILES',
                save_mfp=False, chunk_size=1000, batch_size=128):
        """Predict signatures for given SMILES.

        Perform signature prediction for input SMILES. We recommend that the
        list is sorted and non-redundant, but this is optional. Some input
        SMILES might be impossible to interpret, in this case, no prediction
        is possible and the corresponding signature will be set to NaN.

        Args:
            molecules(list): List of strings representing molecules. Can be
                SMILES (by default) or InChI.
            destination(str): File path where to save predictions.
            chunk_size(int): Perform prediction on chunks of this size.
            keytype(str): Wether to interpret molecules as InChI or SMILES.
        Returns:
            results: `SignaturizerResult` class. The ordering of input SMILES
                is preserved.
        """
        # input must be a list, otherwise we make it so
        if isinstance(molecules, str):
            molecules = [molecules]
        # convert input molecules to InChI
        inchies = list()
        if keytype.upper() == 'SMILES':
            for smi in molecules:
                if smi == '':
                    smi = 'INVALID SMILES'
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    if self.verbose:
                        print("Cannot get molecule from SMILES: %s." % smi)
                    inchies.append('INVALID SMILES')
                    continue
                inchi = Chem.rdinchi.MolToInchi(mol)[0]
                if self.verbose:
                    print('CONVERTED:', smi, inchi)
                inchies.append(inchi)
        else:
            inchies = molecules
        self.inchies = inchies
        # Prepare result object
        features = len(self.model_names) * 128
        results = SignaturizerResult(
            len(inchies), destination, features, save_mfp=save_mfp)
        results.dataset[:] = self.model_names
        if results.readonly:
            raise Exception(
                'Destination file already exists, ' +
                'delete or rename to proceed.')

        # predict by chunk
        all_chunks = range(0, len(inchies), chunk_size)
        for i in tqdm(all_chunks, disable=not self.verbose):
            chunk = slice(i, i + chunk_size)
            sign0s = list()
            failed = list()
            for idx, inchi in enumerate(inchies[chunk]):
                try:
                    # read molecule
                    inchi = inchi.encode('ascii', 'ignore')
                    if self.verbose:
                        # print('READING', inchi, type(inchi))
                        pass
                    mol = Chem.inchi.MolFromInchi(inchi)
                    if mol is None:
                        raise Exception(
                            "Cannot get molecule from InChI.")
                    info = {}
                    fp = AllChem.GetMorganFingerprintAsBitVect(
                        mol, 2, nBits=2048, bitInfo=info)
                    bin_s0 = [fp.GetBit(i) for i in range(
                        fp.GetNumBits())]
                    calc_s0 = np.array(bin_s0).astype(np.float32)
                except Exception as err:
                    # in case of failure save idx to fill NaNs
                    if self.verbose:
                        print("SKIPPING %s: %s" % (inchi, str(err)))
                    failed.append(idx)
                    calc_s0 = np.full((2048, ),  np.nan)
                finally:
                    sign0s.append(calc_s0)
            # stack input fingerprints and run signature predictor
            sign0s = np.vstack(sign0s)
            preds = self.model.predict(sign0s, batch_size=batch_size)
            # add NaN where SMILES conversion failed
            if failed:
                preds[np.array(failed)] = np.full(features,  np.nan)
            results.signature[chunk] = preds
            if save_mfp:
                results.mfp[chunk] = sign0s
            # run applicability predictor
            if self.applicability:
                apreds = self.app_model.predict(sign0s, batch_size=batch_size)
                if failed:
                    apreds[np.array(failed)] = np.nan
                results.applicability[chunk] = apreds
        failed = np.isnan(results.signature[:, 0])
        results.failed[:] = np.isnan(results.signature[:, 0])
        results.close()
        if self.verbose:
            print('PREDICTION complete!')
        if any(failed) > 0:
            print('Some molecules could not be recognized,'
                  ' the corresponding signatures are NaN')
            if self.verbose:
                for idx in np.argwhere(failed).flatten():
                    print(molecules[idx])
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
    """SignaturizerResult class.

    Contain result of a prediction.Results are stored in the following
    numpy vector:

        * ``signatures``: Float matrix where each row is a molecule signature.
        * ``applicability``: Float array with applicability score.
        * ``dataset``: List of bioactivity dataset used.
        * ``failed``: Mask for failed molecules.

    If a destination is specified the result are saved in an HDF5 file with
    the same vector available as HDF5 datasets.
    """

    def __init__(self, size, destination, features=128, save_mfp=False):
        """Initialize a SignaturizerResult instance.

        Args:
            size (int): The number of molecules being signaturized.
            destination (str): Path to HDF5 file where prediction results will
                be saved.
            features (int, optional): how many feature have to be stored.

        """
        self.dst = destination
        self.readonly = False
        self.save_mfp = save_mfp
        if self.dst is None:
            # simple numpy arrays
            self.h5 = None
            self.signature = np.full((size, features), np.nan, order='F',
                                     dtype=np.float32)
            self.applicability = np.full(
                (size, int(np.ceil(features / 128))), np.nan, dtype=np.float32)
            self.dataset = np.full((int(np.ceil(features / 128)),), np.nan,
                                   dtype=h5py.special_dtype(vlen=str))
            self.failed = np.full((size, ), False, dtype=np.bool)
            if self.save_mfp:
                self.mfp = np.full((size, 2048), np.nan, order='F', dtype=int)
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
                self.h5.create_dataset(
                    'failed', (size,),
                    dtype=np.bool)
                if self.save_mfp:
                    self.h5.create_dataset('mfp', (size, 2048), dtype=int)
            # expose the datasets
            self.signature = self.h5['signature']
            self.applicability = self.h5['applicability']
            self.dataset = self.h5['dataset']
            self.failed = self.h5['failed']
            if self.save_mfp:
                self.mfp = self.h5['mfp']

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
        self.failed = self.h5['failed']
        if self.save_mfp:
            self.mfp = self.h5['mfp']
