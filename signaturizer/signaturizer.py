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
                 version='2021_07', local=False, verbose=False,
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
        if version == '2019_01':
            sign_signature = 'serving_default'
            sing_output_key = 'default'
            app_signature = 'applicability'
            app_output_key = 'default'
        else:
            sign_signature = 'signature'
            sing_output_key = 'signature'
            app_signature = 'applicability'
            app_output_key = 'applicability'
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

            sign_layer = hub.KerasLayer(url, signature=sign_signature,
                                        trainable=False, tags=['serve'],
                                        output_key=sing_output_key,
                                        signature_outputs_as_dict=False)
            sign_output.append(sign_layer(main_input))

            if self.applicability:
                try:
                    app_layer = hub.KerasLayer(
                        url, signature=app_signature,
                        trainable=False, tags=['serve'],
                        output_key=app_output_key,
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

    def _smiles_to_mol(self, molecules, keys, drop_invalid=True):
        mol_objects = list()
        valid_keys = list()
        for smi, key in tqdm(zip(molecules, keys), desc='Parsing SMILES'):
            if smi == '':
                smi = 'INVALID SMILES'
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                if self.verbose:
                    print("Cannot get molecule from SMILES: %s." % smi)
                if drop_invalid:
                    continue
            valid_keys.append(key)
            mol_objects.append(mol)
        return mol_objects, valid_keys

    def _inchi_to_mol(self, molecules, keys, drop_invalid=True):
        mol_objects = list()
        valid_keys = list()
        for inchi, key in tqdm(zip(molecules, keys), desc='Parsing InChI'):
            inchi = inchi.encode('ascii', 'ignore')
            if inchi == '':
                inchi = 'INVALID InChI'
            mol = Chem.MolFromInchi(inchi)
            if mol is None:
                if self.verbose:
                    print("Cannot get molecule from InChI: %s." % inchi)
                if drop_invalid:
                    continue
            valid_keys.append(key)
            mol_objects.append(mol)
        return mol_objects, valid_keys


    def predict(self, molecules, destination=None, molecule_fmt='SMILES',
                keys=None, save_mfp=False, drop_invalid=True,
                batch_size=128, chunk_size=32,
                compression=None,  y_scramble=False,):
        """Predict signatures for given molecules.

        Perform signature prediction for input SMILES. We recommend that the
        list is sorted and non-redundant, but this is optional. Some input
        SMILES might be impossible to interpret, in this case, no prediction
        is possible and the corresponding signature will be set to NaN.

        Args:
            molecules(list): List of strings representing molecules. 
                Can be SMILES (by default), InChI or RDKIT molecule objects.
            destination(str): File path where to save predictions. If file 
                exists already it will throw an exception.
            molecule_fmt(str): Molecule format, whether to interpret molecules 
                as InChI, SMILES or RDKIT.
            keys(list): A list of keys that will be saved along with the
                predictions.
            save_mfp(bool): Set to True to save an additional matrix with
                classical Morgan Fingerprint ECFP4.
            drop_invalid(bool): Wether to drop invalid molecules i.e. molecules
                that cannot be interpreted with RdKit.
            batch_size(int): Batch size for prediction.
            chunk_size(int): Chunk size for the reulting HDF5.
            compression(str): Compression used for storing the HDF5.
            y_scramble(bool): Validation test scrambling the MFP before
                prediction.
        Returns:
            results: `SignaturizerResult` class. The ordering of input SMILES
                is preserved.
        """
        # input must be a list, otherwise we make it so
        if isinstance(molecules, str):
            molecules = [molecules]
        # if keys are not specified just use incremental numbers
        if keys is None:
            keys = [str(x) for x in range(len(molecules))]

        # convert input molecules to molecule object
        if molecule_fmt.upper() == 'SMILES':
            molecules, keys = self._smiles_to_mol(
                molecules, keys=keys, drop_invalid=drop_invalid)
        elif molecule_fmt.upper() == 'INCHI':
            molecules, keys = self._inchi_to_mol(
                molecules, keys=keys, drop_invalid=drop_invalid)
        elif molecule_fmt.upper() == 'RDKIT':
            molecules, keys = molecules, keys
        else:
            raise Exception('Unsupported molecule format `%s`' % molecule_fmt)

        # prepare result object
        features = len(self.model_names) * 128
        chunk_size = min(chunk_size, len(molecules))
        results = SignaturizerResult(
            len(molecules), destination, features, save_mfp=save_mfp,
            chunk_size=chunk_size, compression=compression, keys=keys)
        results.dataset[:] = self.model_names
        if results.readonly:
            raise Exception(
                'Destination file already exists, ' +
                'delete or rename to proceed.')

        # predict by chunk
        all_chunks = range(0, len(molecules), batch_size)
        for i in tqdm(all_chunks, desc='Generating signatures'):
            chunk = slice(i, i + batch_size)
            # prepare predictor input
            sign0s = list()
            failed = list()
            for idx, mol in enumerate(molecules[chunk]):
                try:
                    info = {}
                    fp = AllChem.GetMorganFingerprintAsBitVect(
                        mol, 2, nBits=2048, bitInfo=info)
                    bin_s0 = [fp.GetBit(i) for i in range(
                        fp.GetNumBits())]
                    calc_s0 = np.array(bin_s0).astype(np.float32)
                except Exception as err:
                    # in case of failure save idx to fill NaNs
                    if self.verbose:
                        print("FAILED %s: %s" % (idx, str(err)))
                    failed.append(idx)
                    calc_s0 = np.full((2048, ),  np.nan)
                finally:
                    sign0s.append(calc_s0)
            # stack input fingerprints and run signature predictor
            sign0s = np.vstack(sign0s)
            if y_scramble:
                y_shuffle = np.arange(sign0s.shape[1])
                np.random.shuffle(y_shuffle)
                sign0s = sign0s[:, y_shuffle]
            # run prediction
            preds = self.model.predict(
                tf.convert_to_tensor(sign0s, dtype=tf.float32),
                batch_size=batch_size)
            # add NaN where conversion failed
            if failed:
                preds[np.array(failed)] = np.full(features,  np.nan)
            results.signature[chunk] = preds
            if save_mfp:
                results.mfp[chunk] = sign0s
            # run applicability predictor
            if self.applicability:
                apreds = self.app_model.predict(
                    tf.convert_to_tensor(sign0s, dtype=tf.float32),
                    batch_size=batch_size)
                if failed:
                    apreds[np.array(failed)] = np.nan
                results.applicability[chunk] = apreds
        failed = np.isnan(results.signature[:, 0])
        results.failed[:] = np.isnan(results.signature[:, 0])
        results.keys[:] = keys
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

    def __init__(self, size, destination, features=128, compression='gzip',
                 chunk_size=32, save_mfp=False, keys=None):
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
            self.keys = np.full((size, ), np.nan, order='F',
                                     dtype=str)
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
                self.h5.create_dataset('keys', (size,),
                                       dtype=h5py.string_dtype())
                self.h5.create_dataset('signature', (size, features),
                                       dtype=np.float32,
                                       chunks=(chunk_size, features),
                                       compression=compression)
                app_dim = int(np.ceil(features / 128))
                self.h5.create_dataset('applicability', (size, app_dim),
                                       dtype=np.float32)
                ds_dim = int(np.ceil(features / 128))
                self.h5.create_dataset('dataset', (ds_dim,),
                                       dtype=h5py.string_dtype())
                self.h5.create_dataset('failed', (size,), dtype=np.bool)
                if self.save_mfp:
                    self.h5.create_dataset('mfp', (size, 2048), dtype=int,
                                           chunks=(chunk_size, 2048),
                                           compression=compression)
            # expose the datasets
            self.keys = self.h5['keys']
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
