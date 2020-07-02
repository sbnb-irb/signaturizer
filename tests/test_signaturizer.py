import os
import math
import pickle
import shutil
import unittest
import numpy as np

from signaturizer import Signaturizer


class TestSignaturizer(unittest.TestCase):

    def setUp(self):
        # path for test data
        test_dir = os.path.dirname(os.path.realpath(__file__))
        self.data_dir = os.path.join(test_dir, 'data')
        self.tmp_dir = os.path.join(test_dir, 'tmp')
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
        os.mkdir(self.tmp_dir)
        self.test_smiles = [
            # Erlotinib
            'COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC',
            # Diphenhydramine
            'CN(C)CCOC(C1=CC=CC=C1)C2=CC=CC=C2'
        ]
        self.invalid_smiles = ['C', 'C&', 'C']

    def tearDown(self):
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
            pass

    def test_predict(self):
        # load reference predictions
        ref_file = os.path.join(self.data_dir, 'pred.pkl')
        pred_ref = pickle.load(open(ref_file, 'rb'))
        # load module and predict
        module_dir = os.path.join(self.data_dir, 'B1')
        module = Signaturizer(module_dir, local=True)
        res = module.predict(self.test_smiles)
        np.testing.assert_almost_equal(pred_ref, res.signature[:])
        # test saving to file
        destination = os.path.join(self.tmp_dir, 'pred.h5')
        res = module.predict(self.test_smiles, destination)
        self.assertTrue(os.path.isfile(destination))
        np.testing.assert_almost_equal(pred_ref, res.signature[:])
        # test prediction of invalid SMILES
        res = module.predict(self.invalid_smiles)
        for comp in res.signature[0]:
            self.assertFalse(math.isnan(comp))
        for comp in res.signature[1]:
            self.assertTrue(math.isnan(comp))
        for comp in res.signature[2]:
            self.assertFalse(math.isnan(comp))

    def test_predict_multi(self):
        # load multiple model and check that results stacked correctly
        module_dirs = list()
        A1_path = os.path.join(self.data_dir, 'A1')
        B1_path = os.path.join(self.data_dir, 'B1')
        module_dirs.append(A1_path)
        module_dirs.append(B1_path)
        module_A1B1 = Signaturizer(module_dirs, local=True)
        res_A1B1 = module_A1B1.predict(self.test_smiles)
        self.assertEqual(res_A1B1.signature.shape[0], 2)
        self.assertEqual(res_A1B1.signature.shape[1], 128 * 2)

        module_A1 = Signaturizer(A1_path, local=True)
        res_A1 = module_A1.predict(self.test_smiles)
        np.testing.assert_almost_equal(res_A1B1.signature[:, :128],
                                       res_A1.signature)

        module_B1 = Signaturizer(B1_path, local=True)
        res_B1 = module_B1.predict(self.test_smiles)
        np.testing.assert_almost_equal(res_A1B1.signature[:, 128:],
                                       res_B1.signature)

        res = module_A1B1.predict(self.invalid_smiles)
        for comp in res.signature[0]:
            self.assertFalse(math.isnan(comp))
        for comp in res.signature[1]:
            self.assertTrue(math.isnan(comp))
        for comp in res.signature[2]:
            self.assertFalse(math.isnan(comp))

    def test_predict_global_remote(self):
        module = Signaturizer('GLOBAL')
        res = module.predict(self.test_smiles)
        self.assertEqual(res.signature.shape[0], 2)
        self.assertEqual(res.signature.shape[1], 128 * 25)

    def test_overwrite(self):
        module_dir = os.path.join(self.data_dir, 'B1')
        module = Signaturizer(module_dir, local=True)
        destination = os.path.join(self.tmp_dir, 'pred.h5')
        module.predict(self.test_smiles, destination)
        # repeating writing will result in an exception
        with self.assertRaises(Exception):
            module.predict(self.test_smiles, destination)
