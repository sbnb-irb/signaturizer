import os
import math
import pickle
import shutil
import unittest

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
        module = Signaturizer(module_dir)
        res = module.predict(self.test_smiles)
        self.assertEqual(pred_ref.tolist(), res.signature.tolist())
        # test saving to file
        destination = os.path.join(self.tmp_dir, 'pred.h5')
        res = module.predict(self.test_smiles, destination)
        self.assertTrue(os.path.isfile(destination))
        self.assertEqual(pred_ref.tolist(), res.signature[:].tolist())
        # test prediction of invalid SMILES
        res = module.predict(['C', 'C&', 'C'])
        for comp in res.signature[0]:
            self.assertFalse(math.isnan(comp))
        for comp in res.signature[1]:
            self.assertTrue(math.isnan(comp))
        for comp in res.signature[2]:
            self.assertFalse(math.isnan(comp))

    def test_predict_multi(self):
        module_dirs = list()
        A1_path = os.path.join(self.data_dir, 'A1')
        B1_path = os.path.join(self.data_dir, 'B1')
        module_dirs.append(A1_path)
        module_dirs.append(B1_path)
        module_A1B1 = Signaturizer(module_dirs)
        res_A1B1 = module_A1B1.predict(self.test_smiles)
        self.assertEqual(res_A1B1.signature.shape[0], 2)
        self.assertEqual(res_A1B1.signature.shape[1], 128 * 2)

        module_A1 = Signaturizer(A1_path)
        res_A1 = module_A1.predict(self.test_smiles)
        self.assertEqual(res_A1B1.signature[:, :128].tolist(),
                         res_A1.signature.tolist())
        module_B1 = Signaturizer(B1_path)
        res_B1 = module_B1.predict(self.test_smiles)
        self.assertEqual(res_A1B1.signature[:, 128:].tolist(),
                         res_B1.signature.tolist())
