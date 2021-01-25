import os
import math
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
        self.tautomer_smiles = ['CC(O)=Nc1ccc(O)cc1', 'CC(=O)NC1=CC=C(C=C1)O']
        self.inchi = [
            'InChI=1S/C8H9NO2/c1-6(10)9-7-2-4-8(11)5-3-7/h2-5,11H,1H3,(H,9,10)'
        ]

    def tearDown(self):
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir, ignore_errors=True)
            pass

    def test_predict(self):
        # load reference predictions
        ref_file = os.path.join(self.data_dir, 'pred.npy')
        pred_ref = np.load(open(ref_file, 'rb'))
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
        self.assertEqual(len(res.applicability[:]), 2)
        self.assertFalse(np.isnan(res.applicability[0]))
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
        self.assertTrue(all(res.failed == [False, True, False]))

    def test_predict_global_remote(self):
        module = Signaturizer(['GLOBAL'])
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

    def test_tautomers(self):
        module = Signaturizer('A1')
        res = module.predict(self.tautomer_smiles)
        self.assertTrue(all(res.signature[0] == res.signature[1]))

    def test_inchi(self):
        module = Signaturizer('A1')
        res_inchi = module.predict(self.inchi, keytype='InChI')
        res_smiles = module.predict([self.tautomer_smiles[0]])
        self.assertTrue(all(res_inchi.signature[0] == res_smiles.signature[0]))

    def test_all_single(self):
        module = Signaturizer('A1')
        res_all = module.predict(self.test_smiles)
        for idx, smile in enumerate(self.test_smiles):
            res_single = module.predict([smile])
            np.testing.assert_almost_equal(res_all.signature[idx],
                                           res_single.signature[0])
