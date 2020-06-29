import os
import shutil
import unittest
from helper import skip_if_import_exception

from signaturizer import Signaturizer


class TestSignaturizer(unittest.TestCase):

    def setUp(self):
        # path for test data
        test_dir = os.path.dirname(os.path.realpath(__file__))
        self.data_dir = os.path.join(test_dir, 'data')
        self.tmp_dir = os.path.join(test_dir, 'tmp')
        os.environ["CC_CONFIG"] = os.path.join(self.data_dir, 'config.json')

    def tearDown(self):
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    @skip_if_import_exception
    def test_export_consistency(self):

        from chemicalchecker import ChemicalChecker
        from chemicalchecker.core.signature_data import DataSignature
        test_smiles = ['CCC', 'C']

        cc = ChemicalChecker()
        s3 = cc.signature('B1.001', 'sign3')
        tmp_pred_ref = os.path.join(self.tmp_dir, 'tmp.h5')
        s3.predict_from_smiles(test_smiles, tmp_pred_ref)
        pred_ref = DataSignature(tmp_pred_ref)

        # export smilespred
        module_destination = os.path.join(self.tmp_dir, 'dest_smilespred')
        tmp_path = os.path.join(self.tmp_dir, 'export_smilespred')
        Signaturizer._export_smilespred_as_module(
            os.path.join(s3.module_path, 'smiles_final'),
            module_destination, tmp_path=tmp_path)
        # test intermediate step
        module = Signaturizer(tmp_path, compressed=False, local=True)
        pred = module.predict(test_smiles)
        self.assertEqual(pred_ref, pred)
        # test final step
        module = Signaturizer(module_destination, compressed=True, local=True)
        pred = module.predict(test_smiles)
        self.assertEqual(pred_ref, pred)

        # export savedmodel
        module_destination = os.path.join(self.tmp_dir, 'dest_savedmodel')
        tmp_path = os.path.join(self.tmp_dir, 'export_savedmodel')
        Signaturizer._export_savedmodel_as_module(
            os.path.join(s3.module_path, 'smiles_final'),
            module_destination, tmp_path=tmp_path)
        # test intermediate step
        module = Signaturizer(tmp_path, compressed=False, local=True)
        pred = module.predict(test_smiles)
        self.assertEqual(pred_ref, pred)
        # test final step
        module = Signaturizer(module_destination, compressed=True, local=True)
        pred = module.predict(test_smiles)
        self.assertEqual(pred_ref, pred)
