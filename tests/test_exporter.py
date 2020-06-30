import os
import time
import shutil
import unittest
from .helper import skip_if_import_exception, start_http_server

from signaturizer.exporter import export_smilespred, export_savedmodel
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
        self.cwd = os.getcwd()
        os.chdir(self.tmp_dir)
        self.server_port = start_http_server()
        self.test_smiles = [
            # Erlotinib
            'COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC',
            # Diphenhydramine
            'CN(C)CCOC(C1=CC=CC=C1)C2=CC=CC=C2'
        ]

    def tearDown(self):
        os.chdir(self.cwd)
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
            pass

    @skip_if_import_exception
    def test_export_consistency(self):
        """Compare the exported module to the original SMILES predictor.

        N.B. This test is working only with a valid CC instance available.
        """
        from chemicalchecker import ChemicalChecker
        from chemicalchecker.core.signature_data import DataSignature

        # load CC instance and smiles prediction model
        cc = ChemicalChecker()
        s3 = cc.signature('B1.001', 'sign3')
        tmp_pred_ref = os.path.join(self.tmp_dir, 'tmp.h5')
        s3.predict_from_smiles(self.test_smiles, tmp_pred_ref)
        pred_ref = DataSignature(tmp_pred_ref)[:]

        # export smilespred
        module_file = 'dest_smilespred.tar.gz'
        module_destination = os.path.join(
            self.tmp_dir, module_file)
        tmp_path_smilespred = os.path.join(self.tmp_dir, 'export_smilespred')
        export_smilespred(
            os.path.join(s3.model_path, 'smiles_final'),
            module_destination, tmp_path=tmp_path_smilespred, clear_tmp=False)
        # test intermediate step
        module = Signaturizer(tmp_path_smilespred)
        res = module.predict(self.test_smiles)
        pred = res.signature[:]
        self.assertEqual(pred_ref.tolist(), pred.tolist())
        # test final step
        base_url = "http://localhost:%d/" % (self.server_port)
        module = Signaturizer(module_file, base_url=base_url)
        res = module.predict(self.test_smiles)
        pred = res.signature[:]
        self.assertEqual(pred_ref.tolist(), pred.tolist())

        # export savedmodel
        module_destination = os.path.join(
            self.tmp_dir, 'dest_savedmodel.tar.gz')
        tmp_path_savedmodel = os.path.join(self.tmp_dir, 'export_savedmodel')
        export_savedmodel(
            tmp_path_smilespred, module_destination,
            tmp_path=tmp_path_savedmodel, clear_tmp=False)
        # test intermediate step
        module = Signaturizer(tmp_path_savedmodel)
        res = module.predict(self.test_smiles)
        pred = res.signature[:]
        self.assertEqual(pred_ref.tolist(), pred.tolist())
        # test final step
        module = Signaturizer(module_file, base_url=base_url)
        res = module.predict(self.test_smiles)
        pred = res.signature[:]
        self.assertEqual(pred_ref.tolist(), pred.tolist())
