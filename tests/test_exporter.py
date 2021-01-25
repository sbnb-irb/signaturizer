import os
import shutil
import unittest
import numpy as np
from .helper import skip_if_import_exception, start_http_server

from signaturizer.exporter import export_smilespred
from signaturizer import Signaturizer


class TestExporter(unittest.TestCase):

    def setUp(self):
        # path for test data
        test_dir = os.path.dirname(os.path.realpath(__file__))
        self.data_dir = os.path.join(test_dir, 'data')
        self.tmp_dir = os.path.join(test_dir, 'tmp')
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
        os.mkdir(self.tmp_dir)
        os.mkdir(os.path.join(self.tmp_dir, 'vXXX'))
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
    def test_export(self):
        # export smilespred
        version = 'vXXX'
        module_file = 'dest_smilespred.tar.gz'
        module_destination = os.path.join(
            self.tmp_dir, version, module_file)
        tmp_path_smilespred = os.path.join(self.tmp_dir, 'export_smilespred')
        smilespred_path = os.path.join(self.data_dir, 'models', 'smiles')
        export_smilespred(smilespred_path, module_destination,
                          tmp_path=tmp_path_smilespred, clear_tmp=False)
        base_url = "http://localhost:%d/" % (self.server_port)
        module = Signaturizer(module_file, base_url=base_url, version=version)
        res = module.predict(self.test_smiles)
        pred = res.signature[:]
        ref_pred_file = os.path.join(
            self.data_dir, 'models', 'smiles_pred.npy')
        #np.save(ref_pred_file, pred)
        pred_ref = np.load(ref_pred_file)
        np.testing.assert_almost_equal(pred_ref, pred)

    @skip_if_import_exception
    def test_export_applicability(self):
        # export smilespred and applicability
        version = 'vXXX'
        module_file = 'dest_smilespred.tar.gz'
        module_destination = os.path.join(
            self.tmp_dir, version, module_file)
        tmp_path_smilespred = os.path.join(self.tmp_dir, 'export_smilespred')
        smilespred_path = os.path.join(self.data_dir, 'models', 'smiles')
        apppred_path = os.path.join(self.data_dir, 'models', 'applicability')
        export_smilespred(smilespred_path, module_destination,
                          tmp_path=tmp_path_smilespred, clear_tmp=False,
                          applicability_path=apppred_path)
        base_url = "http://localhost:%d/" % (self.server_port)
        module = Signaturizer(module_file, base_url=base_url, version=version,
                              applicability=True)
        res = module.predict(self.test_smiles)
        pred = res.signature[:]
        ref_pred_file = os.path.join(
            self.data_dir, 'models', 'smiles_pred.npy')
        #np.save(ref_pred_file, pred)
        pred_ref = np.load(ref_pred_file)
        np.testing.assert_almost_equal(pred_ref, pred)
        apred = res.applicability[:]
        ref_apred_file = os.path.join(
            self.data_dir, 'models', 'applicability_pred.npy')
        #np.save(ref_apred_file, apred)
        apred_ref = np.load(ref_apred_file)
        np.testing.assert_almost_equal(apred_ref, apred)

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
        version = 'vXXX'
        module_file = 'dest_smilespred.tar.gz'
        module_destination = os.path.join(
            self.tmp_dir, version, module_file)
        tmp_path_smilespred = os.path.join(self.tmp_dir, 'export_smilespred')
        export_smilespred(
            os.path.join(s3.model_path, 'smiles_final'),
            module_destination, tmp_path=tmp_path_smilespred, clear_tmp=False)
        # test intermediate step
        module = Signaturizer(tmp_path_smilespred, local=True)
        res = module.predict(self.test_smiles)
        pred = res.signature[:]
        np.testing.assert_almost_equal(pred_ref, pred)
        # test final step
        base_url = "http://localhost:%d/" % (self.server_port)
        module = Signaturizer(module_file, base_url=base_url, version=version)
        res = module.predict(self.test_smiles)
        pred = res.signature[:]
        np.testing.assert_almost_equal(pred_ref, pred)

