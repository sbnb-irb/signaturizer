import os
import shutil
import tempfile
import tensorflow as tf


class SignaturizerModule(tf.Module):

    def __init__(self, signature_mdl):
        self.signature_mdl = signature_mdl

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 2048), dtype=tf.float32)])
    def signature(self, mfp):
        results = self.signature_mdl(mfp)
        return {"signature": results}


class SignaturizerApplicabilityModule(tf.Module):

    def __init__(self, signature_mdl, applicability_mdl):
        self.signature_mdl = signature_mdl
        self.applicability_mdl = applicability_mdl

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 2048), dtype=tf.float32)])
    def signature(self, mfp):
        results = self.signature_mdl(mfp)
        return {"signature": results}

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 2048), dtype=tf.float32)])
    def applicability(self, mfp):
        results = self.applicability_mdl(mfp)
        return {"applicability": results}


def export_smilespred(smilespred_path, destination,
                      tmp_path=None, clear_tmp=True, compress=True,
                      applicability_path=None):
    """Export our Keras Smiles predictor to the TF-hub module format."""
    from chemicalchecker.tool.smilespred import Smilespred
    from chemicalchecker.tool.smilespred import ApplicabilityPredictor

    if tmp_path is None:
        tmp_path = tempfile.mkdtemp()
    # load models
    smilespred = Smilespred(smilespred_path)
    smilespred.build_model(load=True)
    model = smilespred.model
    # save simple modelor combined with applicability
    if applicability_path is None:
        full_mdl = SignaturizerModule(model)
        tf.saved_model.save(full_mdl, tmp_path,
                            signatures={
                                "signature": full_mdl.signature})
    else:
        app_pred = ApplicabilityPredictor(applicability_path)
        app_pred.build_model(load=True)
        app_model = app_pred.model
        # combine in a module and save to savedmodel format
        full_mdl = SignaturizerApplicabilityModule(model, app_model)
        tf.saved_model.save(full_mdl, tmp_path,
                            signatures={
                                "signature": full_mdl.signature,
                                "applicability": full_mdl.applicability})
    # now export savedmodel to tfhub module
    if compress:
        # compress the exported files to destination
        os.system("tar -cz -f %s --owner=0 --group=0 -C %s ." %
                  (destination, tmp_path))
    else:
        shutil.copytree(tmp_path, destination)
    # clean temporary folder
    if clear_tmp:
        shutil.rmtree(tmp_path)


def export_batch(cc, destination_dir, datasets=None, applicability=True):
    """Export all CC Smiles predictor to the TF-hub module format."""
    if datasets is None:
        datasets = cc.datasets_exemplary()
    for ds in datasets:
        s3 = cc.signature(ds, 'sign3')
        pred_path = os.path.join(s3.model_path, 'smiles_final')
        mdl_dest = os.path.join(destination_dir, ds[:2])
        if applicability:
            apred_path = os.path.join(
                s3.model_path, 'smiles_applicability_final')
            export_smilespred(pred_path, mdl_dest, compress=False,
                              applicability_path=apred_path)
            export_smilespred(pred_path, mdl_dest + '.tar.gz',
                              applicability_path=apred_path)
        else:
            export_smilespred(pred_path, mdl_dest, compress=False)
            export_smilespred(pred_path, mdl_dest + '.tar.gz')
