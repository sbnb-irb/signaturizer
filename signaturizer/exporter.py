import os
import shutil
import tempfile
import tensorflow as tf
import tensorflow_hub as hub


def export_smilespred(smilespred_path, destination,
                      tmp_path=None, clear_tmp=True):
    """Export our Keras Smiles predictor to the TF-hub module format."""
    from keras import backend as K
    from chemicalchecker.tool.smilespred import Smilespred

    if tmp_path is None:
        tmp_path = tempfile.mkdtemp()
    # save to savedmodel format
    with tf.Graph().as_default():
        smilespred = Smilespred(smilespred_path)
        smilespred.build_model(load=True)
        model = smilespred.model
        '''
        with tf.Session() as sess:
            sess.run(tf.tables_initializer())
            sess.run(tf.global_variables_initializer())
            tf.saved_model.simple_save(
                sess,
                tmp_path,
                inputs={'default': model.input},
                outputs={'default': model.output}
            )
        '''
        signature = tf.saved_model.signature_def_utils.predict_signature_def(
            inputs={'default': model.input}, outputs={'default': model.output})
        if tmp_path is None:
            tmp_path = tempfile.mkdtemp()
        builder = tf.saved_model.builder.SavedModelBuilder(tmp_path)
        builder.add_meta_graph_and_variables(
            sess=K.get_session(),
            tags=['serve'],
            signature_def_map={'serving_default': signature})
        builder.save()
    # now export savedmodel to module
    export_savedmodel(tmp_path, destination)
    # clean temporary folder
    if clear_tmp:
        shutil.rmtree(tmp_path)


def export_savedmodel(savedmodel_path, destination,
                      tmp_path=None, clear_tmp=True):
    """Export Tensorflow SavedModel to the TF-hub module format."""
    if tmp_path is None:
        tmp_path = tempfile.mkdtemp()
    # save to hub module format
    print('SAVE export_savedmodel')
    with tf.Graph().as_default():
        spec = hub.create_module_spec_from_saved_model(savedmodel_path)
        module = hub.Module(spec, tags=['serve'])
        with tf.Session() as sess:
            sess.run(tf.tables_initializer())
            sess.run(tf.global_variables_initializer())
            module.export(tmp_path, sess)
    print('DONE export_savedmodel')
    # compress the exported files to destination
    os.system("tar -cz -f %s --owner=0 --group=0 -C %s ." %
              (destination, tmp_path))
    # clean temporary folder
    if clear_tmp:
        shutil.rmtree(tmp_path)


def export_batch(cc, destination_dir, datasets=None):
    """Export all CC Smiles predictor to the TF-hub module format."""
    if datasets is None:
        datasets = cc.datasets_exemplary()
    for ds in datasets:
        s3 = cc.signature(ds, 'sign3')
        pred_path = os.path.join(s3.model_path, 'smiles_final')
        mdl_dest = os.path.join(destination_dir, ds[:2] + '.tar.gz')
        export_smilespred(pred_path, mdl_dest)