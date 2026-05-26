import tensorflow as tf

# implement physical two-device split that auto_partition.py motivates analytically
# Produces three SavedModel artifacts implementing a two-stage heterogeneous pipeline
# Full pipeline (DLC_ma_p1/, targets CPU): complete model, baseline
# Stage 1: (DLC_ma_sub_p1/, targets Coral TPU): input -> heatmap + locref
#   Functionally identical to gen_tflite.py's extraction, with two differences
#   Full graph is loaded under namespace 'DLC'; pruned subgraph is loaded without a prefix,
#   so tensor names revert to 'pose/...'
#
#   No explicit shape fixation; original dynamic Placeholder is used
#
#   This artifact is what gets quantized to int8 and compiled by edgetpu_compiler

# Stage 2: (DLC_ma_p2/, targets CPU): heatmap + locref -> concat_1
#

tfv1 = tf.compat.v1
gdef = tfv1.GraphDef()
with tfv1.io.gfile.GFile("snapshot-700000.pb","rb") as f:  gdef.ParseFromString(f.read())
g = tf.Graph()
g2 = tf.Graph()
with g.as_default():tf.graph_util.import_graph_def(gdef, name="DLC")
input_tensor_name = str(g.get_operations()[0].name) + ":0"
input_tensor = g.get_tensor_by_name(input_tensor_name)
input_tensor
part_pred_output_name = 'DLC/pose/part_pred/block4/BiasAdd:0'
part_pred_output_tensor = g.get_tensor_by_name(part_pred_output_name)
locref_pred_output_name = 'DLC/pose/locref_pred/block4/BiasAdd:0'
locref_pred_output_name = g.get_tensor_by_name(locref_pred_output_name)

with tfv1.Session(graph=g) as s:
    tfv1.saved_model.simple_save(session=s, export_dir='DLC_ma_p1/', inputs={'input':input_tensor}, outputs={'part_pred_output': part_pred_output_tensor, 'locref_pred_output': locref_pred_output_name})
gdef_sub = tfv1.graph_util.extract_sub_graph(gdef, ['pose/part_pred/block4/BiasAdd', 'pose/locref_pred/block4/BiasAdd'])

with g2.as_default():
    tf.graph_util.import_graph_def(gdef_sub, name='')
    
g2.get_operations()[0]
g2.get_operations()[-1]
g2.get_operations()[-2]
g2_input=g2.get_tensor_by_name('Placeholder:0')
g2_part_pred_output=g2.get_tensor_by_name('pose/part_pred/block4/BiasAdd:0')
g2_locref_pred_output=g2.get_tensor_by_name('pose/locref_pred/block4/BiasAdd:0')

with tfv1.Session(graph=g2) as s2:
    tfv1.saved_model.simple_save(
        session=s2, 
        export_dir='DLC_ma_sub_p1/', 
        inputs={'input':g2_input}, 
        outputs={'part_pred_output': g2_part_pred_output, 'locref_pred_output': g2_locref_pred_output}
    )

output_tensor = g.get_tensor_by_name('DLC/concat_1:0') # final merge node

with tfv1.Session(graph=g) as s:
    tfv1.saved_model.simple_save(
        session=s, 
        export_dir='DLC_ma_p2/', 
        inputs={'part_pred_input': part_pred_output_tensor, 'locref_pred_input': locref_pred_output_name}, 
        outputs={'output': output_tensor}
    )
