import tensorflow as tf

model = tf.saved_model.load("yolov8n_tf")
concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

frozen_func = tf.compat.v1.graph_util.convert_variables_to_constants(
    sess=tf.compat.v1.Session(graph=tf.Graph()),
    input_graph_def=concrete_func.graph.as_graph_def(),
    output_node_names=[n.name.split(":")[0] for n in concrete_func.outputs]
)

tf.io.write_graph(graph_or_graph_def=frozen_func,
                  logdir=".",
                  name="frozen_graph.pb",
                  as_text=False)
