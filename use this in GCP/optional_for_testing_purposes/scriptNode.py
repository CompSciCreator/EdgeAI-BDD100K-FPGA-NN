import tensorflow as tf

with tf.io.gfile.GFile("frozen_graphv2.pb", "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

for node in graph_def.node:
    print(node.name)

