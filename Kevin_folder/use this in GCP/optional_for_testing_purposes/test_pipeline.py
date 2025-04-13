import tensorflow as tf
from calib_input import calib_input

# Check frozen graph input/output nodes
def check_graph_nodes(pb_file):
    with tf.io.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    has_input = False
    has_output = False

    for node in graph_def.node:
        if node.name == "images":
            has_input = True
            print("Found input node: images")
        if node.name == "PartitionedCall":
            has_output = True
            print("Found output node: PartitionedCall")

    if not has_input:
        print("Missing input node 'images'")
    if not has_output:
        print("Missing output node 'PartitionedCall'")

# Check calibration input data
def check_calib_samples():
    try:
        gen = calib_input()
        for i, sample in enumerate(gen):
            shape = sample["images"].shape
            print(f"Sample {i+1} shape: {shape}")
            if i == 4:
                break
    except Exception as e:
        print(f"Calibration input failed: {e}")

# Run all checks
if __name__ == "__main__":
    print("Checking frozen_graphv2.pb nodes:")
    check_graph_nodes("frozen_graphv2.pb")
    print("\nChecking calibration samples:")
    check_calib_samples()

