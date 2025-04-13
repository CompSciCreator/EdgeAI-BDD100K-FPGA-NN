#!/bin/bash

vai_q_tensorflow quantize \
  --input_frozen_graph frozen_graphv2.pb \
  --input_nodes=images \
  --input_shapes=?,640,640,3 \
  --output_nodes=PartitionedCall \
 --input_fn calib_input.calib_input \
  --method 1 \
  --output_dir quantized_model

