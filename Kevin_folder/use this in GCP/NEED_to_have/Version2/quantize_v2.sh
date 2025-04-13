#!/bin/bash

vai_q_tensorflow quantize \
  --input_frozen_graph frozen_graphv2.pb \
  --input_nodes=images \
  --input_shapes=?,3,640,640 \
  --output_nodes=PartitionedCall \
  --input_fn calib_input_v2.calib_input_v2 \
  --method 1 \
  --output_dir quantized_model_v2
   
