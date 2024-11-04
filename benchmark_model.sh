# !/bin/bash

model_path="shared_with_container/outputs/yolov6n-base-quant/yolov6n-r2-288x512.dlc"
model_name="yolov6n-base-quant"
test_data_path="data/test_raw"
input_bit_width=int8 #implicit

python benchmark_model.py \
    --model_path $model_path \
    --model_name $model_name \
    --test_data_path $test_data_path \
    --input_bit_width $input_bit_width