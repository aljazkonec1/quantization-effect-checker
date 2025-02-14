# !/bin/bash

model_path="shared_with_container/yolov6n-r2-288x512.dlc"
model_name="yolov6n-base-quant"
# onnx_model_path="shared_with_container/yolov6n-r2-288x512-conv-transpose-modified.onnx"
onnx_model_path="shared_with_container/yolov6n-r2-288x512-modified.onnx"
test_img_path="data/000000135673.jpg"

python quant_checker/benchmark_model.py \
    --model_path $model_path \
    --model_name $model_name \
    --test_img_path $test_img_path \
    --onnx_model_path $onnx_model_path