## quantization-effect-checker
Some helpful files for checking and comparing the effect that quantization has on model performance. Intented for use with Luxonis repos.
This is in early version, need to add automatic running of whole procedure.


The checker uses COCO style annotated dataset so you need to create a postprocess function that transforms detections into compatible type. Currently I made it only for COCO val2017 and only for yolov6n-288x512. 
It should work with other yolos and with minimal adjustments for other models.
## Environment
For model conversion I used local docker installation of [modelconverter](https://github.com/luxonis/modelconverter/tree/main).

There is some kind of bug with postporcessing function that is loaded from luxonis-train on python 3.10. I just created two environments, *snpe* with python v3.10  based on [Qualcomm setup](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-2/setup.html) that is used for running *to_raw* and *benchmark_model*. THe second one is python 3.12 for running *get_detections, create_statistics, visualize_detections* built from *requirements.txt*

## Repo structure

```python
├── snpe # snpe dir from instalation 
├── data #download data into here
├── raw_outputs # raw output files copied from device
├── models_dlc # each dlc model is stored in its own dir and should contain
                # dlc, info.json, throughput.txt and model-graph.txt
    ├── <model-one>
        ├── model-one.dlc
        ├── info.json
        ├── throughput.txt
        ├── model-one-graph.txt
├── models_onnx #.onnx versions of the dlc models. These are the base comparisons for the same mentioned model
├── quant_checker # all the files to run benchmarking of a model
    ├── <model-one.onnx>
├── results
   ├── similarities # csv files, each containing similarity scores for a model
   ├── speeds # csv files, each conaining num cycles for each layer for a model
├── shared_with_container # for model converter
├── benchmark_model.py # a CLI tool to quickly get all benchmarks of the model
├── per_layer_profiler.py # Creates a csv containing average execution times per layer
├── create_statistics.py # creates csv with perf metrics (reads results dir)
├── get_detections.py # creates jsons in results from models_dlc and models_onnx
├── make_dlc_info_json.py #creates the info.json file that is in each dlc model dir
├── postprocess_functions.py # funcions for postprocessing model outputs 
                            # (same function for both onnx and dlc)
├── to_raw.py # conversion and preprocess of RGB images to raw format for upload to device
├── utils.py # utils
├── visualize_detections.py # visualizes predictions of all results in results dir
├── visualize_similarities.py # to create ploty plots for all similaritie csv files in /results/similarities
```
In addition to the repo setup you need ssh access to a RVC4 device. 
To store your data make a dir in */data* directory, I made a */data/tests/* dir and will refer to it as home dir on the device.

## How to use

For ease of use I made *benchmark_model.py*, a CLI tool for faster execution of full benchmark process. The arguments you need to specify are model_path, model_name, onnx_model_path, test_img_path, *optional test_data_path*:
```python
   # !/bin/bash

   model_path="shared_with_container/yolov6n-r2-288x512.dlc"
   model_name="yolov6n-base-quant"
   onnx_model_path="shared_with_container/yolov6n-r2-288x512-modified.onnx"
   test_img_path="data/000000135673.jpg"

   python quant_checker/benchmark_model.py \
      --model_path $model_path \
      --model_name $model_name \
      --test_img_path $test_img_path \
      --onnx_model_path $onnx_model_path
```

:exclamation: **Important: make sure you select the correct onnx file! Dont select the one you specified to model converter as that one potentially doesnt have any normalization/ layer fusions. Use the onnx file that is in at intermediate_outputs/*model_name*-modified.onnx**

**Before running *benchmark_model.py:***
   1) Edit .env file to have correct credentials for ssh access to RVC4 device
   2) Get dataset and split it into test and quantization datasets. Save test to *data* dir and quantization dataset to *shared_with_container/calibration_data*
   3) Use *to_raw.py* to convert test data into correct dataformat (ie. uint8, fp16)
   4) Get onnx model and put it in *shared_with_container/models* and into *model_onnx*
   5) Create config yaml file in *shared_with_container/configs* for the model
   6) Use modelconverter to convert to dlc
   
After this you can just select an image and the script will handle the rest. By default, the image is resized to ONNX models input size and saved as a raw file in FP32 bitwidth. SNPE should be smart enought to cast fp32 to int8 while running the img through quantized model

**What running benchmark_model.py does:**
   1) Make a *model_name* dir in *models_dlc* and copy the created model.dlc to  *models_dlc/model_name/model_name.dlc*.
   2) Make a *model_name* dir in *models_onnx* and copy the onnx model to it. 
   3) Create a modified onnx model that outputs on every layer and run inference using it. Store the outputs as raw files in models_onnx/*model_name*/output. Create raw file to send to device
   4) Copy data in *test_data_path* and *model_name.dlc* to the RVC4 device
   5) Now that the model and the data are on device it will run predictions and benchmark the inference speed. This is done with the commands:
      1) Inference speed: 
   ```snpe-throughput-net-run --duration 60 --use_dsp --container <<model_name>> --perf_profile balanced > throughput.txt ``` 
      2) Run inference over test image:
    ```snpe-net-run --container <<model.dlc>> --input_list test_raw/inputs_raw.txt--use_native_input_files --use_native_output_files```
   6)  The throughput and model outputs are copied back to host. *throughput.txt* is copied to *models_dlc/model_name/throughput.txt* while the model outputs are copied to *raw_outputs/raw_outputs_model_name*.
   7)  Next, a model graph is created and saved as a txt file in *models_dlc/model_name/dlc-info-graph.txt*. This is done with:
   ```snpe-dlc-info -i models_dlc/model_name/model_name.dlc > models_dlc/model_name/dlc-info-graph.txt``` 
   The graph contains layer names, operations and quantization information per each layer of the model. For testing only quantized model performance we are mostly interested in the last cuple of lines where the output quantization information is stored. 
   8)  A per layer analysis performed with the command:
   ```snpe-diagview --input_log results_log_file.log,--csv_format_version 2 --output models_dlc/model_name/layer_stats.csv```
   and then processed with *per_layer_profiler.py*
   9)  This information is then, in conjunction with the *throughputs.txt* file from step *3.i*, extracted and saved into a *info.json* file with the use of *make_dlc_info_json.py* script.

**After running the script :** 
After running this for as may models as you like, the results/similarities and optionally results/speeds (if you used extra command). You can visualize similarities with *visualize_similarities.py*

## Protips

Here are some protips to help you debug your problem faster:
* For easier use first convert multiple models and then create a *benchmark_models.sh* script file that runs benchmark_models multiple times. 
* For onnx models makes sure you input the image in the correct format (either BGR or RGB), the get_detections/get_segmentations assumes RGB model input. Switching the input will usually still produce good results, a couple of % lower and caould be hard to spot.
* A good way to test if you setup the buffers correctly is, when you run the `snpe-net-run` command it should print out only one image name per QNN operation, if you have it set wrong it will print two or more images per QNN operation.
* Onnx models accept images in NCHW format while dlc models accept in NHWC format so make sure you convert it correctly. In addition to this, the dlc models outputs are also in a different shape (you can check it in the dlc-model-graph.txt file). For example in yolov6n the onnx model outputs are of shape (1, 85, 18, 32) but dlc model returns in (1, 18, 32, 85) so keep that in mind.

## Quantization flags
There are multiple additional flags available for quantization, here I just list a few. For more info on them check [dlc-quant-info](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-2/tools.html#snpe-dlc-quant) and [quantized vs unquantized models](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-2/quantized_models.html) from Qualcomm.

- `--use_per_channel_quantization` selects per-channel quantization, only works for conv, deconv, and FC loperations, for all other regular per-layer quant is used
- `--act_bitwidth` bitwidth for activation layers (either 8 or 16 bit)
- `--bias_bitwidth` bitwidth for biases
- `--act_quantizer` indicates what quantizer to use, can choose between `tf`, `enhanced` and `symmetric`. You can check the docs for more info

## Findings
This is a list of findings on what works/ doesn't work when quantizing and optimizing models:
  1.  `--per-channel-quantization` works ONLY on Conv, Deconv and Fully connected layers, while all other layers use the default `--per-layer-quantization`. Per channel works by creating quantization values for each conv kernel seperately, since Convolution is highly optimized with dedicated compute cores, the inference speed decrease is < 1 FPS. In addition, per-channel-quantization improves model accuracy to be on par with the original (unquantized) model so it should always be used.
  2.  Most models expect an RGB input and modelconverter added a split and a concat layer at the begining of the onnx model, this is highly costly as the concat layer copies each value to a new memory location and uses up a large portion of the inference time (20+ % of the model). This problem is exacerbated for larger input models as the number of input pixels increase exponentionally with height/width. To avoid using split/concat there are multiple options. Depthai can just request an RGB image instead of BGR, or if the models first layer is Conv we simply transpose the kernel weights by the channel.
  3.  Division in SNPE is implemented very badly, at the minimum, use Multiplication layer instead. If the division is adjacent to a Conv layer, you can multiply the kernel weights by the division values. This can be easily done to the onnx model before converting to dlc.
  4.  Substitution/Addition layers are also slow as they are element-wise operations. Again you can fuse the layer into the bias of a conv layer, you just have to make sure you multiply the Add/Sub values by the per-channel sum of kernel weights if the add/sub is before the Conv layer. This is also done to the onnx model before converting to dlc.
  5.  SNPE can also run models saved in FP16 format with no quantization, this model should have no accuracy drop compared to base onnx. But sometimes fp16 model has lower FPS. If the model has Concat, Mul, Sub, Div layers, they are around 2x slower than quantized versions of the layers. This method should be reservered only for models that are very hard to quantize.
  6.  If a convolutional layer has a not uniform stride (not [1, 1], [2, 2] or [3, 3]) then the perforace will be terrible. The fix is to update the onnx graph by changing the stride to [1, 1] and using *Slice node* to take every other value that the convolutional layer returns. I made a sample script on how this can be done in *onnx_modifications.py* in the method *stride_removal*.
  7. A sequence of layers like `add/sub -> mul -> conv`, `mul -> add/sub -> conv` and `conv -> mul -> add/sub` can be fused into the conv layer without any accuarcy loss while decreasing inference time. A sample script is in *onnx_modifications.py* in the *fuse_add_mul_into_conv* method.
  8. On the DSP, matrix multiplications (e.g. fully connected layers) run very slow and are not optimized. Even though we do not have access to the GPU, I was able to run `snpe-throughput-net-run` with `--use_gpu_fp16` flag to benchmark a model with MatMul Layers. On DSP the model has 7 FPS, on GPU the model has 600 FPS. So, if possible, when choosing a model to use, please avoid ones that have lots of MatMul or Gemm layers. 

## Grand idea
So this is still a very early project (if you can even call it that). But the end goal is to better understand how quantization affects models and have a data-based approach to measuring model performance, inference speed and finding problematic layers in the model.
* I need to better explore mixed quantization as snpe has that option but I didnt fully explore it yet
* snpe-dlc-info can be used to get per layer quant info, I need to combine this info across models and do some analysis to get corelated and problematic layers based on this

:exclamation:**If you find any bugs, have any suggestions, implement any postprocessing functions, have any questions or are confused please message me**:exclamation: