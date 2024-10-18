## quantization-effect-checker
Some helpful files for checking and comparing the effect that quantization has on model performance. Intented for use with Luxonis repos.
This is in early version, need to add automatic running of whole procedure.


The checker uses COCO style annotated dataset so you need to create a postprocess function that transforms detections into compatible type. Currently I made it only for COCO val2017 and only for yolov6n-288x512. 
It should work with other yolos and with minimal adjustments for other models.
## Environment
For model conversion I used local docker installation of [modelconverter](https://github.com/luxonis/modelconverter/tree/main).
I used two python envs, first one was for snpe operations and was installed based on [Qualcomm setup](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-2/setup.html). The second was built for conversion and general usage like in  *requirements.txt*.

## Repo structure

```python
├── snpe # snpe dir
├── data #download data into here
├── raw_outputs # raw output files copied from device
├── models_dlc # each dlc model is stored in its own dir and should contain
                # dlc, info.json, throughput.txt and model-graph.txt
    ├── <model-one>
        ├── model-one.dlc
        ├── info.json
        ├── throughput.txt
        ├── model-one-graph.txt
├── models_onnx #.onnx files of models, usually just one file
    ├── <model-one.onnx>
├── results # results saved as .json files in COCO style annotations
            # each file should have "annotations" (the predicions), "images" (same as gt), 
            # "info" (form models_dlc)
├── shared_with_container # for model converter
├── create_statistics.py # creates csv with perf metrics (reads results dir)
├── get_detections.py # creates jsons in results from models_dlc and models_onnx
├── make_dlc_info_json.py #creates the info.json file that is in each dlc model dir
├── postprocess_functions.py # funcions for postprocessing model outputs 
                            # (same function for both onnx and dlc)
├── to_raw.py # conversion and preprocess of RGB images to raw format for upload to device
├── utils.py # utils
├── visualize_detections.py # visualizes predictions of all results in results dir
```
In addition to the repo setup you need ssh access to a RVC4 device. 
To store your data make a dir in */data* directory, I made a */data/tests/* dir and will refer to it as home dir on the device.

## How to use
This is the general flow:
   1) Get dataset and split it into test and quantization datasets. Save test to *data* dir and quantization dataset to *shared_with_container/calibration_data*
   2) Use *to_raw.py* to convert test data into correct dataformat and upload the files to device
   3) Get onnx model and put it in *shared_with_container/models* and into *model_onnx*
   4) Create config yaml file in *shared_with_container/configs* for the model
   5) Use modelconverter to convert to dlc
   6) Once you have the model, make a *model_name* dir in *models_dlc* and copy the *model_name.dlc* to it.
   7) Copy *model_name.dlc* to the RVC4 device
   8) Now that you have the model and the data on the device you can run predictions and benchmark the inference speed. To do this you run the commands:
      1) Inference speed: 
   ```snpe-throughput-net-run --duration 60 --use_dsp --container <<model_name>> --perf_profile balanced > throughput.txt ``` 
    You can then copy the throughput.txt file and store it in the model dir as described in repo structure
      2) Run inference over test data:
    ```snpe-net-run --container <<model.dlc>> --userbuffer_tf8 --userbuffer_tf8_output --input_list test_raw/inputs_raw.txt --use_dsp --use_native_input_files --use_native_output_files```
        Here there are some protips that I write in sectio **Protips**
    The *snpe-net-run* will produce an output dir where each tested image will have its own dir which contains seperate .raw files for each output head of the model like *outputname_1.raw*, *outputname_2.raw* ...
    You can now copy the entire dir, with all image dirs to the *raw_outputs*. This concludes everything we have to do on the device and can continue on your host machine.
   9) First we will create a model graph and save it as a txt file in the models dir. This is done with:
   ```snpe-dlc-info -i models_dlc/model_name/model_name.dlc > models_dlc/model_name/dlc-info-graph.txt``` 
   The graph contains layer names, operations and quantization information per each layer of the model. For testing only quantized model performance we are mostly interested in the last cuple of line where the output quantization information is stored. 
   10) This information is then, in conjunction with the *throughputs.txt* file from step **8.i**, extracted and saved into a *info.json* file with the use of *make_dlc_info_json.py* script. I didn't manage to add argparse to the script so you have to actually call the function in the file, you can also see what parameters need to be set in the functions description.
   11) If there is no compatible postprocessing function stored in *postprocess_functions.py* you have to make it yourself. The function should return the bbox in either xmin, ymin, xmax, ymax or even better yet in xmin, ymin, w, h format with additional class index and confidence scores.
   12) The postprocess function is used in the *get_detections.py* script which runs through all the models saved in the models_dlc and models_onnx folders. The script then creates result json in *results* folder for each model. Similarly to before, I didnt manage to add argparse so consult function description and set params in the below main run of the file.
   13) Now just run the *create_statistics.py* script to get a .csv file with results :rocket: 
   14) Additionally you can run *visualize_detections.py* that just visualizes all json files in *results/* directory.

## Protips

Here are some protips to help you debug your problem faster:
* Check the order in which your model predicts classes, as different tools store categories with different ids. I just downloaded the original [MS COCO val2017](https://cocodataset.org/#home) dataset which has the expanded classes type annotation while the tested yolov6 model has the "old" style. I added logic to map between the two id systems. 
* If the model expects input to be FP16, you need to adjust *to_raw.py* to save in FP16 format, but be careful! For example if the model has a normalization layer a the start and is FP16 format you need to save in range 0-255 with fp16 type (ie. the difference between 255 143 97 and 255. 143. 97.)
* For `snpe-net-run` it is very important to correctly set *userbuffer* flags, the above command will work if data is stored in INT8 format, if not you have to set flags like `--userbufferN 16 --userbufferN_output 16` more options can be found on [qualcomm website](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-2/tools.html) under `snpe-net-run` tool.
* A good way if to test if you setup the buffers correctly is, when you run the `snpe-net-run` command it should print out only one image name per QNN operation, if you have it set wrong it will print two or more images per QNN operation.
* Onnx models accept images in NCHW format while dlc models accept in NHWC format so make sure you convert it correctly. In addition to this, the dlc models outputs are also in a different shape (you can check it in the dlc-model-graph.txt file). For example in yolov6n the onnx model outputs are of shape (1, 85, 18, 32) but dlc model returns in (1, 18, 32, 85) so keep that in mind.

## Quantization flags
There are multiple additional flags available for quantization, here I just list a few. For more info on them check [dlc-quant-info](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-2/tools.html#snpe-dlc-quant) and [quantized vs unquantized models](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-2/quantized_models.html) from Qualcomm.

- `--use_per_channel_quantization` selects per-channel quantization, only works for conv, deconv, and FC loperations, for all other regular per-layer quant is used
- `--act_bitwidth` bitwidth for activation layers (either 8 or 16 bit)
- `--bias_bitwidth` bitwidth for biases
- `--act_quantizer` indicates what quantizer to use, can choose between `tf`, `enhanced` and `symmetric`. You can check the docs for more info


## Grand idea
So this is still a very early project (if you can even call it that). But the end goal is to better understand how quantization affects models and have a data-based approach to measuring model performance, inference speed and finding problematic layers in the model.
* I need to add parse args and a script to run the entire pipeline automatically. 
* Need to explore why FP16 model has better mAP then the underlying ONNX model
* I need to beter explore mixed quantization as snpe has that option but I didnt fully explore it yet
* SNPE offers a per-layer inference speed benchmark and I will add some higher level analysis of these values and try to find correlation/problematic layers between different quantization approaches.
* snpe-dlc-info can be used to get per layer quant info, I need to combine this info across models and do some analysis to get corelated and problematic layers based on this

:exclamation:**If you find any bugs, have any suggestions, implement any postprocessing functions, have any questions or are confused please message me**:exclamation: