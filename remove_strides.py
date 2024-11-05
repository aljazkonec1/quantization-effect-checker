import onnx
import numpy as np
from onnx import helper, TensorProto, numpy_helper
    
def stride_removal(model_path, output_path):
    # Load the ONNX model
    model = onnx.load(model_path)
    graph = model.graph
    conv_index = 0
    conv_node = None
    mul_node = None
    start = [0, 0, 0, 0]
    axes = [0, 1, 2, 3]
    
    strides = [1, 1]
    for i, node in enumerate(graph.node):
        
        if node.op_type == "Conv":
            for attr in node.attribute:
                if attr.name == "strides":
                    strides = attr.ints
                    break
            if strides[0] != strides[1]:
                print(f"Convolution node {node.name} has different strides in each dimension: {strides}")
                conv_node = node
                conv_index = i
                continue
            
        if conv_node:
            next_node = node
            shape = None
            
            for value_info in model.graph.value_info:
                if value_info.name == conv_node.input[0]:
                    shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
                    break
            if shape is None:
                raise ValueError(f"Could not find shape for tensor {conv_node.input[0]}")
            print(f"Shape of input tensor: {shape}")
            
            steps = [1, 1]
            steps.extend(strides)

            steps_tensor = helper.make_tensor("steps_" + str(i), onnx.TensorProto.INT64, [4], steps)
            end_tensor = helper.make_tensor("end_" + str(i), onnx.TensorProto.INT64, [4], shape)
            start_tensor = helper.make_tensor("start_" + str(i), onnx.TensorProto.INT64, [4], start)
            axes_tensor = helper.make_tensor("axes_" + str(i), onnx.TensorProto.INT64, [4], axes)
            
            for attr in conv_node.attribute:
                if attr.name == "strides":
                    attr.ints[:] = [1, 1]

            model.graph.initializer.extend([start_tensor, end_tensor, axes_tensor, steps_tensor])
            
            # Create and insert the Slice node after Conv node
            slice_node = helper.make_node(
                "Slice",
                inputs=[conv_node.output[0], "start_" + str(i), "end_" + str(i), "axes_" + str(i), "steps_" + str(i)],
                outputs=[conv_node.output[0] + "_sliced"],
                name="SliceNode_" + str(i),
            )
            graph.node.insert(conv_index + 1, slice_node) 

            next_node.input[1] = slice_node.output[0]

            conv_node = None
            strides = [1, 1]
            
  
    onnx.save(model, output_path)
    
        
model_path = "shared_with_container/models/ocr.onnx"
output = "shared_with_container/models/ocr-stride.onnx"

stride_removal(model_path, output)

model = onnx.load(output)

# validate graph
onnx.checker.check_model(model)
