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
    
def check_initializers(graph, name):
    tensor = next((x for x in graph.initializer if x.name == name), None)
    if tensor is None:
        return None
    return numpy_helper.to_array(tensor)

def get_values(graph, names):
    value = None
    index = 0
    for i, name in enumerate(names):
        tensor = next((x for x in graph.initializer if x.name == name), None)
        if tensor is not None:
            value = numpy_helper.to_array(tensor)    
            return value, i

def fuse_add_mul_into_conv(model_path, output_path):
    """A function to fuse multiple nodes around a Conv node into a single Conv node.
    The function first takes all constant nodes and converts them into initializers.
    Then it looks for the following patterns:
    1. Add/Sub -> Mul -> Conv
    2. Mul -> Add/Sub -> Conv
    3. Conv -> Mul -> Add/Sub
    
    These patterns can be relatively easily fused into a single Conv node without any loss of accuracy and providing an inference time speedup.
    """
     # Load the ONNX model
    model = onnx.load(model_path)
    graph = model.graph
    first_node = None
    second_node = None
    third_node = None
    
    constants ={}
    const_nodes = {}
    nodes_to_remove = []
    new_initializers = []
    # need to make initialiters from const nodes
    for i, node in enumerate(graph.node):
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    const_name = node.output[0]
                    constant_value = numpy_helper.to_array(attr.t)

                    new_initializers.append(numpy_helper.from_array(constant_value, name=const_name))
                    nodes_to_remove.append(node)
                    
                    for other_node in graph.node:
                        for i, input in enumerate(other_node.input):
                            if input == const_name:
                                other_node.input[i] = const_name
                    
    graph.initializer.extend(new_initializers)
    for node in nodes_to_remove:
        graph.node.remove(node)
    
    nodes_to_remove = []

    for i, node in enumerate(graph.node):
        first_node = second_node
        second_node = third_node
        third_node = node

        if not first_node or not second_node or not third_node:
            continue
        # add/sub -> mul -> conv
        if first_node.op_type in ["Add", "Sub"] and second_node.op_type == "Mul" and third_node.op_type == "Conv":
            print(f"Found nodes: {first_node.op_type}, {second_node.op_type}, {third_node.op_type}")
            
            # Get the Conv node weights
            conv_weight_name = third_node.input[1]
            conv_weight_tensor = next(x for x in graph.initializer if x.name == conv_weight_name)
            conv_weight = numpy_helper.to_array(conv_weight_tensor)

            bias = 0
            bias_name = None
            bias_tensor = None
            if len(third_node.input) > 2:
                bias_name = third_node.input[2]
                bias_tensor = next((x for x in model.graph.initializer if x.name == bias_name), None)
                bias = numpy_helper.to_array(bias_tensor)

            # Get the Mul node weights
            mul_weight, mul_index = get_values(graph, second_node.input)

            # Get the Add/Sub node weights
            add_weight, add_index = get_values(graph, first_node.input)
            if first_node.op_type == "Sub":
                add_weight = -add_weight
            
            add_weight = add_weight * mul_weight
            
            # Calculate the new weights
            new_weight = conv_weight * mul_weight
            
            bias = bias + np.sum(add_weight * conv_weight, axis=(1, 2, 3))
            
            # Update the Conv node weights
            new_conv_tensor = numpy_helper.from_array(new_weight, name=conv_weight_name)
            graph.initializer.remove(conv_weight_tensor)
            graph.initializer.append(new_conv_tensor)

            #adjust the input of the conv node
            third_node.input[0] = first_node.input[abs(add_index - 1)]
            
            # Update the bias tensor
            old_tensor = bias_tensor
            if bias_name is None:
                bias_name = first_node.input[add_index] + "_bias"
            bias_tensor = numpy_helper.from_array(bias, name=bias_name)
            if len(third_node.input) == 2: # no bias present, so add it
                graph.initializer.append(bias_tensor)
                third_node.input.append(bias_name)
            else: # bias is present
                graph.initializer.remove(old_tensor)
                graph.initializer.append(bias_tensor)
            
            nodes_to_remove.append(first_node)
            nodes_to_remove.append(second_node)
            
        # mul -> add -> conv
        if first_node.op_type == "Mul" and second_node.op_type in ["Add", "Sub"] and third_node.op_type == "Conv":
            print(f"Found nodes: {first_node.op_type}, {second_node.op_type}, {third_node.op_type}")
            conv_weight_name = third_node.input[1]
            conv_weight_tensor = next(x for x in graph.initializer if x.name == conv_weight_name)
            conv_weight = numpy_helper.to_array(conv_weight_tensor)

            bias = 0
            bias_name = None
            bias_tensor = None
            if len(third_node.input) > 2:
                bias_name = third_node.input[2]
                bias_tensor = next((x for x in model.graph.initializer if x.name == bias_name), None)
                bias = numpy_helper.to_array(bias_tensor)

            mul_weight, mul_index = get_values(graph, first_node.input)

            add_weight, add_index = get_values(graph, second_node.input)
            if second_node.op_type == "Sub":
                add_weight = -add_weight
            
            # Calculate the new weights
            new_weight = conv_weight * mul_weight
            
            bias = bias + np.sum(add_weight * conv_weight, axis=(1, 2, 3))
            
            # Update the Conv node weights
            new_conv_tensor = numpy_helper.from_array(new_weight, name=conv_weight_name)
            graph.initializer.remove(conv_weight_tensor)
            graph.initializer.append(new_conv_tensor)

            #adjust the input of the conv node
            third_node.input[0] = first_node.input[abs(mul_index - 1)]
            
            # Update the bias tensor
            old_tensor = bias_tensor
            if bias_name is None:
                bias_name = second_node.input[mul_index] + "_bias"
            bias_tensor = numpy_helper.from_array(bias, name=bias_name)
            if len(third_node.input) == 2: # no bias present, so add it
                graph.initializer.append(bias_tensor)
                third_node.input.append(bias_name)
            else: # bias is present
                graph.initializer.remove(old_tensor)
                graph.initializer.append(bias_tensor)
                
            nodes_to_remove.append(first_node)
            nodes_to_remove.append(second_node)
            
        # conv -> mul -> add
        if first_node.op_type == "Conv" and second_node.op_type == "Mul" and third_node.op_type in ["Add", "Sub"]:
            print(f"Found nodes: {first_node.op_type}, {second_node.op_type}, {third_node.op_type}")
            # Get the Conv node weights
            conv_weight_name = first_node.input[1]
            conv_weight_tensor = next(x for x in graph.initializer if x.name == conv_weight_name)
            conv_weight = numpy_helper.to_array(conv_weight_tensor)

            bias = 0
            bias_name = None
            bias_tensor = None
            if len(first_node.input) > 2:
                bias_name = first_node.input[2]
                bias_tensor = next((x for x in model.graph.initializer if x.name == bias_name), None)
                bias = numpy_helper.to_array(bias_tensor)

            mul_weight, mul_index = get_values(graph, second_node.input)

            add_weight, add_index = get_values(graph, third_node.input)
            if third_node.op_type == "Sub":
                add_weight = -add_weight
            
            new_weight = conv_weight * mul_weight
            bias = bias * mul_weight + add_weight
            
            new_conv_tensor = numpy_helper.from_array(new_weight, name=conv_weight_name)
            graph.initializer.remove(conv_weight_tensor)
            graph.initializer.append(new_conv_tensor)

            #adjust the OUTPUT of the conv node              
            first_node.output[0] = third_node.output[0]
            
            # Update the bias tensor
            old_tensor = bias_tensor
            
            if len(first_node.input) == 2: # no bias present, so add it
                bias_name = first_node.input[mul_index] + "_bias"
                bias_tensor = numpy_helper.from_array(bias, name=bias_name)
                graph.initializer.append(bias_tensor)
                first_node.input.append(bias_name)
            else: # bias is present
                bias_tensor = numpy_helper.from_array(bias, name=bias_name)
                graph.initializer.remove(old_tensor)
                graph.initializer.append(bias_tensor)
            
            nodes_to_remove.append(second_node)
            nodes_to_remove.append(third_node)

    for node in nodes_to_remove:
        graph.node.remove(node)
        
    onnx.save(model, output_path)
    
        
model_path = "shared_with_container/models/ocr.onnx"
output = "shared_with_container/models/ocr-stride.onnx"

stride_removal(model_path, output)

model = onnx.load(output)

# validate graph
onnx.checker.check_model(model)
