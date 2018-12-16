import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np


def check_same(stride):
    if isinstance(stride, (list, tuple)):
        assert len(stride) == 2 and stride[0] == stride[1]
        stride = stride[0]
    return stride

def summary(model, input_size, batch_size=-1, device="cuda",  has_receptive_field=False):

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size                
                
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

            if has_receptive_field:
                rf_module_idx = len(receptive_field)
                rf_m_key = "%i" % rf_module_idx
                rf_p_key = "%i" % (rf_module_idx - 1)
                
                receptive_field[rf_m_key] = OrderedDict()
                if not receptive_field["0"]["conv_stage"]:
                    # print("Enter in deconv_stage")
                    receptive_field[rf_m_key]["j"] = 0
                    receptive_field[rf_m_key]["r"] = 0
                    receptive_field[rf_m_key]["start"] = 0
                else:
                    p_j = receptive_field[rf_p_key]["j"]
                    p_r = receptive_field[rf_p_key]["r"]
                    p_start = receptive_field[rf_p_key]["start"]
                    
                    if class_name == "Conv2d" or class_name == "MaxPool2d":
                        kernel_size = module.kernel_size
                        stride = module.stride
                        padding = module.padding
                        kernel_size, stride, padding = map(check_same, [kernel_size, stride, padding])
                        receptive_field[rf_m_key]["j"] = p_j * stride
                        receptive_field[rf_m_key]["r"] = p_r + (kernel_size - 1) * p_j
                        receptive_field[rf_m_key]["start"] = p_start + (int((kernel_size - 1) / 2) - padding) * p_j
                    elif class_name in ["BatchNorm2d", "ReLU", "Bottleneck", "Dropout2d"]:
                        receptive_field[rf_m_key]["j"] = p_j
                        receptive_field[rf_m_key]["r"] = p_r
                        receptive_field[rf_m_key]["start"] = p_start
                    elif class_name in ["ConvTranspose2d", "Linear"] :
                        receptive_field["0"]["conv_stage"] = False
                        receptive_field[rf_m_key]["j"] = 0
                        receptive_field[rf_m_key]["r"] = 0
                        receptive_field[rf_m_key]["start"] = 0
                    else:
                        print("class_name", class_name)
                        raise ValueError("module not ok")
                receptive_field[rf_m_key]["input_shape"] = list(input[0].size()) # only one
                receptive_field[rf_m_key]["input_shape"][0] = batch_size        
                
                if isinstance(output, (list, tuple)):
                    # list/tuple
                    receptive_field[rf_m_key]["output_shape"] = [
                        [-1] + list(o.size())[1:] for o in output
                    ]
                else:
                    # tensor
                    receptive_field[rf_m_key]["output_shape"] = list(output.size())
                    receptive_field[rf_m_key]["output_shape"][0] = batch_size

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))
    
    
    # create properties
    summary = OrderedDict()
    # create properties
    if has_receptive_field:
        receptive_field = OrderedDict()
        receptive_field["0"] = OrderedDict()
        receptive_field["0"]["j"] = 1.0
        receptive_field["0"]["r"] = 1.0
        receptive_field["0"]["start"] = 0.5
        receptive_field["0"]["conv_stage"] = True
        # receptive_field["0"]["output_shape"] = list(x.size())
        receptive_field["0"]["output_shape"] = [2, in_size[0], in_size[1], in_size[2]]
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    # return summary
    

    summary_keys = summary.keys()
    if has_receptive_field:
        print("\n\n------------------------------------------------------------------------------")
        line_new = "{:>20}  {:>10} {:>10} {:>10} {:>15} ".format("Layer (type)", "map size", "start", "jump", "field_size")
        print(line_new)
        print("==============================================================================")
        total_params = 0
        total_output = 0
        trainable_params = 0
        # for sy_layer, layer in zip(summary, receptive_field):
        for ldx, layer in enumerate(receptive_field):
            assert "start" in receptive_field[layer], layer
            # assert len(receptive_field[layer]["output_shape"]) == 4
            if ldx == 0:
                layer_name = "Input"
            else:
                layer_name = summary_keys[ldx - 1]
            line_new = "{:7} {:12}  {:>10} {:>10} {:>10} {:>15} ".format(
                "",
                # layer,
                # sy_layer,
                layer_name,
                str(receptive_field[layer]["output_shape"][2:]),
                str(receptive_field[layer]["start"]),
                str(receptive_field[layer]["j"]),
                format(str(receptive_field[layer]["r"]))
            )
            print(line_new)

        print("==============================================================================")
