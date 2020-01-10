import torch
import pandas as pd
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np


def summary(model, input_size, batch_size=-1, device="cuda", verbose = False, csv = False):

    def register_hook(module):

        def hook(module, input, output):
            # name of operation (ex: Conv2)
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            # Conv2-index
            m_key = "%s-%i" % (class_name, module_idx + 1)
            # element of m_key is another OrderedDict()
            summary[m_key] = OrderedDict()
            # Pass Key and Elements to inside OrderedDict()
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
                   # if class if Conv, attain: kernel_size, stride, padding
            if class_name.__contains__("Conv"):
                summary[m_key]["kernel_size"] = module.kernel_size
                summary[m_key]["stride"] = module.stride
                summary[m_key]["padding"] = module.padding

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
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    if verbose:
        print("-" * 112)
        line_new = "{:>20} {:>20}  {:>20}  {:>10}  {:>10}  {:>10}  {:>10}".format("Layer (type)", "Input Shape","Output Shape","kernel_size","stride","padding", "Param #")
        print(line_new)
        print("=" * 112)
    else:
        print("----------------------------------------------------------------")
        line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
        print(line_new)
        print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    if verbose:
        for layer in summary:
            # if false, add parameters
            if not layer.__contains__("Conv"):
                summary[layer]['kernel_size'] = 0
                summary[layer]['stride'] = 0
                summary[layer]['padding'] = 0

            line_new = "{:>20} {:>20}  {:>20}  {:>10}  {:>10}  {:>10}  {:>10}".format(
                layer,
                str(summary[layer]["input_shape"]),
                str(summary[layer]["output_shape"]),
                "{0}".format(summary[layer]['kernel_size']),
                "{0}".format(summary[layer]['stride']),
                "{0}".format(summary[layer]['padding']),
                "{0:,}".format(summary[layer]["nb_params"]),
            )
            total_params += summary[layer]["nb_params"]
            total_output += np.prod(summary[layer]["output_shape"])
            if "trainable" in summary[layer]:
                if summary[layer]["trainable"] == True:
                    trainable_params += summary[layer]["nb_params"]
            print(line_new)


    else:
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
    if verbose:
        print("="*112)
        print("Total params: {0:,}".format(total_params))
        print("Trainable params: {0:,}".format(trainable_params))
        print("Non-trainable params: {0:,}".format(total_params - trainable_params))
        print("-"*112)
        print("Input size (MB): %0.2f" % total_input_size)
        print("Forward/backward pass size (MB): %0.2f" % total_output_size)
        print("Params size (MB): %0.2f" % total_params_size)
        print("Estimated Total Size (MB): %0.2f" % total_size)
        print("-"*112)
    else:
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

    # if csv == True, print a df
    if csv:
        if verbose:
            cols = ["input_shape","output_shape","kernel_size","stride","padding", "nb_params"]
        else:
            cols = ["output_shape", "nb_params"]
        idx = summary.keys()
        vals = []
        for layer in summary:
            dict_vals = [summary[layer][key] if type(summary[layer][key]) != torch.Tensor else summary[layer][key].item() for key in cols]
            vals.append(dict_vals)



        df = pd.DataFrame(vals, index = idx, columns = cols)
        df.index.name = 'Layer'
        df.to_csv('model_layers.csv')

        # Create a second df with: total params, trainable params, non-trainable params, input size, forward/backward pass, params size, estimated total size
        cols = ['Total params', 'Trainable params', 'Non-trainable params', 'Input size (MB)', "Forward/backward pass size (MB):","Params size (MB)", "Estimated Total Size (MB)"]
        vals = [total_params.item(), trainable_params.item(), (total_params - trainable_params).item(), total_input_size, total_output_size, total_params_size, total_size]
        df2 = pd.DataFrame(vals, index = cols, columns = ['model params']).transpose()
        df2.to_csv('model_params.csv')







