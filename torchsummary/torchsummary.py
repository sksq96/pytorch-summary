import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
from numbers import Number
import numpy as np

def summary(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    result, params_info = summary_string(
        model, input_size, batch_size, device, dtypes)
    print(result)

    return params_info

def _extract_shapes(output, batch_size=-1):
    if isinstance(output, (list, tuple)):
        result = []
        for o in output:
            result.append(_extract_shapes(o))
        return result
    else:
        output_shape = list(output.size())
        output_shape[0] = batch_size
        return output_shape

def _normalize_depth(lst):
    if isinstance(lst[0], Number):
        # Assume this is a flat list already
        return lst
    else:
        result = []
        for element in lst:
            result.extend(_normalize_depth(element))
        return result

def _overflow_sizes(summary):
    kStartLength = 20
    kMidLength = 25
    kEndLength = 15
    kSpaces = 2
    for layer in summary:
        kStartLength = max(kStartLength, len(str(layer)))
        kMidLength = max(kMidLength, len(str(summary[layer]["output_shape"])))
        kEndLength = max(kEndLength, len(str(summary[layer]["nb_params"])))
    total_length = kStartLength + kMidLength + kEndLength + kSpaces
    return kStartLength, kMidLength, kEndLength, total_length


def summary_string(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    if dtypes == None:
        dtypes = [torch.FloatTensor]*len(input_size)

    summary_str = ''

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            summary[m_key]["output_shape"] = _extract_shapes(output, batch_size)

            params = 0
            for param in module.parameters():
                params += torch.prod(torch.LongTensor(list(param.size())))
            summary[m_key]["trainable"] = param.requires_grad
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype).to(device=device)
         for in_size, dtype in zip(input_size, dtypes)]

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

    (kStartLength, kMidLength, kEndLength, kTotalLength) = _overflow_sizes(summary)

    summary_str += "-" * kTotalLength + "\n"
    line_new = "{:>{kStartLength}} {:>{kMidLength}} {:>{kEndLength}}".format(
        "Layer (type)", "Output Shape", "Param #",
        kStartLength=kStartLength,
        kMidLength=kMidLength,
        kEndLength=kEndLength)
    summary_str += line_new + "\n"
    summary_str += "=" * kTotalLength + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0

    for layer in summary:
        # input_shape, output_shape, trainable nb_params
        line_new = "{:>{kStartLength}} {:>{kMidLength}} {:>{kEndLength},}".format(
            layer,
            str(summary[layer]["output_shape"]),
            summary[layer]["nb_params"],
            kStartLength=kStartLength,
            kMidLength=kMidLength,
            kEndLength=kEndLength
        )
        total_params += summary[layer]["nb_params"]
        output_size = np.reshape(
            _normalize_depth(summary[layer]["output_shape"]), (-1,))
        total_output += np.prod(output_size)
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(sum(input_size, ()))
                           * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. /
                            (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    summary_str += "=" * kTotalLength + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(total_params -
                                                        trainable_params) + "\n"
    summary_str += "-" * kTotalLength + "\n"
    summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    summary_str += "-" * kTotalLength + "\n"
    # return summary
    return summary_str, (total_params, trainable_params)
