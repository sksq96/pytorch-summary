import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np

def summary(model, input_size, batch_size=-1, precision=32, device="cuda", print_model=True):

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(model_summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            model_summary[m_key] = OrderedDict()
            model_summary[m_key]["input_shape"] = list(input[0].size())
            model_summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                model_summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                model_summary[m_key]["output_shape"] = list(output.size())
                model_summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                model_summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            model_summary[m_key]["nb_params"] = params
            model_summary[m_key]["is_occupy"] = len(module._modules) == 0

            if isinstance(module, nn.ReLU) and module.inplace:
                model_summary[m_key]["is_occupy"] = False

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
        if precision == 32:
            dtype = torch.cuda.FloatTensor
        elif precision == 16:
            dtype = torch.cuda.HalfTensor
    else:
        if precision == 32:
            dtype = torch.FloatTensor
        elif precision == 16:
            dtype = torch.HalfTensor

    # check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        x = [Variable(torch.rand(2, *in_size)).type(dtype) for in_size in input_size]
    else:
        x = Variable(torch.rand(2, *input_size)).type(dtype)

    # create properties
    model_summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(x)

    # remove these hooks
    for h in hooks:
        h.remove()

    if print_model:
        print("----------------------------------------------------------------")
        line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
        print(line_new)
        print("================================================================")

    total_output = 0
    for layer in model_summary:
        if print_model:
            # input_shape, output_shape, trainable, nb_params
            line_new = "{:>20}  {:>25} {:>15}".format(
                layer,
                str(model_summary[layer]["output_shape"]),
                "{0:,}".format(model_summary[layer]["nb_params"]),
            )
            print(line_new)

        if model_summary[layer]["is_occupy"]:
            total_output += np.prod(model_summary[layer]["output_shape"])

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    total_input_size = abs(np.prod(input_size) * batch_size * precision / 8. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * precision / 8. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * precision / 8. / (1024 ** 2.))
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
