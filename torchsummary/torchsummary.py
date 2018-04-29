import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict


def _summary(model, input_size):
        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split('.')[-1].split("'")[0]
                module_idx = len(summary)

                m_key = '%s-%i' % (class_name, module_idx+1)
                summary[m_key] = OrderedDict()
                summary[m_key]['input_shape'] = list(input[0].size())
                summary[m_key]['input_shape'][0] = -1
                if isinstance(output, (list,tuple)):
                    summary[m_key]['output_shape'] = [[-1] + list(o.size())[1:] for o in output]
                else:
                    summary[m_key]['output_shape'] = list(output.size())
                    summary[m_key]['output_shape'][0] = -1

                params = 0
                if hasattr(module, 'weight'):
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    summary[m_key]['trainable'] = module.weight.requires_grad
                if hasattr(module, 'bias') and hasattr(module.bias, 'size'):
                    params +=  torch.prod(torch.LongTensor(list(module.bias.size())))
                summary[m_key]['nb_params'] = params
                
            if (not isinstance(module, nn.Sequential) and 
               not isinstance(module, nn.ModuleList) and 
               not (module == model)):
                hooks.append(module.register_forward_hook(hook))

        input_size = list(input_size)                
        if torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor
        
        x = Variable(torch.rand(1,*input_size)).type(dtype)
            
            
        # print(type(x[0]))
        # create properties
        summary = OrderedDict()
        hooks = []
        # register hook
        model.apply(register_hook)
        # make a forward pass
        # print(x.shape)
        model(x)
        # remove these hooks
        for h in hooks:
            h.remove()

        print('----------------------------------------------------------------')
        line_new = '{:>20}  {:>25} {:>15}'.format('Layer (type)', 'Output Shape', 'Param #')
        print(line_new)
        print('================================================================')
        line_new = '{:>20}  {:>25} {:>15}'.format('Input Tensor', str([-1,]+input_size), 'None')
        print(line_new)
        total_params = 0
        trainable_params = 0
        for layer in summary:
            # input_shape, output_shape, trainable, nb_params
            line_new = '{:>20}  {:>25} {:>15}'.format(layer, str(summary[layer]['output_shape']), summary[layer]['nb_params'])
            total_params += summary[layer]['nb_params']
            if 'trainable' in summary[layer]:
                if summary[layer]['trainable'] == True:
                    trainable_params += summary[layer]['nb_params']
            print(line_new)
        print('================================================================')
        print('Total params: ' + str(total_params))
        print('Trainable params: ' + str(trainable_params))
        print('Non-trainable params: ' + str(total_params - trainable_params))
        print('----------------------------------------------------------------')
        # return summary


def summary(model, sizes):
    if isinstance(sizes[0], (list, tuple)):
        for idx, size in enumerate(sizes):
            _summary(model, size)
            if idx != len(sizes)-1:
                print('*'*70 +'\n\n')
    else:
        _summary(model, sizes)