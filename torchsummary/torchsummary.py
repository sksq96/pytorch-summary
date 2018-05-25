import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict


def summary(model, input_size):
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
                if hasattr(module, 'weight') and hasattr(module.weight, 'size'):
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    summary[m_key]['trainable'] = module.weight.requires_grad
                if hasattr(module, 'bias') and hasattr(module.bias, 'size'):
                    params +=  torch.prod(torch.LongTensor(list(module.bias.size())))
                summary[m_key]['nb_params'] = params
                
            if (not isinstance(module, nn.Sequential) and 
               not isinstance(module, nn.ModuleList) and 
               not (module == model)):
                hooks.append(module.register_forward_hook(hook))
                
        if torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor
        
        # check if there are multiple inputs to the network
        if isinstance(input_size[0], (list, tuple)):
            x = [Variable(torch.rand(2,*in_size)).type(dtype) for in_size in input_size]
        else:
            x = Variable(torch.rand(2,*input_size)).type(dtype)
            
            
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
        total_params = 0
        trainable_params = 0
        for layer in summary:
            # input_shape, output_shape, trainable, nb_params
            line_new = '{:>20}  {:>25} {:>15}'.format(layer, str(summary[layer]['output_shape']), '{0:,}'.format(summary[layer]['nb_params']))
            total_params += summary[layer]['nb_params']
            if 'trainable' in summary[layer]:
                if summary[layer]['trainable'] == True:
                    trainable_params += summary[layer]['nb_params']
            print(line_new)
        print('================================================================')
        print('Total params: {0:,}'.format(total_params))
        print('Trainable params: {0:,}'.format(trainable_params))
        print('Non-trainable params: {0:,}'.format(total_params - trainable_params))
        print('----------------------------------------------------------------')
        # return summary
