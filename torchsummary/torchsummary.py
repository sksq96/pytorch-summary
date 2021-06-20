import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# import cifar_net, mnist_net
from models import config, cifar_net, mnist_net


from collections import OrderedDict
import numpy as np

def summary(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    result, params_info = summary_string(
        model, input_size, batch_size, device, dtypes)
    print(result)

    return params_info


def summary_string(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    if dtypes == None:
        dtypes = [torch.FloatTensor]*len(input_size)
        
    # initialize variables
    jump_in, jump_out, rf_list = [1], [1], [1] # to hold jump params &  receptive field values for each layer
    receptive_field = 1
    
    summary_str = ''

    def register_hook(module):
        def hook(module, input, output):
            conv2d_layer = torch.nn.modules.conv.Conv2d(5,10,3)
            maxpool_layer = torch.nn.modules.MaxPool2d(2)
            
           
            if isinstance(module, conv2d_layer.__class__) or isinstance(module, maxpool_layer.__class__):    
                
                # print('inside if\n---------')
                kernel_size = module.kernel_size if isinstance(module.kernel_size, int)  else module.kernel_size[0]
                stride = module.stride if isinstance(module.stride, int)  else module.stride[0]
                dilation = module.dilation if isinstance(module.dilation, int)  else module.dilation[0]


                # print(f'kernel_size:{kernel_size}, stride:{stride}, dilation:{dilation}')
                
                # naive logic, but will refactor it in future
                if dilation > 1:
                    kernel_size = kernel_size + dilation                
                
                 # add receptive_field calcn
                current_jump_in = jump_out[-1]
                current_jump_out = current_jump_in * stride
                jump_out.append(current_jump_out) # appending current jump out value to list
                
                # calculating receptive field
                rf_out = rf_list[-1] + (kernel_size - 1)*current_jump_in
                rf_list.append(rf_out)
           
                
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
                
            try:
                summary[m_key]["receptive_field"] = rf_list[-1]
            except:
                summary[m_key]["receptive_field"] = 0
                
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

    summary_str += "------------------------------------------------------------------------------------------------" + "\n"
    line_new = "{:>20}  {:>25} {:>15} {:>25}".format(
        "Layer (type)", "Output Shape", "Param #", "Receptive Field")
    summary_str += line_new + "\n"
    summary_str += "================================================================================================" + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    total_receptive_field = [1]
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
            str(summary[layer]["receptive_field"])
        )
        total_params += summary[layer]["nb_params"]
        total_receptive_field.append(summary[layer]["receptive_field"])
        

        total_output += np.prod(summary[layer]["output_shape"])
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

    summary_str += "================================================================================================" + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Total receptive field: {0:,}".format(max(total_receptive_field)) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(total_params -
                                                        trainable_params) + "\n"
    summary_str += "------------------------------------------------------------------------------------------------" + "\n"
    summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    summary_str += "------------------------------------------------------------------------------------------------" + "\n"
    return summary_str, (total_params, trainable_params)


# TestCases

modelconfig = config.GlobalConfig

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, dilation=2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = Net().to(device)


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, dilation=2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(20, 30, kernel_size=3)
        self.conv4 = nn.Conv2d(30, 50, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2)
        self.conv5 = nn.Conv2d(50, 10, kernel_size=1)
        self.conv6 = nn.Conv2d(10, 30, kernel_size=3, padding=1)
     

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool1(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x
    
    

class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        
        self.pool1 = nn.MaxPool2d(2)
        self.trans1 = nn.Conv2d(20, 10, kernel_size=1)
        
        self.conv3 = nn.Conv2d(10, 30, kernel_size=3)
        self.conv4 = nn.Conv2d(30, 50, kernel_size=3)
        
        self.pool2 = nn.MaxPool2d(2)
        self.trans2 = nn.Conv2d(50, 10, kernel_size=1)
        
        self.conv5 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv6 = nn.Conv2d(20, 30, kernel_size=3)
        
        self.pool3 = nn.MaxPool2d(2)
        self.trans3 = nn.Conv2d(30, 10, kernel_size=1)
        
        self.conv7 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv8 = nn.Conv2d(20, 30, kernel_size=3)
     

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = self.pool1(x)
        x = self.trans1(x)
        
        
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = self.pool2(x)
        x = self.trans2(x)
        
        x = self.conv5(x)
        x = self.conv6(x)
        
        
        x = self.pool3(x)
        x = self.trans3(x)
        
        x = self.conv7(x)
        x = self.conv8(x)
        
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model1 = Net().to(device)
model2 = Net2().to(device)
model3 = Net3().to(device)

