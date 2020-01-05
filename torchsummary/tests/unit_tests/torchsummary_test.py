import unittest
from torchsummary import summary
from torchsummary.tests.test_models.test_model import SingleInputNet, MultipleInputNet, MultipleInputNetDifferentDtypes
import torch

class torchsummaryTests(unittest.TestCase):
    def test_single_input(self):
        model = SingleInputNet()
        input = (1, 28, 28)
        total_params, trainable_params = summary(model, input, device="cpu")
        self.assertEqual(total_params, 21840)
        self.assertEqual(trainable_params, 21840)

    def test_multiple_input(self):
        model = MultipleInputNet()
        input1 = (1, 300)
        input2 = (1, 300)
        total_params, trainable_params = summary(model, [input1, input2], device="cpu")
        self.assertEqual(total_params, 31120)
        self.assertEqual(trainable_params, 31120)

    def test_single_layer_network(self):
        model = torch.nn.Linear(2, 5)
        input = (1, 2)
        total_params, trainable_params = summary(model, input, device="cpu")
        self.assertEqual(total_params, 15)
        self.assertEqual(trainable_params, 15)

    def test_single_layer_network_on_gpu(self):
        model = torch.nn.Linear(2, 5)
        model.cuda()
        input = (1, 2)
        total_params, trainable_params = summary(model, input, device="cuda:0")
        self.assertEqual(total_params, 15)
        self.assertEqual(trainable_params, 15)

    def test_multiple_input_types(self):
        model = MultipleInputNetDifferentDtypes()
        input1 = (1, 300)
        input2 = (1, 300)
        dtypes = [torch.FloatTensor, torch.LongTensor]
        total_params, trainable_params = summary(model, [input1, input2], device="cpu", dtypes=dtypes)
        self.assertEqual(total_params, 31120)
        self.assertEqual(trainable_params, 31120)

if __name__ == '__main__':
    unittest.main(buffer=True)
