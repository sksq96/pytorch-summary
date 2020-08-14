import unittest
from torchsummary import summary, summary_string
from torchsummary.torchsummary import _build_summary_dict, _build_summary_string
from torchsummary.tests.test_models.test_model import SingleInputNet, MultipleInputNet, \
    MultipleInputNetDifferentDtypes, NestedNet, CustomModule
import torch

gpu_if_available = "cuda:0" if torch.cuda.is_available() else "cpu"


class TorchSummaryTests(unittest.TestCase):
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
        total_params, trainable_params = summary(
            model, [input1, input2], device="cpu")
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
        if torch.cuda.is_available():
            model.cuda()
        input = (1, 2)
        total_params, trainable_params = summary(model, input, device=gpu_if_available)
        self.assertEqual(total_params, 15)
        self.assertEqual(trainable_params, 15)

    def test_multiple_input_types(self):
        model = MultipleInputNetDifferentDtypes()
        input1 = (1, 300)
        input2 = (1, 300)
        dtypes = [torch.FloatTensor, torch.LongTensor]
        total_params, trainable_params = summary(
            model, [input1, input2], device="cpu", dtypes=dtypes)
        self.assertEqual(total_params, 31120)
        self.assertEqual(trainable_params, 31120)

    def test_recursive(self):
        model = NestedNet()
        input = (1, 28, 28)
        summary = _build_summary_dict(model, [input], device='cpu')
        summary_str, (total_params, trainable_params) = _build_summary_string(summary, [input])

        self.assertListEqual(list(summary.keys()), ['Conv2d-1', 'BatchNorm2d-2', 'MaxPool2d-3', 'ConvBlock-4',
                                                    'Conv2d-5', 'BatchNorm2d-6', 'MaxPool2d-7', 'ConvBlock-8',
                                                    'Dropout2d-9', 'Linear-10', 'Linear-11', 'NestedNet-12'])
        self.assertEqual(total_params, 21900)
        self.assertEqual(trainable_params, 21900)

        summary = _build_summary_dict(model, [input], device='cpu', recurse=False)
        summary_str, (total_params, trainable_params) = _build_summary_string(summary, [input])
        self.assertListEqual(list(summary.keys()), ['ConvBlock-1', 'ConvBlock-2', 'Dropout2d-3', 'Linear-4',
                                                    'Linear-5', 'NestedNet-6'])
        self.assertEqual(total_params, 21900)
        self.assertEqual(trainable_params, 21900)

    def test_custom_module(self):
        model = CustomModule()
        input = (1, 50)
        total_params, trainable_params = summary(model, input, device='cpu')
        self.assertEqual(total_params, 2500)
        self.assertEqual(trainable_params, 2500)


class TorchSummaryStringTests(unittest.TestCase):
    def test_single_input(self):
        model = SingleInputNet()
        input = (1, 28, 28)
        result, (total_params, trainable_params) = summary_string(
            model, input, device="cpu")
        self.assertEqual(type(result), str)
        self.assertEqual(total_params, 21840)
        self.assertEqual(trainable_params, 21840)


if __name__ == '__main__':
    unittest.main(buffer=True)
