# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
import torch
from parameterized import parameterized

from monai.losses import DiceLoss

TEST_CASES = [
    [  # shape: (1, 1, 2, 2), (1, 1, 2, 2)
        {"include_background": True, "do_sigmoid": True},
        {
            "input": torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]]]),
            "target": torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]]),
            "smooth": 1e-6,
        },
        0.307576,
    ],
    [  # shape: (2, 1, 2, 2), (2, 1, 2, 2)
        {"include_background": True, "do_sigmoid": True},
        {
            "input": torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]], [[[1.0, -1.0], [-1.0, 1.0]]]]),
            "target": torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]], [[[1.0, 0.0], [1.0, 0.0]]]]),
            "smooth": 1e-4,
        },
        0.416657,
    ],
    [  # shape: (2, 2, 3), (2, 1, 3)
        {"include_background": False, "to_onehot_y": True},
        {
            "input": torch.tensor([[[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]),
            "target": torch.tensor([[[0.0, 0.0, 1.0]], [[0.0, 1.0, 0.0]]]),
            "smooth": 0.0,
        },
        0.0,
    ],
    [  # shape: (2, 2, 3), (2, 1, 3)
        {"include_background": True, "to_onehot_y": True, "do_sigmoid": True},
        {
            "input": torch.tensor([[[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
            "target": torch.tensor([[[1.0, 0.0, 0.0]], [[1.0, 1.0, 0.0]]]),
            "smooth": 1e-4,
        },
        0.435050,
    ],
    [  # shape: (2, 2, 3), (2, 1, 3)
        {"include_background": True, "to_onehot_y": True, "do_sigmoid": True, "reduction": "none"},
        {
            "input": torch.tensor([[[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
            "target": torch.tensor([[[1.0, 0.0, 0.0]], [[1.0, 1.0, 0.0]]]),
            "smooth": 1e-4,
        },
        [[0.296529, 0.415136], [0.599976, 0.428559]],
    ],
    [  # shape: (2, 2, 3), (2, 1, 3)
        {"include_background": True, "to_onehot_y": True, "do_softmax": True},
        {
            "input": torch.tensor([[[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
            "target": torch.tensor([[[1.0, 0.0, 0.0]], [[1.0, 1.0, 0.0]]]),
            "smooth": 1e-4,
        },
        0.383713,
    ],
    [  # shape: (2, 2, 3), (2, 1, 3)
        {"include_background": True, "to_onehot_y": True, "do_softmax": True, "reduction": "sum"},
        {
            "input": torch.tensor([[[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
            "target": torch.tensor([[[1.0, 0.0, 0.0]], [[1.0, 1.0, 0.0]]]),
            "smooth": 1e-4,
        },
        1.534853,
    ],
    [  # shape: (1, 1, 2, 2), (1, 1, 2, 2)
        {"include_background": True, "do_sigmoid": True},
        {
            "input": torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]]]),
            "target": torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]]),
            "smooth": 1e-6,
        },
        0.307576,
    ],
    [  # shape: (1, 1, 2, 2), (1, 1, 2, 2)
        {"include_background": True, "do_sigmoid": True, "squared_pred": True},
        {
            "input": torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]]]),
            "target": torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]]),
            "smooth": 1e-5,
        },
        0.178337,
    ],
    [  # shape: (1, 1, 2, 2), (1, 1, 2, 2)
        {"include_background": True, "do_sigmoid": True, "jaccard": True},
        {
            "input": torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]]]),
            "target": torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]]),
            "smooth": 1e-5,
        },
        -0.059094,
    ],
]


class TestDiceLoss(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_shape(self, input_param, input_data, expected_val):
        result = DiceLoss(**input_param).forward(**input_data)
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, rtol=1e-5)

    def test_ill_shape(self):
        loss = DiceLoss()
        with self.assertRaisesRegex(AssertionError, ""):
            loss.forward(torch.ones((1, 2, 3)), torch.ones((4, 5, 6)))


if __name__ == "__main__":
    unittest.main()
