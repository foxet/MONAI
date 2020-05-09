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

import torch
import torch.nn as nn
import torch.optim as optim
from parameterized import parameterized

from monai.losses import DiceLoss, FocalLoss, GeneralizedDiceLoss, TverskyLoss

TEST_CASES = [
    [DiceLoss, {"to_onehot_y": True, "do_sigmoid": True}, {"smooth": 1e-4}],
    [FocalLoss, {"gamma": 1.5}, {}],
    [GeneralizedDiceLoss, {"to_onehot_y": True, "do_sigmoid": True}, {}],
    [TverskyLoss, {"to_onehot_y": True, "do_sigmoid": True}, {}],
]


class TestSegLossIntegration(unittest.TestCase):
    def setUp(self):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(0)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")

    def tearDown(self):
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    @parameterized.expand(TEST_CASES)
    def test_convergence(self, loss_type, loss_args, forward_args):
        """
        The goal of this test is to assess if the gradient of the loss function
        is correct by testing if we can train a one layer neural network
        to segment one image.
        We verify that the loss is decreasing in almost all SGD steps.
        """
        learning_rate = 0.001
        max_iter = 20

        # define a simple 3d example
        target_seg = torch.tensor(
            [
                [
                    # raw 0
                    [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
                    # raw 1
                    [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
                    # raw 2
                    [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
                ]
            ],
            device=self.device,
        )
        target_seg = torch.unsqueeze(target_seg, dim=0)
        image = 12 * target_seg + 27
        image = image.float().to(self.device)
        num_classes = 2
        num_voxels = 3 * 4 * 4

        # define a one layer model
        class OnelayerNet(nn.Module):
            def __init__(self):
                super(OnelayerNet, self).__init__()
                self.layer = nn.Linear(num_voxels, num_voxels * num_classes)

            def forward(self, x):
                x = x.view(-1, num_voxels)
                x = self.layer(x)
                x = x.view(-1, num_classes, 3, 4, 4)
                return x

        # initialise the network
        net = OnelayerNet().to(self.device)

        # initialize the loss
        loss = loss_type(**loss_args)

        # initialize an SGD
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

        loss_history = []
        # train the network
        for _ in range(max_iter):
            # set the gradient to zero
            optimizer.zero_grad()

            # forward pass
            output = net(image)

            loss_val = loss(output, target_seg, **forward_args)

            # backward pass
            loss_val.backward()
            optimizer.step()

            # stats
            loss_history.append(loss_val.item())

        # count the number of SGD steps in which the loss decreases
        num_decreasing_steps = 0
        for i in range(len(loss_history) - 1):
            if loss_history[i] > loss_history[i + 1]:
                num_decreasing_steps += 1
        decreasing_steps_ratio = float(num_decreasing_steps) / (len(loss_history) - 1)
        print(f"{loss_type.__name__}: ratio of decreasing_steps {decreasing_steps_ratio}")

        # verify that the loss is decreasing for 80% of the SGD steps
        self.assertGreaterEqual(decreasing_steps_ratio, 0.8)


if __name__ == "__main__":
    unittest.main()
