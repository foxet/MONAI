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
import torch.nn.functional as F

from monai.losses import FocalLoss


class TestFocalLoss(unittest.TestCase):
    def test_consistency_with_cross_entropy_2d(self):
        # For gamma=0 the focal loss reduces to the cross entropy loss
        focal_loss = FocalLoss(gamma=0.0, reduction="mean")
        ce = nn.CrossEntropyLoss(reduction="mean")
        max_error = 0
        class_num = 10
        batch_size = 128
        for _ in range(100):
            # Create a random tensor of shape (batch_size, class_num, 8, 4)
            x = torch.rand(batch_size, class_num, 8, 4, requires_grad=True)
            # Create a random batch of classes
            l = torch.randint(low=0, high=class_num, size=(batch_size, 8, 4))
            l = l.long()
            if torch.cuda.is_available():
                x = x.cuda()
                l = l.cuda()
            output0 = focal_loss.forward(x, l)
            output1 = ce.forward(x, l)
            a = float(output0.cpu().detach())
            b = float(output1.cpu().detach())
            if abs(a - b) > max_error:
                max_error = abs(a - b)
        self.assertAlmostEqual(max_error, 0.0, places=3)

    def test_consistency_with_cross_entropy_classification(self):
        # for gamma=0 the focal loss reduces to the cross entropy loss
        focal_loss = FocalLoss(gamma=0.0, reduction="mean")
        ce = nn.CrossEntropyLoss(reduction="mean")
        max_error = 0
        class_num = 10
        batch_size = 128
        for _ in range(100):
            # Create a random scores tensor of shape (batch_size, class_num)
            x = torch.rand(batch_size, class_num, requires_grad=True)
            # Create a random batch of classes
            l = torch.randint(low=0, high=class_num, size=(batch_size,))
            l = l.long()
            if torch.cuda.is_available():
                x = x.cuda()
                l = l.cuda()
            output0 = focal_loss.forward(x, l)
            output1 = ce.forward(x, l)
            a = float(output0.cpu().detach())
            b = float(output1.cpu().detach())
            if abs(a - b) > max_error:
                max_error = abs(a - b)
        self.assertAlmostEqual(max_error, 0.0, places=3)

    def test_bin_seg_2d(self):
        # define 2d examples
        target = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]])
        # add another dimension corresponding to the batch (batch size = 1 here)
        target = target.unsqueeze(0)  # shape (1, H, W)
        pred_very_good = 1000 * F.one_hot(target, num_classes=2).permute(0, 3, 1, 2).float()

        # initialize the mean dice loss
        loss = FocalLoss()

        # focal loss for pred_very_good should be close to 0
        focal_loss_good = float(loss.forward(pred_very_good, target).cpu())
        self.assertAlmostEqual(focal_loss_good, 0.0, places=3)

        # Same test, but for target with a class dimension
        target = target.unsqueeze(1)  # shape (1, 1, H, W)
        focal_loss_good = float(loss.forward(pred_very_good, target).cpu())
        self.assertAlmostEqual(focal_loss_good, 0.0, places=3)

    def test_empty_class_2d(self):
        num_classes = 2
        # define 2d examples
        target = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        # add another dimension corresponding to the batch (batch size = 1 here)
        target = target.unsqueeze(0)  # shape (1, H, W)
        pred_very_good = 1000 * F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # initialize the mean dice loss
        loss = FocalLoss()

        # focal loss for pred_very_good should be close to 0
        focal_loss_good = float(loss.forward(pred_very_good, target).cpu())
        self.assertAlmostEqual(focal_loss_good, 0.0, places=3)

    def test_multi_class_seg_2d(self):
        num_classes = 6  # labels 0 to 5
        # define 2d examples
        target = torch.tensor([[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]])
        # add another dimension corresponding to the batch (batch size = 1 here)
        target = target.unsqueeze(0)  # shape (1, H, W)
        pred_very_good = 1000 * F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # initialize the mean dice loss
        loss = FocalLoss()

        # focal loss for pred_very_good should be close to 0
        focal_loss_good = float(loss.forward(pred_very_good, target).cpu())
        self.assertAlmostEqual(focal_loss_good, 0.0, places=3)

    def test_bin_seg_3d(self):
        # define 2d examples
        target = torch.tensor(
            [
                # raw 0
                [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
                # raw 1
                [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
                # raw 2
                [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
            ]
        )
        # add another dimension corresponding to the batch (batch size = 1 here)
        target = target.unsqueeze(0)  # shape (1, H, W, D)
        pred_very_good = 1000 * F.one_hot(target, num_classes=2).permute(0, 4, 1, 2, 3).float()

        # initialize the mean dice loss
        loss = FocalLoss()

        # focal loss for pred_very_good should be close to 0
        focal_loss_good = float(loss.forward(pred_very_good, target).cpu())
        self.assertAlmostEqual(focal_loss_good, 0.0, places=3)


if __name__ == "__main__":
    unittest.main()
