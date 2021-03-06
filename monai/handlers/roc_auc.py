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

from typing import Sequence

import torch
from ignite.metrics import Metric

from monai.metrics import compute_roc_auc


class ROCAUC(Metric):
    """Computes Area Under the Receiver Operating Characteristic Curve (ROC AUC).
    accumulating predictions and the ground-truth during an epoch and applying `compute_roc_auc`.

    Args:
        to_onehot_y (bool): whether to convert `y` into the one-hot format. Defaults to False.
        add_softmax (bool): whether to add softmax function to `y_pred` before computation. Defaults to False.
        average (`macro|weighted|micro|None`): type of averaging performed if not binary classification.
            Default is 'macro'.

            - 'macro': calculate metrics for each label, and find their unweighted mean.
              this does not take label imbalance into account.
            - 'weighted': calculate metrics for each label, and find their average,
              weighted by support (the number of true instances for each label).
            - 'micro': calculate metrics globally by considering each element of the label
              indicator matrix as a label.
            - None: the scores for each class are returned.

        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.Engine` `process_function` output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.

    Note:
        ROCAUC expects y to be comprised of 0's and 1's.  y_pred must either be probability estimates or confidence values.

    """

    def __init__(self, to_onehot_y=False, add_softmax=False, average="macro", output_transform=lambda x: x):
        super().__init__(output_transform=output_transform)
        self.to_onehot_y = to_onehot_y
        self.add_softmax = add_softmax
        self.average = average

    def reset(self):
        self._predictions = []
        self._targets = []

    def update(self, output: Sequence[torch.Tensor]):
        y_pred, y = output
        if y_pred.ndimension() not in (1, 2):
            raise ValueError("predictions should be of shape (batch_size, n_classes) or (batch_size, ).")
        if y.ndimension() not in (1, 2):
            raise ValueError("targets should be of shape (batch_size, n_classes) or (batch_size, ).")

        self._predictions.append(y_pred.clone())
        self._targets.append(y.clone())

    def compute(self):
        _prediction_tensor = torch.cat(self._predictions, dim=0)
        _target_tensor = torch.cat(self._targets, dim=0)
        return compute_roc_auc(_prediction_tensor, _target_tensor, self.to_onehot_y, self.add_softmax, self.average)
