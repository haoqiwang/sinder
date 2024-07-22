# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.


from sinder.hub.backbones import (
    dinov2_vitb14,
    dinov2_vitb14_reg,
    dinov2_vitg14,
    dinov2_vitg14_reg,
    dinov2_vitl14,
    dinov2_vitl14_reg,
    dinov2_vits14,
    dinov2_vits14_reg,
)

dependencies = ['torch']

__all__ = [
    'dinov2_vitb14',
    'dinov2_vitb14_reg',
    'dinov2_vitg14',
    'dinov2_vitg14_reg',
    'dinov2_vitl14',
    'dinov2_vitl14_reg',
    'dinov2_vits14',
    'dinov2_vits14_reg',
]
