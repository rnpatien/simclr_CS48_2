# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SimCLR configurations."""
import dataclasses
import os
from typing import List, Optional

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.vision.beta.projects.simclrRP.modeling import simclr_model
from official.vision.configs import backbones
from official.vision.configs import common


@dataclasses.dataclass
class Decoder(hyperparams.Config):
  decode_label: bool = True


@dataclasses.dataclass
class Parser(hyperparams.Config):
  """Parser config."""
  aug_rand_crop: bool = True
  aug_rand_hflip: bool = True
  aug_color_distort: bool = True
  aug_color_jitter_strength: float = 1.0
  aug_color_jitter_impl: str = 'simclrv2'  # 'simclrv1' or 'simclrv2'
  aug_rand_blur: bool = True
  parse_label: bool = True
  test_crop: bool = True
  mode: str = simclr_model.PRETRAIN


@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
  """Training data config."""
  input_path: str = ''
  global_batch_size: int = 0
  is_training: bool = True
  dtype: str = 'float32'
  shuffle_buffer_size: int = 10000
  cycle_length: int = 10
  # simclr specific configs
  parser: Parser = Parser()
  decoder: Decoder = Decoder()
  # Useful when doing a sanity check that we absolutely use no labels while
  # pretrain by setting labels to zeros (default = False, keep original labels)
  input_set_label_to_zero: bool = False

@dataclasses.dataclass
class DecodeHead(hyperparams.Config):
  zdim: int = 1024

@dataclasses.dataclass
class ProjectionHead(hyperparams.Config):
  proj_output_dim: int = 128
  num_proj_layers: int = 3
  ft_proj_idx: int = 1  # layer of the projection head to use for fine-tuning.


@dataclasses.dataclass
class SupervisedHead(hyperparams.Config):
  num_classes: int = 1001
  zero_init: bool = False


@dataclasses.dataclass
class ContrastiveLoss(hyperparams.Config):
  projection_norm: bool = True
  temperature: float = 0.1
  l2_weight_decay: float = 0.0


@dataclasses.dataclass
class ClassificationLosses(hyperparams.Config):
  label_smoothing: float = 0.0
  one_hot: bool = True
  l2_weight_decay: float = 0.0


@dataclasses.dataclass
class Evaluation(hyperparams.Config):
  top_k: int = 5
  one_hot: bool = True


@dataclasses.dataclass
class SimCLRModel(hyperparams.Config):
  """SimCLR model config."""
  input_size: List[int] = dataclasses.field(default_factory=list)
  backbone: backbones.Backbone = backbones.Backbone(
      type='resnet', resnet=backbones.ResNet())
  decoder_head: DecodeHead = DecodeHead(zdim=2048) 
  projection_head: ProjectionHead = ProjectionHead(
      proj_output_dim=128, num_proj_layers=3, ft_proj_idx=1)
  supervised_head: SupervisedHead = SupervisedHead(num_classes=1001)
  norm_activation: common.NormActivation = common.NormActivation(
      norm_momentum=0.9, norm_epsilon=1e-5, use_sync_bn=False)
  mode: str = simclr_model.PRETRAIN
  backbone_trainable: bool = True


@dataclasses.dataclass
class SimCLRPretrainTask(cfg.TaskConfig):
  """SimCLR pretraining task config."""
  model: SimCLRModel = SimCLRModel(mode=simclr_model.PRETRAIN)
  train_data: DataConfig = DataConfig(
      parser=Parser(mode=simclr_model.PRETRAIN), is_training=True)
  validation_data: DataConfig = DataConfig(
      parser=Parser(mode=simclr_model.PRETRAIN), is_training=False)
  loss: ContrastiveLoss = ContrastiveLoss()
  evaluation: Evaluation = Evaluation()
  init_checkpoint: Optional[str] = None
  # all or backbone
  init_checkpoint_modules: str = 'all'


@dataclasses.dataclass
class SimCLRFinetuneTask(cfg.TaskConfig):
  """SimCLR fine tune task config."""
  model: SimCLRModel = SimCLRModel(
      mode=simclr_model.FINETUNE,
      supervised_head=SupervisedHead(num_classes=1001, zero_init=True))
  train_data: DataConfig = DataConfig(
      parser=Parser(mode=simclr_model.FINETUNE), is_training=True)
  validation_data: DataConfig = DataConfig(
      parser=Parser(mode=simclr_model.FINETUNE), is_training=False)
  loss: ClassificationLosses = ClassificationLosses()
  evaluation: Evaluation = Evaluation()
  init_checkpoint: Optional[str] = None
  # all, backbone_projection or backbone
  init_checkpoint_modules: str = 'backbone_projection'

@dataclasses.dataclass
class SimCLRDecodeTask(cfg.TaskConfig):
  """SimCLR fine tune task config."""
  model: SimCLRModel = SimCLRModel(
      mode=simclr_model.DECODE,
      supervised_head=SupervisedHead(num_classes=10, zero_init=True))
  train_data: DataConfig = DataConfig(
      parser=Parser(mode=simclr_model.DECODE), is_training=True)
  validation_data: DataConfig = DataConfig(
      parser=Parser(mode=simclr_model.DECODE), is_training=False)
  loss: ContrastiveLoss = ContrastiveLoss()
  evaluation: Evaluation = Evaluation()
  init_checkpoint: Optional[str] = None
  # all, backbone_projection or backbone
  init_checkpoint_modules: str = 'backbone_projection'

@exp_factory.register_config_factory('simclr_pretraining')
def simclr_pretraining() -> cfg.ExperimentConfig:
  """Image classification general."""
  return cfg.ExperimentConfig(
      task=SimCLRPretrainTask(),
      trainer=cfg.TrainerConfig(),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])


@exp_factory.register_config_factory('simclr_finetuning')
def simclr_finetuning() -> cfg.ExperimentConfig:
  """Image classification general."""
  return cfg.ExperimentConfig(
      task=SimCLRFinetuneTask(),
      trainer=cfg.TrainerConfig(),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])

@exp_factory.register_config_factory('simclr_decode')
def simclr_decode() -> cfg.ExperimentConfig:
  """Image classification general."""
  return cfg.ExperimentConfig(
      task=SimCLRDecodeTask(),
      trainer=cfg.TrainerConfig(),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])
