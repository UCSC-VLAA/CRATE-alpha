# Copyright 2022 Big Vision Authors.
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

# pylint: disable=line-too-long


import configs.common as bvcc
import ml_collections as mlc




def get_config(arg=None):
  """Config for training."""
  arg = bvcc.parse_arg(arg, variant='L/14',
                       base_lr=0.00001,
                       batchsize=4096,
                       res=224,
                       runlocal=False, aug='')
  config = mlc.ConfigDict()

  config.seed = 0
  config.total_epochs = 91
  config.num_classes = 1000
  config.loss = 'softmax_xent'
  config.init_head_bias = 0.


  config.input = dict()
  config.input.data = dict(
      name='imagenet2012',
      split='train[:100%]',
  )
  
  config.input.batch_size = arg.batchsize
  config.input.cache_raw = not arg.runlocal  # Needs up to 120GB of RAM!
  config.input.shuffle_buffer_size = 250_000
  config.input.re_sample = 1
  config.input.shard=True

  pp_common = (
      '|vgg_value_range'
      '|onehot(1000, key="{lbl}", key_result="labels")'
      '|keep("image", "labels")'
  )
  config.input.pp = (
      'decode_jpeg_and_inception_crop(224, area_min=8, method="bilinear", antialias=True)'
      '|flip_lr'
      '|randaug(2, 9, increase="True", timm=True)'    
      '|vgg_value_range'
      f'|onehot(1000, key="label", key_result="labels", on=0.9, off={0.1/1000})' # label smoothing
      '|keep("image", "labels")'


  )
  pp_eval = 'decode|resize_small(256, method="area", antialias=True)|central_crop(224)' + pp_common

  # Aggressive pre-fetching because our models here are small, so we not only
  # can afford it, but we also need it for the smallest models to not be
  # bottle-necked by the input pipeline. Play around with it for -L models tho.
  config.input.prefetch = 8
  config.prefetch_to_device = 4

  config.log_training_steps = 50
  config.ckpt_steps = 4000
  config.save_ckpt = True
  config.debug_data = False


  config.input.mixup = dict(p=0.8, fold_in=None)
  config.input.cutmix = dict(alpha=1., beta=1.)
  config.input.switch_prob = 0.5
  config.input.cpu_mixup = False
  config.input.mix_cutmix_shards = 8
  config.input.use_mixup = True

  # Model section
  config.model_name = 'crate'
  config.model = dict(
      variant=arg.variant,
      pool_type='tok',
      drop_path=0., # drop path rate
  )

  # Optimizer section

  # clip gradient
  config.grad_clip_norm = 1.0
  config.optax_name = 'scale_by_adam'
  config.optax = dict(mu_dtype='bfloat16', b1=0.9, b2=0.95)

  # The modified AdaFactor we introduced in https://arxiv.org/abs/2106.04560
  # almost always behaves exactly like adam, but at a fraction of the memory
  # cost (specifically, adam_bf16 = +1.5M, adafactor = +0.5M), hence it is a
  # good idea to try it when you are memory-bound!
  # config.optax_name = 'big_vision.scale_by_adafactor'
  # A good flag to play with when hitting instabilities, is the following:
  # config.optax = dict(beta2_cap=0.95)

  config.lr = arg.base_lr * (arg.batchsize // 256)
  config.wd = 0.1
  config.schedule = dict(warmup_epochs=10,
                         decay_type='cosine',
                         min_lr=1e-6, max_lr=arg.base_lr*(arg.batchsize // 256))
  
  config.lwd = 0.0

  config.model_init = ""
#   config.model_load = {'dont_load': ['head/bias','head/kernel']}
  config.model_load = {'dont_load': ['head/kernel','head/bias','embedding/kernel']}

  # Eval section
  def get_eval(split, dataset='imagenet2012'):
    return dict(
        type='classification',
        data=dict(name=dataset, split=split),
        pp_fn=pp_eval.format(lbl='label'),
        loss_name=config.loss,
        log_epochs=1,  # Very fast O(seconds) so it's fine to run it often.
        cache_final=not arg.runlocal,
    )
  config.evals = {}
  config.evals.val = get_eval('validation')

  config.wandb = dict(
      log_wandb=True,
      wandb_offline=False,
      resume=False,
      project='The name of your project',
      experiment=f'The name of your experiment',
      entity='The name of your entity',
  )


  # Make a few things much smaller for quick local debugging testruns.
  if arg.runlocal:
    config.input.shuffle_buffer_size = 10
    config.input.batch_size = 8
    config.input.cache_raw = False
    config.evals.train.data.split = 'train[:16]'
    config.evals.minival.data.split = 'train[:16]'
    config.evals.val.data.split = 'validation[:16]'
    config.evals.v2.data.split = 'test[:16]'
    config.evals.real.data.split = 'validation[:16]'

  return config