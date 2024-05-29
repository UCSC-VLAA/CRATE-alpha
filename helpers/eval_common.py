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

"""Utils for evaluators in general."""

import dataclasses
import functools
import importlib
from typing import Any, Callable
import losses as common_loss
import flax
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

"""Evaluator for the classfication task."""
# pylint: disable=consider-using-from-import
from functools import partial, lru_cache

import datasets.core as ds_core
import datasets.input_pipeline as input_pipeline
import datasets.build_transforms as pp_builder
import helpers.utils as u

import jax
import jax.numpy as jnp
import numpy as np


# To avoid re-compiling the function for every new instance of the same
# evaluator on a different dataset!
@lru_cache(None)
def get_eval_fn(predict_fn, loss_name):
  """Produces eval function, also applies pmap."""
  @partial(jax.pmap, axis_name='batch')
  def _eval_fn(params, batch, labels, mask, rng):
    logits, *_, rng = predict_fn(params, **batch, labels=labels, rng=rng)

    # Ignore the entries with all zero labels for evaluation.
    mask *= labels.max(axis=1)

    losses = getattr(common_loss, loss_name)(
        logits=logits, labels=labels, reduction=False)
    loss = jax.lax.psum(losses * mask, axis_name='batch')

    top1_idx = jnp.argmax(logits, axis=1)
    # Extracts the label at the highest logit index for each image.
    top1_correct = jnp.take_along_axis(
        labels, top1_idx[:, None], axis=1)[:, 0]
    ncorrect = jax.lax.psum(top1_correct * mask, axis_name='batch')
    n = jax.lax.psum(mask, axis_name='batch')
    return ncorrect, loss, n, rng
  return _eval_fn


class Evaluator:
  """Classification evaluator."""

  def __init__(self, predict_fn, data, pp_fn, batch_size, loss_name,
               cache_final=True, cache_raw=False, prefetch=1,
               label_key='labels'):
    data = ds_core.get(**data)
    pp_fn = pp_builder.get_preprocess_fn(pp_fn)
    self.ds, self.steps = input_pipeline.make_for_inference(
        data.get_tfdata(ordered=True), pp_fn, batch_size,
        num_ex_per_process=data.num_examples_per_process(),
        cache_final=cache_final, cache_raw=cache_raw)
    self.data_iter = input_pipeline.start_input_pipeline(self.ds, prefetch)
    self.eval_fn = get_eval_fn(predict_fn, loss_name)
    self.label_key = label_key

  def run(self, params, rng=None):
    """Computes all metrics."""
    ncorrect, loss, nseen = 0, 0, 0
    for step, batch in zip(range(self.steps), self.data_iter):
      labels, mask = batch.pop(self.label_key), batch.pop('_mask')
      batch_ncorrect, batch_losses, batch_n, rng = self.eval_fn(
          params, batch, labels, mask, rng)
      # All results are a replicated array shaped as follows:
      # (local_devices, per_device_batch_size, elem_shape...)
      # with each local device's entry being identical as they got psum'd.
      # So let's just take the first one to the host as numpy.
      ncorrect += np.sum(np.array(batch_ncorrect[0]))
      loss += np.sum(np.array(batch_losses[0]))
      nseen += np.sum(np.array(batch_n[0]))
    yield ('prec@1', ncorrect / nseen)
    yield ('losses', loss / nseen)


def from_config(config, predict_fns,
                write_note=lambda s: s,
                get_steps=lambda key, cfg: cfg[f"{key}_steps"]):
  """Creates a list of evaluators based on `config`."""
  evaluators = []
  specs = config.get("evals", {})

  for name, cfg in specs.items():
    write_note(name)

    # Pop all generic settings off so we're left with eval's kwargs in the end.
    cfg = cfg.to_dict()
    module = cfg.pop("type", name)
    pred_key = cfg.pop("pred", "predict")
    pred_kw = cfg.pop("pred_kw", None)
    prefix = cfg.pop("prefix", f"{name}/")
    logsteps = get_steps("log", cfg)
    for typ in ("steps", "epochs", "examples", "percent"):
      cfg.pop(f"log_{typ}", None)

    # Use same batch_size as eval by default, to reduce fragmentation.
    # TODO: eventually remove all the deprecated names...
    cfg["batch_size"] = cfg.get("batch_size") or config.get("batch_size_eval") or config.get("input.batch_size") or config.get("batch_size")  # pylint: disable=line-too-long

    #module = importlib.import_module(f"helpers.{module}")
    try:
      predict_fn = predict_fns[pred_key]
    except KeyError as e:
      raise ValueError(
          f"Unknown predict_fn '{pred_key}'. Available predict_fns are:\n"
          + "\n".join(predict_fns)) from e
    if pred_kw is not None:
      predict_fn = _CacheablePartial(predict_fn, flax.core.freeze(pred_kw))
    evaluator = Evaluator(predict_fn, **cfg)
    evaluators.append((name, evaluator, logsteps, prefix))

  return evaluators


@dataclasses.dataclass(frozen=True, eq=True)
class _CacheablePartial:
  """partial(fn, **kwargs) that defines hash and eq - to help with jit caches.

  This is particularly common in evaluators when one has many evaluator
  instances that run on difference slices of data.

  Example:

  ```
    f1 = _CacheablePartial(fn, a=1)
    jax.jit(f1)(...)
    jax.jit(_CacheablePartial(fn, a=1))(...)   # fn won't be retraced.
    del f1
    jax.jit(_CacheablePartial(fn, a=1))(...)   # fn will be retraced.
  ```
  """
  fn: Callable[..., Any]
  kwargs: flax.core.FrozenDict

  def __call__(self, *args, **kwargs):
    return functools.partial(self.fn, **self.kwargs)(*args, **kwargs)
