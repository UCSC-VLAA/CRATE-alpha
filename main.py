import functools
import importlib
import os

import flax
import jax.random
import optax
from absl import app
from absl import flags
from absl import logging
from ml_collections import config_flags
import tensorflow as tf

from clu import parameter_overview
from tensorflow.io import gfile
import multiprocessing.pool

import losses
import optim
from datasets import input_pipeline
from datasets.input_pipeline import shard_and_put
from helpers import utils, eval_common
from helpers.utils import *
from transforms.mixup import mixup, cutmix, MixupAndCutmix

try:
    import wandb
    has_wandb = True
   # os.environ["WANDB_API_KEY"] = YOUR_KEY_HERE
except ImportError:
    has_wandb = False
    print('please install wandb')


# prepare config
from optim import replace_frozen, steps

config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)

flags.DEFINE_string("workdir", default=None, help="Work unit directory.")
flags.DEFINE_boolean("cleanup", default=False,
                     help="Delete workdir (only) after successful completion.")

# Adds jax flags to the program.
jax.config.parse_flags_with_absl()



def main(argv):
  del argv
  jax.distributed.initialize()

  tf.config.experimental.set_visible_devices([], "GPU")
  print('please check the region of dataset and workdir:', os.environ['TFDS_DATA_DIR'], flags.FLAGS.workdir)
  print('if the region is different from TPU VM, please change it!!!!!')
  time.sleep(10)
  config = flags.FLAGS.config
  workdir = flags.FLAGS.workdir


  logging.info(
      f"\u001b[33mHello from process {jax.process_index()} holding "
      f"{jax.local_device_count()}/{jax.device_count()} devices and "
      f"writing to workdir {workdir}.\u001b[0m")

  if config.wandb.log_wandb:
      if has_wandb and jax.process_index() == 0:
          if config.wandb.wandb_offline:
              os.environ["WANDB_MODE"] = 'offline'
          else:
              wandb.init(project=str(config.wandb.project), name=str(config.wandb.experiment), entity=str(config.wandb.entity), resume=config.wandb.resume)
              wandb.config.update(dict(config))
      else:
          logging.warning("You've requested to log metrics to wandb but package not found. "
                          "Metrics not being logged to wandb, try `pip install wandb`")

  save_ckpt_path = None
  if workdir:  # Always create if requested, even if we may not write into it.
      gfile.makedirs(workdir)
      save_ckpt_path = os.path.join(workdir, "checkpoint.npz")

  # The pool is used to perform misc operations such as logging in async way.
  pool = multiprocessing.pool.ThreadPool()


  # This seed makes the Jax part of things (like model init) deterministic.
  # However, full training still won't be deterministic, for example due to the
  # tf.data pipeline not being deterministic even if we would set TF seed.
  # See (internal link) for a fun read on what it takes.
  rng = jax.random.PRNGKey(config.get("seed", 0))
  np.random.seed(config.get("seed", 0))

  # These functions do more stuff internally, for OSS release we mock them by
  # trivial alternatives in order to minize disruptions in the code.
  xid, wid = -1, -1
  fillin = lambda s: s

  def info(s, *a):
      logging.info("\u001b[33mNOTE\u001b[0m: " + s, *a)

  def write_note(note):
      if jax.process_index() == 0:
          info("%s", note)

  write_note("Initializing...")

  batch_size = config.input.batch_size
  if batch_size % jax.device_count() != 0:
      raise ValueError(f"Batch size ({batch_size}) must "
                       f"be divisible by device number ({jax.device_count()})")
  info("Global batch size %d on %d hosts results in %d local batch size. With "
       "%d dev per host (%d dev total), that's a %d per-device batch size.",
       batch_size, jax.process_count(), batch_size // jax.process_count(),
       jax.local_device_count(), jax.device_count(),
       batch_size // jax.device_count())

  metric = BigVisionMetricWriter(xid, wid, workdir, config)

  # Build the dataset
  write_note("Initializing train dataset...")


  train_ds, ntrain_img = input_pipeline.training(config.input)



  # Start prefetching already.
  n_prefetch = config.get("prefetch_to_device", 1)
  train_iter = input_pipeline.start_input_pipeline(train_ds, n_prefetch, shard=not config.get('debug_data', False), mix_fn=None)



  total_steps = steps("total", config, ntrain_img, batch_size)
  def get_steps(name, default=ValueError, cfg=config):
    return steps(name, cfg, ntrain_img, batch_size, total_steps, default)

  if config.debug_data:
      for step, batch in zip(range(0 + 1, total_steps + 1), train_iter):
          metric.step_start(step)
          #print(batch['lam'][0]*batch['labels1'][0]+(1-batch['lam'][0])*batch['labels2'][0] == batch['labels'][0])
          if isinstance(batch, tuple):
              wandb_image = [wandb.Image(batch[1]['image'][i]) for i in range(0, 1024, 50)]
              wandb.log({'diff_generated':wandb_image})
              wandb_image = [wandb.Image(batch[0]['image'][i]) for i in range(0, 1024, 50)]
              wandb.log({'orginal': wandb_image})
              if step == 10:
                  exit(0)

          #print(tf.where(batch['labels'][0]!=1e-4))
      exit(0)

  chrono.inform(total_steps=total_steps, global_bs=batch_size,
                  steps_per_epoch=ntrain_img / batch_size,
                  measure=metric.measure, write_note=write_note)

  info("Running for %d steps, that means %f epochs",
       total_steps, total_steps * batch_size / ntrain_img)

  write_note(f"Initializing {config.model_name} model...")
  model_mod = importlib.import_module(f"models.{config.model_name}")
  model = model_mod.Model(
      num_classes=config.num_classes, **config.get("model", {}))

  # We want all parameters to be created in host RAM, not on any device, they'll
  # be sent there later as needed, otherwise we already encountered two
  # situations where we allocate them twice.
  @functools.partial(jax.jit, backend="cpu")
  def init(rng):
      bs = batch_size // jax.device_count()
      if isinstance(train_ds, list):
          shape  = train_ds[0].element_spec["image"].shape[1:]
      else:
          shape = train_ds.element_spec["image"].shape[1:]
      image_size = tuple(shape)
      no_image = jnp.zeros((bs,) + image_size, jnp.float32)
      params = flax.core.unfreeze(model.init(rng, no_image))["params"]

      # Set bias in the head to a low value, such that losses is small initially.
      if "init_head_bias" in config:
          params["head"]["bias"] = jnp.full_like(params["head"]["bias"],
                                                 config["init_head_bias"])

      return params

  rng, rng_init = jax.random.split(rng)
  with chrono.log_timing("z/secs/init"):
      params_cpu = init(rng_init)

  if jax.process_index() == 0:
    num_params = sum(p.size for p in jax.tree_leaves(params_cpu))
    parameter_overview.log_parameter_overview(params_cpu, msg="init params")
    metric.measure("num_params", num_params)

  write_note(f"Initializing {config.optax_name} optimizer...")
  tx, sched_fns = optim.make(config, params_cpu, sched_kw=dict(
      total_steps=total_steps, batch_size=batch_size, data_size=ntrain_img))

  # We jit this, such that the arrays are created on the CPU, not device[0].
  opt_cpu = jax.jit(tx.init, backend="cpu")(params_cpu)
  sched_fns_cpu = [jax.jit(sched_fn, backend="cpu") for sched_fn in sched_fns]


  @functools.partial(jax.pmap, axis_name="batch", donate_argnums=(0, 1))
  def update_fn(params, opt, rng, images, labels):
    """Update step."""

    measurements = {}

    if not config.input.cpu_mixup and config.input.use_mixup:
        rng, rng_switch = jax.random.split(rng, 2)

        # switch between each process/device?
        #rng_switch = jax.random.fold_in(rng_switch, jax.process_index())

        rng_switch = jax.random.fold_in(rng_switch, jax.lax.axis_index("batch"))

        switch_sampled = jax.random.uniform(rng_switch)

        rng, images, labels = jax.lax.cond(
            jnp.less(switch_sampled, jnp.asarray(config.input.switch_prob)),
                                           mixup, cutmix, (images, labels, rng),
                 )
    # Get device-specific losses rng.
    rng, rng_model = jax.random.split(rng, 2)
    rng_model_local = jax.random.fold_in(rng_model, jax.lax.axis_index("batch"))

    rng, rng_drop_path = jax.random.split(rng, 2)
    rng_drop_path_local = jax.random.fold_in(rng_drop_path, jax.lax.axis_index("batch"))

    rngs = {"dropout": rng_model_local,  "drop_path": rng_drop_path_local}

    if config.get('mask_ratio', None):
        rng, rng_random_mask = jax.random.split(rng, 2)
        rng_random_mask = jax.random.fold_in(rng_random_mask, jax.lax.axis_index("batch"))
        rngs['random_mask'] = rng_random_mask

    def loss_fn(params, images, labels):
      logits, _ = model.apply(
          {"params": params}, images,
          train=True, rngs=rngs)
      return getattr(losses, config.loss)(
          logits=logits, labels=labels)

    if config.get('adv', None):
        if config.adv.get('adv_train', False):
            from adv import attack
            rng, rng_adv = jax.random.split(rng)
            rng_adv_local = jax.random.fold_in(rng_adv, jax.lax.axis_index("batch"))

            fgsm_attack = attack.UntargetedAttack(
                attack.PGD(
                    attack.IteratedFGSM(config.adv.step_size),
                    num_steps=config.adv.num_steps,
                    initialize_fn=attack.linf_initialize_fn(config.adv.epsilon),
                    project_fn=attack.linf_project_fn(config.adv.epsilon, bounds=(0., 1.))),
                    loss_fn=attack.untargeted_cross_entropy)
            train_mask = bool(config.get('mask_ratio'))  # Train with mask if mask_ratio is specified in config.
            images = fgsm_attack(model, params, rng_adv_local, images, labels,
                                 train=train_mask,
                                 rngs=rngs if train_mask else None)

    l, grads = jax.value_and_grad(loss_fn)(params, images, labels)
    l, grads = jax.lax.pmean((l, grads), axis_name="batch")
    updates, opt = tx.update(grads, opt, params)
    params = optax.apply_updates(params, updates)

    gs = jax.tree_util.tree_leaves(replace_frozen(config.schedule, grads, 0.))
    measurements["l2_grads"] = jnp.sqrt(sum([jnp.vdot(g, g) for g in gs]))
    ps = jax.tree_util.tree_leaves(params)
    measurements["l2_params"] = jnp.sqrt(sum([jnp.vdot(p, p) for p in ps]))
    us = jax.tree_util.tree_leaves(updates)
    measurements["l2_updates"] = jnp.sqrt(sum([jnp.vdot(u, u) for u in us]))

    return params, opt, rng, l, measurements

  # We do not jit/pmap this function, because it is passed to evaluator that
  # does it later. We output as many intermediate tensors as possible for
  # maximal flexibility. Later `jit` will prune out things that are not needed.
  def predict_fn(params, image, labels=None, rng=None):
    # introduce labels, rng keys for consistency

    logits, out = model.apply({"params": params}, image)
    return logits, out, rng

  def predict_pgd_fn(params, image, labels, pgd_steps=5, rng=None):

    from adv import attack
    rng, rng_adv = jax.random.split(rng)
    rng_adv_local = jax.random.fold_in(rng_adv, jax.lax.axis_index("batch"))

    fgsm_attack = attack.UntargetedAttack(
        attack.PGD(
            attack.IteratedFGSM(config.adv_eval.step_size),
            num_steps=pgd_steps,
            initialize_fn=attack.linf_initialize_fn(config.adv_eval.epsilon),
            project_fn=attack.linf_project_fn(config.adv_eval.epsilon, bounds=(0., 1.))),
        loss_fn=attack.untargeted_cross_entropy)
    image = fgsm_attack(model, params, rng_adv_local, image, labels)



    logits, out = model.apply({"params": params}, image)
    return logits, out, rng

  if config.get('adv', None):
      # add pgd  into predict function
      predict_fns = {"predict": predict_fn,
                     "predict_pgd": predict_pgd_fn}
  else:
      predict_fns = {"predict": predict_fn}


  # Only initialize evaluators when they are first needed.
  @functools.lru_cache(maxsize=None)
  def evaluators():
    return eval_common.from_config(
        config, predict_fns,
        lambda s: write_note(f"Init evaluator: {s}…\n{chrono.note}"),
        lambda key, cfg: get_steps(key, default=None, cfg=cfg),
    )
    # Decide how to initialize training. The order is important.
    # 1. Always resumes from the existing checkpoint, e.g. resumes a finetune job.
    # 2. Resume from a previous checkpoint, e.g. start a cooldown training job.
    # 3. Initialize model from something, e,g, start a fine-tuning job.
    # 4. Train from scratch.
  resume_ckpt_path = None
  if save_ckpt_path and gfile.exists(save_ckpt_path):
      resume_ckpt_path = save_ckpt_path
  elif config.get("resume"):
      resume_ckpt_path = fillin(config.resume)
  if resume_ckpt_path:
      write_note("Resume training from checkpoint...")
      checkpoint = {
          "params": params_cpu,
          "opt": opt_cpu,
          "chrono": chrono.save(),
      }
      checkpoint_tree = jax.tree_structure(checkpoint)
      loaded = load_checkpoint(checkpoint_tree, resume_ckpt_path)
      # bfloat16 type gets lost when data is saved to disk, so we recover it.
      checkpoint = jax.tree_map(recover_dtype, loaded)
      params_cpu, opt_cpu = checkpoint["params"], checkpoint["opt"]
      chrono.load(checkpoint["chrono"])
  elif config.get("model_init", None):
      write_note(f"Initialize model from {config.model_init}...")
      params_cpu = model_mod.load(
          params_cpu, config.model_init, config.get("model"),
          **config.get("model_load", {}))
      if jax.process_index() == 0:
          parameter_overview.log_parameter_overview(
              params_cpu, msg="restored params")
  elif config.get('load_mae', None):
       write_note(f"Initialize model from MAE Fintuned...")
       params_cpu = load_mae_weights(params_cpu, model_variant=config.model.variant, pretrained=config.get('mae_pretrained'))
       if jax.process_index() == 0:
           parameter_overview.log_parameter_overview(
               params_cpu, msg="restored params")

  write_note("Kicking off misc stuff...")
  first_step = optim.get_count(opt_cpu)
  chrono.inform(first_step=first_step)
  prof = None  # Keeps track of start/stop of profiler state.

  write_note(f"Replicating...\n{chrono.note}")
  params_repl = flax.jax_utils.replicate(params_cpu)
  opt_repl = flax.jax_utils.replicate(opt_cpu)

  rng, rng_loop = jax.random.split(rng, 2)
  rngs_loop = flax.jax_utils.replicate(rng_loop)
  ckpt_writer = None

  write_note(f"First step compilations...\n{chrono.note}")

  if config.get('eval_only', False):
      step = 0
      for (name, evaluator, _, prefix) in evaluators():
          write_note(f"{name} evaluation...\n{chrono.note}")
          with chrono.log_timing(f"z/secs/eval/{name}"):
              for key, value in evaluator.run(params_repl, rng=rngs_loop):
                  metric.measure(f"{prefix}{key}", value)
      exit(0)

  # Using a python integer for step here, because opt.state.step is allocated
  # on TPU during replication.
  for step, batch in zip(range(first_step + 1, total_steps + 1), train_iter):
      metric.step_start(step)
      if isinstance(batch, tuple) and config.get('adv.adv_train', False):
          new_batch = {}
          new_batch["image"] = jnp.concatenate([batch[0]["image"], batch[1]["image"]], axis=1)
          new_batch["labels"] = jnp.concatenate([batch[0]["labels"], batch[1]["labels"]], axis=1)
          batch = new_batch
      with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
          with chrono.log_timing("z/secs/update0", noop=step > first_step + 1):

              params_repl, opt_repl, rngs_loop, loss_value, measurements = update_fn(
                  params_repl, opt_repl, rngs_loop, batch["image"], batch["labels"])

      # On the first host, let's always profile a handful of early steps.
      if jax.process_index() == 0:
          prof = startstop_prof(prof, step, first_step, get_steps("log_training"))

      # Report training progress
      if (utils.itstime(step, get_steps("log_training"), total_steps, host=0)
              or chrono.warmup and jax.process_index() == 0):
          for i, sched_fn_cpu in enumerate(sched_fns_cpu):
              metric.measure(f"global_lr_schedule{i if i else ''}", config.lr*sched_fn_cpu(step - 1))
          l = metric.measure("training_loss", loss_value[0])
          for name, value in measurements.items():
              metric.measure(name, value[0])
          chrono.tick(step)
          if not np.isfinite(l):
              raise RuntimeError(f"The losses became nan or inf somewhere within steps "
                                 f"[{step - get_steps('log_training')}, {step}]")

      # Checkpoint saving
      if (config.get('save_ckpt') and save_ckpt_path and
              (utils.itstime(step, get_steps("ckpt", None), total_steps, host=0) or
               utils.itstime(step, get_steps("keep_ckpt", None), total_steps, host=0))):
          chrono.pause(wait_for=(params_repl, opt_repl))
          checkpointing_timeout(ckpt_writer, config.get("ckpt_timeout", 1))
          # We need to transfer the weights over now or else we risk keeping them
          # alive while they'll be updated in a future step, creating hard to debug
          # memory errors (see (internal link)). Also, takes device 0's params only.
          params_cpu = jax.tree_map(lambda x: np.array(x[0]), params_repl)
          opt_cpu = jax.tree_map(lambda x: np.array(x[0]), opt_repl)

          # Check whether we want to keep a copy of the current checkpoint.
          copy_step = None
          if utils.itstime(step, get_steps("keep_ckpt", None), total_steps):
              copy_step = step

          ckpt = {"params": params_cpu, "opt": opt_cpu, "chrono": chrono.save()}
          ckpt_writer = pool.apply_async(
              save_checkpoint, (ckpt, save_ckpt_path, copy_step))
          chrono.resume()

      for (name, evaluator, log_steps, prefix) in evaluators():
          if itstime(step, log_steps, total_steps, first=log_steps < total_steps,
                       last=False):
              chrono.pause(wait_for=params_repl)
              chrono.tick(step)  # Record things like epoch number, core hours etc.
              write_note(f"{name} evaluation...\n{chrono.note}")
              with chrono.log_timing(f"z/secs/eval/{name}"):
                  for key, value in evaluator.run(params_repl, rng=rngs_loop):
                      metric.measure(f"{prefix}{key}", value)
              chrono.resume()
      metric.step_end()
      if has_wandb and jax.process_index()==0:
          if config.wandb.log_wandb:
              wandb.log(metric.step_metrics, step=step)


  # Run evals after done with training. Running them here guarantees evals
  # will run if job is restarted after writting the last checkpoint and
  # also supports eval only runs (when total_steps or num_epochs is 0).
  metric.step_start(total_steps)
  for (name, evaluator, _, prefix) in evaluators():
      write_note(f"{name} evaluation...\n{chrono.note}")
      with chrono.log_timing(f"z/secs/eval/{name}"):
          for key, value in evaluator.run(params_repl, rng=rngs_loop):
              metric.measure(f"{prefix}{key}", value)

  if has_wandb and jax.process_index() == 0:
      if config.wandb.log_wandb:
          wandb.log(metric.step_metrics, step=total_steps)

  # Always give a chance to stop the profiler, no matter how things ended.
  # TODO: can we also do this when dying of an exception like OOM?
  if jax.process_index() == 0 and prof is not None:
      startstop_prof(prof)

  # Last note needs to happen before the pool's closed =)
  write_note(f"Done!\n{chrono.note}")

  pool.close()
  pool.join()
  metric.close()


  # Make sure all hosts stay up until the end of main.
  sync()
  maybe_cleanup_workdir(workdir, flags.FLAGS.cleanup, info)


if __name__ == "__main__":
  app.run(main)
