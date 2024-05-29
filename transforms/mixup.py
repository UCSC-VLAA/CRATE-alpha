import functools
from typing import Tuple

import einops
import jax

import jax.numpy as jnp

import tensorflow as tf

from transforms.random_erasing import _fill_rectangle


class MixupAndCutmix:
  """Applies Mixup and/or Cutmix to a batch of images.
  - Mixup: https://arxiv.org/abs/1710.09412
  - Cutmix: https://arxiv.org/abs/1905.04899
  Implementaion is inspired by https://github.com/rwightman/pytorch-image-models
  """

  def __init__(self,
               mixup_alpha: float = .8,
               cutmix_alpha: float = 1.,
               prob: float = 1.0,
               switch_prob: float = 0.5,
               label_smoothing: float = 0.1,
               mix_cutmix_shards: int = 1,
               num_classes: int = 1001):
    """Applies Mixup and/or Cutmix to a batch of images.
    Args:
      mixup_alpha (float, optional): For drawing a random lambda (`lam`) from a
        beta distribution (for each image). If zero Mixup is deactivated.
        Defaults to .8.
      cutmix_alpha (float, optional): For drawing a random lambda (`lam`) from a
        beta distribution (for each image). If zero Cutmix is deactivated.
        Defaults to 1..
      prob (float, optional): Of augmenting the batch. Defaults to 1.0.
      switch_prob (float, optional): Probability of applying Cutmix for the
        batch. Defaults to 0.5.
      label_smoothing (float, optional): Constant for label smoothing. Defaults
        to 0.1.
      num_classes (int, optional): Number of classes. Defaults to 1001.
    """
    self.mixup_alpha = mixup_alpha
    self.cutmix_alpha = cutmix_alpha
    self.mix_prob = prob
    self.switch_prob = switch_prob
    self.label_smoothing = label_smoothing
    self.num_classes = num_classes
    self.mode = 'batch'
    self.mixup_enabled = True
    self.mix_cutmix_shards = mix_cutmix_shards

    if self.mixup_alpha and not self.cutmix_alpha:
      self.switch_prob = -1
    elif not self.mixup_alpha and self.cutmix_alpha:
      self.switch_prob = 1

  def __call__(self, *args) :
    return self.distort(args)

  def distort(self, args) :
    """Applies Mixup and/or Cutmix to batch of images and transforms labels.
    Args:
      images (tf.Tensor): Of shape [batch_size, height, width, 3] representing a
        batch of image, or [batch_size, time, height, width, 3] representing a
        batch of video.
      labels (tf.Tensor): Of shape [batch_size, ] representing the class id for
        each image of the batch.
    Returns:
      Tuple[tf.Tensor, tf.Tensor]: The augmented version of `image` and
        `labels`.
    """
    images, labels= args[0]['image'], args[0]['labels']

    batch_size = tf.shape(images)[0]
    images = tf.reshape(images, (self.mix_cutmix_shards ,batch_size // self.mix_cutmix_shards, tf.shape(images)[1], tf.shape(images)[2], tf.shape(images)[3]))
    labels = tf.reshape(labels, (self.mix_cutmix_shards ,batch_size  // self.mix_cutmix_shards, tf.shape(labels)[1]))

    new_images = []
    new_labels = []
    for i  in range(self.mix_cutmix_shards):
      new_image, new_label = self._update_labels(*tf.cond(
          tf.less(
            tf.random.uniform(shape=[], minval=0., maxval=1.0), self.switch_prob
          ), lambda: self._cutmix(images[i], labels[i]), lambda: self._mixup(
            images[i], labels[i])))
      new_images.append(new_image)
      new_labels.append(new_label)
    new_images = tf.reshape(tf.stack(new_images, axis=0), (batch_size, tf.shape(images)[2], tf.shape(images)[3], tf.shape(images)[4]))
    new_labels = tf.reshape(tf.stack(new_labels, axis=0), (batch_size, tf.shape(labels)[2]))
    # augment_cond = tf.less(
    #     tf.random.uniform(shape=[], minval=0., maxval=1.0), self.mix_prob)
    # # pylint: disable=g-long-lambda
    # augment_a = lambda: self._update_labels(*tf.cond(
    #     tf.less(
    #         tf.random.uniform(shape=[], minval=0., maxval=1.0), self.switch_prob
    #     ), lambda: self._cutmix(images, labels), lambda: self._mixup(
    #         images, labels)))
    # augment_b = lambda: (images, labels)
    # pylint: enable=g-long-lambda

    return {'image':new_images, 'labels':new_labels}

  @staticmethod
  def _sample_from_beta(alpha, beta, shape):
    sample_alpha = tf.random.gamma(shape, 1., beta=alpha)
    sample_beta = tf.random.gamma(shape, 1., beta=beta)
    return sample_alpha / (sample_alpha + sample_beta)

  def _cutmix(self, images: tf.Tensor,
              labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Applies cutmix."""
    lam = MixupAndCutmix._sample_from_beta(self.cutmix_alpha, self.cutmix_alpha,
                                           (tf.shape(labels)[0],))

    ratio = tf.math.sqrt(1 - lam)

    batch_size = tf.shape(images)[0]

    if len(images.shape) == 4:
      image_height, image_width = tf.shape(images)[1], tf.shape(images)[2]
      fill_fn = _fill_rectangle
    elif images.shape.rank == 5:
      image_height, image_width = tf.shape(images)[2], tf.shape(images)[3]
      fill_fn = _fill_rectangle_video
    else:
      raise ValueError('Bad image rank: {}'.format(images.shape.rank))

    cut_height = tf.cast(
      ratio * tf.cast(image_height, dtype=tf.float32), dtype=tf.int32)
    cut_width = tf.cast(
      ratio * tf.cast(image_height, dtype=tf.float32), dtype=tf.int32)

    random_center_height = tf.random.uniform(
      shape=[batch_size], minval=0, maxval=image_height, dtype=tf.int32)
    random_center_width = tf.random.uniform(
      shape=[batch_size], minval=0, maxval=image_width, dtype=tf.int32)

    bbox_area = cut_height * cut_width
    lam = 1. - bbox_area / (image_height * image_width)
    lam = tf.cast(lam, dtype=tf.float32)

    images = tf.map_fn(
      lambda x: fill_fn(*x),
      (images, random_center_width, random_center_height, cut_width // 2,
       cut_height // 2, tf.reverse(images, [0])),
      dtype=(
        images.dtype, tf.int32, tf.int32, tf.int32, tf.int32, images.dtype),
      fn_output_signature=tf.TensorSpec(images.shape[1:], dtype=images.dtype))

    return images, labels, lam

  def _mixup(self, images: tf.Tensor,
             labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Applies mixup."""
    lam = MixupAndCutmix._sample_from_beta(self.mixup_alpha, self.mixup_alpha,
                                           (tf.shape(labels)[0],))
    if len(images.shape) == 4:
      lam = tf.reshape(lam, [-1, 1, 1, 1])
    elif images.shape.rank == 5:
      lam = tf.reshape(lam, [-1, 1, 1, 1, 1])
    else:
      raise ValueError('Bad image rank: {}'.format(images.shape.rank))

    lam_cast = tf.cast(lam, dtype=images.dtype)
    images = lam_cast * images + (1. - lam_cast) * tf.reverse(images, [0])

    return images, labels, tf.squeeze(lam)


  def _update_labels(self, images: tf.Tensor, labels: tf.Tensor,
                     lam: tf.Tensor):
    labels_1 = labels
    labels_2 = tf.reverse(labels_1, [0])

    lam = tf.reshape(lam, [-1, 1])
    labels = lam * labels_1 + (1. - lam) * labels_2

    return images, labels



# def mixup(*args, p=0.8, fold_in=None, n=2, **more_things):
#   """Perform mixup https://arxiv.org/abs/1710.09412.
#
#   Args:
#     rng: The random key to use.
#     *things: further arguments are the arrays to be mixed.
#     p: the beta/dirichlet concentration parameter, typically 0.1 or 0.2.
#     fold_in: One of None, "host", "device", or "sample". Whether to sample a
#       global mixing coefficient, one per host, one per device, or one per
#       example, respectively. The latter is usually a bad idea.
#     n: with how many other images an image is mixed. Default mixup is n=2.
#     **more_things: further kwargs are arrays to be mixed.  See also (internal link)
#       for further experiments and investigations.
#
#   Returns:
#     A new rng key. A list of mixed *things. A dict of mixed **more_things.
#   """
#   images, labels, rng = args[0]
#   rng, rng_m = jax.random.split(rng, 2)
#   if fold_in == "host":
#     rng_m = jax.random.fold_in(rng_m, jax.process_index())
#   elif fold_in in ("device", "sample"):
#     rng_m = jax.random.fold_in(rng_m, jax.lax.axis_index("batch"))
#   ashape = (len(images[0]),) if fold_in == "sample" else (1,)
#   alpha = jax.random.dirichlet(rng_m, jnp.array([p]*n), ashape)
#   # Sort alpha values in decreasing order. This avoids destroying examples when
#   # the concentration parameter p is very small, due to Dirichlet's symmetry.
#   alpha = -jnp.sort(-alpha, axis=-1)
#   # def mix(batch):
#   #   if batch is None: return None  # For call-side convenience!
#   #   def mul(a, b):  # B * BHWC -> B111 * BHWC
#   #     return b * jnp.expand_dims(a, tuple(range(1, b.ndim)))
#   #   return sum(mul(alpha[:, i], jnp.roll(batch, i, axis=0)) for i in range(n))
#     # flip_batch = jnp.flip(batch, axis=0)
#     # mix_batch = (batch, flip_batch)
#     # return sum(mul(alpha[:, i], mix_batch[i]) for i in range(n))
#    # return sum(mul(alpha[:, 0], batch), mul(alpha[:, 1], jnp.flip(batch, axis=0)))
#   return rng,  mix(images),  mix(labels)

def mixup(*args,  alpha: float = 0.8, beta: float = 0.8):
  """Perform mixup https://arxiv.org/abs/1710.09412.

  Args:
    rng: The random key to use.

  """
  images, labels, rng = args[0]
  batch_size, height, width, _ = images.shape
  rng, rng_m = jax.random.split(rng, 2)

  # switch between each process/devices?

     #rng_m = jax.random.fold_in(rng_m, jax.process_index())
  #rng_m = jax.random.fold_in(rng_m, jax.lax.axis_index("batch"))

  # if fold_in == "host":
  #rng_m = jax.random.fold_in(rng_m, jax.process_index())
  # elif fold_in in ("device", "sample"):
  rng_m = jax.random.fold_in(rng_m, jax.lax.axis_index("batch"))

  lam = jax.random.beta(rng_m, a=alpha, b=beta, shape=())
  idx = jnp.flip(jnp.arange(batch_size))  # using flip-order to mix

  # rng, rng_id = jax.random.split(rng, 2)
  # rng_id = jax.random.fold_in(rng_id, jax.lax.axis_index("batch"))
  # idx = jax.random.permutation(rng_id, batch_size)

  def mixup(x, y):
    images_a = x
    images_b = x[idx, :, :, :]
    y = lam * y + (1. - lam) * y[idx, :]
    x = lam * images_a + (1. - lam) * images_b
    return x, y
  x, y = mixup(images, labels)
  return rng,  x,  y


def cutmix(
           *args,
           alpha: float = 1.,
           beta: float = 1.,
           split: int = 1):
  """Composing two images by inserting a patch into another image."""

  images, labels, rng = args[0]
  batch_size, height, width, _ = images.shape
  split_batch_size = batch_size // split if split > 1 else batch_size

  # Masking bounding box.
  rng, lam_rng = jax.random.split(rng, num=2)

  # switch between hosts/device
  #if mixup_fold_in:
#  lam_rng = jax.random.fold_in(lam_rng, jax.process_index())
  lam_rng = jax.random.fold_in(lam_rng, jax.lax.axis_index("batch"))


  lam = jax.random.beta(lam_rng, a=alpha, b=beta, shape=())

  cut_rat = jnp.sqrt(1. - lam)
  cut_w = jnp.array(width * cut_rat, dtype=jnp.int32)
  cut_h = jnp.array(height * cut_rat, dtype=jnp.int32)
  box_coords, rng = _random_box(rng, height, width, cut_h, cut_w)
  lam = 1. - (box_coords[2] * box_coords[3] / (height * width))

 # idx = jax.random.permutation(rng, split_batch_size)
  idx = jnp.flip(jnp.arange(split_batch_size)) # using flip-order to mix
  def _cutmix(x, y):
    images_a = x
    images_b = x[idx, :, :, :]
    y = lam * y + (1. - lam) * y[idx, :]
    x = _compose_two_images(images_a, images_b, box_coords)
    return  x, y

  if split <= 1:
    x, y = _cutmix(images, labels)
    return rng,  x, y

  # Apply CutMix separately on each sub-batch. This reverses the effect of
  # `repeat` in datasets.
  images = einops.rearrange(images, '(b1 b2) ... -> b1 b2 ...', b2=split)
  labels = einops.rearrange(labels, '(b1 b2) ... -> b1 b2 ...', b2=split)
  images, labels = jax.vmap(_cutmix, in_axes=1, out_axes=1)(images, labels)
  images = einops.rearrange(images, 'b1 b2 ... -> (b1 b2) ...', b2=split)
  labels = einops.rearrange(labels, 'b1 b2 ... -> (b1 b2) ...', b2=split)
  return rng,  images,  labels


def _random_box(rng: jax.random.PRNGKey,
                height: float,
                width: float,
                cut_h: jax.Array,
                cut_w: jax.Array,
                ) :
  """Sample a random box of shape [cut_h, cut_w]."""
  rng, height_rng, width_rng = jax.random.split(rng, 3)

  #switch between process/device
  #height_rng = jax.random.fold_in(height_rng, jax.process_index())
 # width_rng = jax.random.fold_in(width_rng, jax.process_index())

  height_rng = jax.random.fold_in(height_rng, jax.lax.axis_index("batch"))
  width_rng = jax.random.fold_in(width_rng,  jax.lax.axis_index("batch"))

  i = jax.random.randint(
      height_rng, shape=(), minval=0, maxval=height, dtype=jnp.int32)
  j = jax.random.randint(
      width_rng, shape=(), minval=0, maxval=width, dtype=jnp.int32)
  bby1 = jnp.clip(i - cut_h // 2, 0, height)
  bbx1 = jnp.clip(j - cut_w // 2, 0, width)
  h = jnp.clip(i + cut_h // 2, 0, height) - bby1
  w = jnp.clip(j + cut_w // 2, 0, width) - bbx1
  return jnp.array([bby1, bbx1, h, w]), rng




def _compose_two_images(images: jax.Array,
                        image_permutation: jax.Array,
                        bbox: jax.Array) -> jax.Array:
  """Inserting the second minibatch into the first at the target locations."""
  def _single_compose_two_images(image1, image2):
    height, width, _ = image1.shape
    mask = _window_mask(bbox, (height, width))
    return image1 * (1. - mask) + image2 * mask
  return jax.vmap(_single_compose_two_images)(images, image_permutation)


def _window_mask(destination_box: jax.Array,
                 size: Tuple[int, int]) -> jnp.ndarray:
  """Mask a part of the image."""
  height_offset, width_offset, h, w = destination_box
  h_range = jnp.reshape(jnp.arange(size[0]), [size[0], 1, 1])
  w_range = jnp.reshape(jnp.arange(size[1]), [1, size[1], 1])
  return jnp.logical_and(
      jnp.logical_and(height_offset <= h_range,
                      h_range < height_offset + h),
      jnp.logical_and(width_offset <= w_range,
                      w_range < width_offset + w)).astype(jnp.float32)




# def mix_and_cutmix(imgs: jax.Array,
#                    label: jax.Array,
#                    switch_sampled:jax.Array,
#                    rng_new: jax.random.PRNGKey = None,
#                    mixup_alpha: float = .8,
#                    cutmix_alpha: float = 1.,
#                    switch_prob: float = 0.5):
#   return jax.lax.cond(switch_sampled < switch_prob, mixup,  cutmix,
#                       (imgs, label, rng_new),
#                      )
#   #     rng, imgs, labels = cutmix(rng, imgs, labels, cutmix_alpha)
#   # else:
#   #     rng, imgs, labels = mixup(rng, imgs, labels, p=mixup_alpha)
#   #
#   # return rng,  imgs, labels
#
#
#
#



