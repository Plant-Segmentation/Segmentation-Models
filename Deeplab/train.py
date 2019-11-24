# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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
# ==============================================================================
"""Training script for the DeepLab model.

See model.py for more details and usage.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import six
import tensorflow as tf
from tensorflow.python.ops import math_ops
from deeplab import common
from deeplab import model
from deeplab.datasets import data_generator
from deeplab.utils import train_utils
from deeplab.core import preprocess_utils
import numpy as np
import subprocess
import sys
import shutil
flags = tf.app.flags
FLAGS = flags.FLAGS

# Settings for multi-GPUs/multi-replicas training.

flags.DEFINE_integer('num_clones', 1, 'Number of clones to deploy.') #run on single gpu

flags.DEFINE_boolean('clone_on_cpu', False, 'Use CPUs to deploy clones.')

flags.DEFINE_integer('num_replicas', 1, 'Number of worker replicas.')

flags.DEFINE_integer('startup_delay_steps', 15,
                     'Number of training steps between replicas startup.')

flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then '
    'the parameters are handled locally by the worker.')

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

flags.DEFINE_integer('task', 0, 'The task ID.')

# Settings for logging.

flags.DEFINE_string('train_logdir', "xception/",
                    'Where the checkpoint and logs are stored.')

flags.DEFINE_integer('log_steps', 10,
                     'Display logging information at every log_steps.')

flags.DEFINE_integer('save_interval_secs', 60,
                     'How often, in seconds, we save the model to disk.')

flags.DEFINE_integer('save_summaries_secs', 60,
                     'How often, in seconds, we compute the summaries.')

flags.DEFINE_boolean(
    'save_summaries_images', True,
    'Save sample inputs, labels, and semantic predictions as '
    'images to summary.')

# Settings for profiling.

flags.DEFINE_string('profile_logdir', None,
                    'Where the profile files are stored.')

# Settings for training strategy.

flags.DEFINE_enum('learning_policy', 'step', ['poly', 'step'],
                  'Learning rate policy for training.')

# Use 0.007 when training on PASCAL augmented training set, train_aug. When
# fine-tuning on PASCAL trainval set, use learning rate=0.0001.
flags.DEFINE_float('base_learning_rate', 0.00007, #0.00025,
                   'The base learning rate for model training.')

flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                   'The rate to decay the base learning rate.')

flags.DEFINE_integer('learning_rate_decay_step', 2000,
                     'Decay the base learning rate at a fixed step.')

flags.DEFINE_float('learning_power', 0.9,
                   'The power value used in the poly learning policy.')

flags.DEFINE_integer('training_number_of_steps', int(20000),
                     'The number of steps used for training')

flags.DEFINE_float('momentum', 0.9, 'The momentum value to use')

# When fine_tune_batch_norm=True, use at least batch size larger than 12
# (batch size more than 16 is better). Otherwise, one could use smaller batch
# size and set fine_tune_batch_norm=False.
flags.DEFINE_integer('train_batch_size', 2,
                     'The number of images in each batch during training.')

# For weight_decay, use 0.00004 for MobileNet-V2 or Xcpetion model variants.
# Use 0.0001 for ResNet model variants.
flags.DEFINE_float('weight_decay', 0.0001,
                   'The value of the weight decay for training.')

flags.DEFINE_list('train_crop_size', '513,513',
                  'Image crop size [height, width] during training.')

flags.DEFINE_float(
    'last_layer_gradient_multiplier', 1.0,
    'The gradient multiplier for last layers, which is used to '
    'boost the gradient of last layers if the value > 1.')

flags.DEFINE_boolean('upsample_logits', True,
                     'Upsample logits during training.')

flags.DEFINE_boolean('calculate_iou', True,
                     'Calculate Mean IoU during training.')

# Hyper-parameters for NAS training strategy.

flags.DEFINE_float(
     'drop_path_keep_prob', 1.0,
    'Probability to keep each path in the NAS cell when training.')
# Settings for fine-tuning the network.

flags.DEFINE_string('tf_initial_checkpoint', 'deeplabv3_pascal_trainval/model.ckpt', #'ckpt/model.ckpt-58337', #
                    'The initial checkpoint in tensorflow format.')

# Set to False if one does not want to re-use the trained classifier weights.
flags.DEFINE_boolean('initialize_last_layer', False, 
                     'Initialize the last layer.')

flags.DEFINE_boolean('last_layers_contain_logits_only', True,
                     'Only consider logits as last layers or not.')

flags.DEFINE_integer('slow_start_step', 40000,
                     'Training model with small learning rate for few steps.')

flags.DEFINE_float('slow_start_learning_rate', 0.0001,#0.00025,
                   'Learning rate employed during slow start.')

# Set to True if one wants to fine-tune the batch norm parameters in DeepLabv3.
# Set to False and use small batch size to save GPU memory.
flags.DEFINE_boolean('fine_tune_batch_norm', False, 
                     'Fine tune the batch norm parameters or not.')

flags.DEFINE_float('min_scale_factor', 1.0,
                   'Mininum scale factor for data augmentation.')

flags.DEFINE_float('max_scale_factor', 1.0,
                   'Maximum scale factor for data augmentation.')

flags.DEFINE_float('scale_factor_step_size', 1.0, 
                   'Scale factor step size for data augmentation.')

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during training/evaluation.
flags.DEFINE_multi_integer('atrous_rates', [6, 12, 18],
                           'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

# Hard example mining related flags.
flags.DEFINE_integer(
    'hard_example_mining_step', 60000,
    'The training step in which exact hard example mining kicks off. Note we '
    'gradually reduce the mining percent to the specified '
    'top_k_percent_pixels. For example, if hard_example_mining_step=100K and '
    'top_k_percent_pixels=0.25, then mining percent will gradually reduce from '
    '100% to 25% until 100K steps after which we only mine top 25% pixels.')


flags.DEFINE_float(
    'top_k_percent_pixels', 0.7,
    'The top k percent pixels (in terms of the loss values) used to compute '
    'loss during training. This is useful for hard pixel mining.')

# Quantization setting.
flags.DEFINE_integer(
    'quantize_delay_step', -1,
    'Steps to start quantized training. If < 0, will not quantize model.')

# Dataset settings.
flags.DEFINE_string('dataset', 'category4',
                    'Name of the segmentation dataset.')

flags.DEFINE_string('train_split', 'train',
                    'Which split of the dataset to be used for training')

flags.DEFINE_string('dataset_dir', './deeplab/datasets/Plant-data/tfrecord', 'Where the dataset reside.')

min_loss = 1

def _build_deeplab(iterator, outputs_to_num_classes, ignore_label):
  """Builds a clone of DeepLab.

  Args:
    iterator: An iterator of type tf.data.Iterator for images and labels.
    outputs_to_num_classes: A map from output type to the number of classes. For
      example, for the task of semantic segmentation with 21 semantic classes,
      we would have outputs_to_num_classes['semantic'] = 21.
    ignore_label: Ignore label.
  """
  samples = iterator.get_next()

  print("num of classes: ", outputs_to_num_classes)
  print("ignore labels: ", ignore_label)

  # Add name to input and label nodes so we can add to summary.
  samples[common.IMAGE] = tf.identity(samples[common.IMAGE], name=common.IMAGE)
  samples[common.LABEL] = tf.identity(samples[common.LABEL], name=common.LABEL)

  model_options = common.ModelOptions(
      outputs_to_num_classes=outputs_to_num_classes,
      crop_size=[int(sz) for sz in FLAGS.train_crop_size],
      atrous_rates=FLAGS.atrous_rates,
      output_stride=FLAGS.output_stride)

  outputs_to_scales_to_logits = model.multi_scale_logits(
      samples[common.IMAGE],
      model_options=model_options,
      image_pyramid=FLAGS.image_pyramid,
      weight_decay=FLAGS.weight_decay,
      is_training=True,
      fine_tune_batch_norm=FLAGS.fine_tune_batch_norm,
      nas_training_hyper_parameters={
          'drop_path_keep_prob': FLAGS.drop_path_keep_prob,
          'total_training_steps': FLAGS.training_number_of_steps,
      })

  # Add name to graph node so we can add to summary.
  output_type_dict = outputs_to_scales_to_logits[common.OUTPUT_TYPE]
  output_type_dict[model.MERGED_LOGITS_SCOPE] = tf.identity(
      output_type_dict[model.MERGED_LOGITS_SCOPE], name=common.OUTPUT_TYPE)

  for output, num_classes in six.iteritems(outputs_to_num_classes):
    train_utils.add_softmax_cross_entropy_loss_for_each_scale(
        outputs_to_scales_to_logits[output],
        samples[common.LABEL],
        num_classes,
        ignore_label,
        loss_weight=1.0,
        upsample_logits=FLAGS.upsample_logits,
        hard_example_mining_step=FLAGS.hard_example_mining_step,
        top_k_percent_pixels=FLAGS.top_k_percent_pixels,
        scope=output)

    # Log the summary
    _log_summaries(samples[common.IMAGE], samples[common.LABEL], num_classes,
                   output_type_dict[model.MERGED_LOGITS_SCOPE])

    # calculate mean iou
    # miou, miou_update_ops = _mean_iou_calc(samples[common.LABEL], num_classes,
    #                output_type_dict[model.MERGED_LOGITS_SCOPE], ignore_label)


def _tower_loss(iterator, num_of_classes, ignore_label, scope, reuse_variable):
  """Calculates the total loss on a single tower running the deeplab model.

  Args:
    iterator: An iterator of type tf.data.Iterator for images and labels.
    num_of_classes: Number of classes for the dataset.
    ignore_label: Ignore label for the dataset.
    scope: Unique prefix string identifying the deeplab tower.
    reuse_variable: If the variable should be reused.

  Returns:
     The total loss for a batch of data.
  """
  with tf.variable_scope(
      tf.get_variable_scope(), reuse=True if reuse_variable else None):
    _build_deeplab(iterator, {common.OUTPUT_TYPE: num_of_classes}, ignore_label)

  losses = tf.losses.get_losses(scope=scope)
  for loss in losses:
    tf.summary.scalar('Losses/%s' % loss.op.name, loss)

  regularization_loss = tf.losses.get_regularization_loss(scope=scope)
  tf.summary.scalar('Losses/%s' % regularization_loss.op.name,
                    regularization_loss)

  total_loss = tf.add_n([tf.add_n(losses), regularization_loss])
  return total_loss


def _average_gradients(tower_grads):
  """Calculates average of gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list is
      over individual gradients. The inner list is over the gradient calculation
      for each tower.

  Returns:
     List of pairs of (gradient, variable) where the gradient has been summed
       across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads, variables = zip(*grad_and_vars)
    grad = tf.reduce_mean(tf.stack(grads, axis=0), axis=0)

    # All vars are of the same value, using the first tower here.
    average_grads.append((grad, variables[0]))

  return average_grads

# def _mean_iou_calc(labels, num_classes, output, ignore_label):
#   """
#     Calculates and logs the iou score
#   """
#   if FLAGS.calculate_iou:
#       predictions = tf.expand_dims(tf.argmax(output, 3), -1)
#       # import pdb; pdb.set_trace()
#       scaled_labels = tf.image.resize_nearest_neighbor(labels, preprocess_utils.resolve_shape(predictions, 4)[1:3], align_corners=True)

#       scaled_labels = tf.squeeze(scaled_labels, axis=[0,3])
#       predictions = tf.squeeze(predictions, axis=[0,3])
#       # labels_one_hot = tf.one_hot(scaled_labels, num_classes, on_value=0, off_value=1)

#       weights = tf.to_float(tf.not_equal(scaled_labels, ignore_label))
#       scaled_labels = tf.where(tf.equal(scaled_labels, ignore_label), tf.zeros_like(scaled_labels), scaled_labels)

#       predictions = tf.reshape(predictions, [-1,])
#       scaled_labels = tf.reshape(scaled_labels, [-1,])
#       miou, update_op = tf.metrics.mean_iou(predictions, scaled_labels, num_classes, weights=weights)
#       tf.summary.scalar('MeanIoU', miou)
#       return miou, update_op


def _log_summaries(input_image, label, num_of_classes, output):
  """Logs the summaries for the model.

  Args:
    input_image: Input image of the model. Its shape is [batch_size, height,
      width, channel].
    label: Label of the image. Its shape is [batch_size, height, width].
    num_of_classes: The number of classes of the dataset.
    output: Output of the model. Its shape is [batch_size, height, width].
  """
  # Add summaries for model variables.
  for model_var in tf.model_variables():
    tf.summary.histogram(model_var.op.name, model_var)

  # Add summaries for images, labels, semantic predictions.
  if FLAGS.save_summaries_images:
    print("adding summaries")
    tf.summary.image('samples/%s' % common.IMAGE, input_image)

    # Scale up summary image pixel values for better visualization.
    pixel_scaling = max(1, 255 // num_of_classes)
    summary_label = tf.cast(label * pixel_scaling, tf.uint8)
    tf.summary.image('samples/%s' % common.LABEL, summary_label)

    predictions = tf.expand_dims(tf.argmax(output, 3), -1)
    summary_predictions = tf.cast(predictions * pixel_scaling, tf.uint8)
    tf.summary.image('samples/%s' % common.OUTPUT_TYPE, summary_predictions)


def _train_deeplab_model(iterator, num_of_classes, ignore_label):
  """Trains the deeplab model.

  Args:
    iterator: An iterator of type tf.data.Iterator for images and labels.
    num_of_classes: Number of classes for the dataset.
    ignore_label: Ignore label for the dataset.

  Returns:
    train_tensor: A tensor to update the model variables.
    summary_op: An operation to log the summaries.
  """
  global_step = tf.train.get_or_create_global_step()

  learning_rate = train_utils.get_model_learning_rate(
      FLAGS.learning_policy, FLAGS.base_learning_rate,
      FLAGS.learning_rate_decay_step, FLAGS.learning_rate_decay_factor,
      FLAGS.training_number_of_steps, FLAGS.learning_power,
      FLAGS.slow_start_step, FLAGS.slow_start_learning_rate)

  tf.summary.scalar('learning_rate', learning_rate)

  optimizer = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)

  tower_losses = []
  tower_grads = []
  for i in range(FLAGS.num_clones):
    with tf.device('/gpu:%d' % i):
      print("using gpu")
      # First tower has default name scope.
      name_scope = ('clone_%d' % i) if i else ''
      with tf.name_scope(name_scope) as scope:
        loss = _tower_loss(
            iterator=iterator,
            num_of_classes=num_of_classes,
            ignore_label=ignore_label,
            scope=scope,
            reuse_variable=(i != 0))
        tower_losses.append(loss)

  if FLAGS.quantize_delay_step >= 0:
    if FLAGS.num_clones > 1:
      raise ValueError('Quantization doesn\'t support multi-clone yet.')
    tf.contrib.quantize.create_training_graph(
        quant_delay=FLAGS.quantize_delay_step)

  for i in range(FLAGS.num_clones):
    with tf.device('/gpu:%d' % i):
      name_scope = ('clone_%d' % i) if i else ''
      with tf.name_scope(name_scope) as scope:
        grads = optimizer.compute_gradients(tower_losses[i])
        tower_grads.append(grads)

  with tf.device('/cpu:0'):
    grads_and_vars = _average_gradients(tower_grads)

    # Modify the gradients for biases and last layer variables.
    last_layers = model.get_extra_layer_scopes(
        FLAGS.last_layers_contain_logits_only)
    grad_mult = train_utils.get_model_gradient_multipliers(
        last_layers, FLAGS.last_layer_gradient_multiplier)
    if grad_mult:
      grads_and_vars = tf.contrib.training.multiply_gradients(
          grads_and_vars, grad_mult)

    # Create gradient update op.
    grad_updates = optimizer.apply_gradients(
        grads_and_vars, global_step=global_step)

    # Gather update_ops. These contain, for example,
    # the updates for the batch_norm variables created by model_fn.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    update_ops.append(grad_updates)
    update_op = tf.group(*update_ops)

    total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

    # Print total loss to the terminal.
    # This implementation is mirrored from tf.slim.summaries.
    should_log = math_ops.equal(math_ops.mod(global_step, FLAGS.log_steps), 0)
    total_loss = tf.cond(
        should_log,
        lambda: tf.Print(total_loss, [total_loss], 'Total loss is :'),
        lambda: total_loss)
    tf.summary.scalar('total_loss', total_loss)
    with tf.control_dependencies([update_op]):
      train_tensor = tf.identity(total_loss, name='train_op')
      # miou_train_tensor = tf.identitiy(miou, name='train_miou')
      # update_ops_tensor = tf.identity(miou_update_ops, name='train_iou_ops')

    # Excludes summaries from towers other than the first one.
    summary_op = tf.summary.merge_all(scope='(?!clone_)')
    # summary_op = tf.summary.merge_all()
    # print("Summary output: ", summary_op)
    # import pdb; pdb.set_trace()
  return train_tensor, summary_op


def main(unused_argv):

  print("logging params")
  print("Learning rate: ", FLAGS.base_learning_rate)
  print("Momentum: ", FLAGS.momentum)
  print("Weight decay: ", FLAGS.weight_decay)
  print("training steps: ", FLAGS.training_number_of_steps)
  print("Dataset name: ",FLAGS.dataset)
  print("Using dataset for training: ",FLAGS.train_split)
  print("Dataset directory: ",FLAGS.dataset_dir)
  print("batch size: ", FLAGS.train_batch_size)
  print("crop size: ", FLAGS.train_crop_size)
  print("Model variant used: ",FLAGS.model_variant)
  print("Train log directory: ", FLAGS.train_logdir)
  count= 0
  best_val_mean_iou = 0.80
  dir_count = 1

  tf.logging.set_verbosity(tf.logging.INFO)

  tf.gfile.MakeDirs(FLAGS.train_logdir)
  tf.logging.info('Training on %s set', FLAGS.train_split)

  graph = tf.Graph()
  with graph.as_default():
    with tf.device(tf.train.replica_device_setter(ps_tasks=FLAGS.num_ps_tasks)):
      assert FLAGS.train_batch_size % FLAGS.num_clones == 0, (
          'Training batch size not divisble by number of clones (GPUs).')
      clone_batch_size = FLAGS.train_batch_size // FLAGS.num_clones # will be equivalent to train_batch_size

      dataset = data_generator.Dataset(
          dataset_name=FLAGS.dataset,
          split_name=FLAGS.train_split,
          dataset_dir=FLAGS.dataset_dir,
          batch_size=clone_batch_size,
          crop_size=[int(sz) for sz in FLAGS.train_crop_size],
          min_resize_value=FLAGS.min_resize_value,
          max_resize_value=FLAGS.max_resize_value,
          resize_factor=FLAGS.resize_factor,
          min_scale_factor=FLAGS.min_scale_factor,
          max_scale_factor=FLAGS.max_scale_factor,
          scale_factor_step_size=FLAGS.scale_factor_step_size,
          model_variant=FLAGS.model_variant,
          num_readers=1, #check??
          is_training=True,
          should_shuffle=True,
          should_repeat=True)

      train_tensor, summary_op = _train_deeplab_model(
          dataset.get_one_shot_iterator(), dataset.num_of_classes,
          dataset.ignore_label)

      # Soft placement allows placing on CPU ops without GPU implementation.
      session_config = tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=False)

      last_layers = model.get_extra_layer_scopes(
          FLAGS.last_layers_contain_logits_only)
      init_fn = None
      if FLAGS.tf_initial_checkpoint:
        init_fn = train_utils.get_model_init_fn(
            FLAGS.train_logdir,
            FLAGS.tf_initial_checkpoint,
            FLAGS.initialize_last_layer,
            last_layers,
            ignore_missing_vars=True)

      scaffold = tf.train.Scaffold(
          init_fn=init_fn,
          summary_op=summary_op,
      )

      stop_hook = tf.train.StopAtStepHook(
          last_step=FLAGS.training_number_of_steps)

      profile_dir = FLAGS.profile_logdir
      if profile_dir is not None:
        tf.gfile.MakeDirs(profile_dir)

      with tf.contrib.tfprof.ProfileContext(
          enabled=profile_dir is not None, profile_dir=profile_dir):
        with tf.train.MonitoredTrainingSession(
            master=FLAGS.master,
            is_chief=(FLAGS.task == 0),
            config=session_config,
            scaffold=scaffold,
            checkpoint_dir=FLAGS.train_logdir,
            summary_dir=FLAGS.train_logdir,
            log_step_count_steps=FLAGS.log_steps,
            save_summaries_steps=FLAGS.save_summaries_secs,
            save_checkpoint_secs=FLAGS.save_interval_secs,
            hooks=[stop_hook]) as sess:
          while not sess.should_stop():
            count+=1
            training_loss = sess.run([train_tensor])
            if np.isnan(training_loss):
                print("learning rate too high. exiting!")
                exit()

            if count%500==0:
              val_iou = subprocess.check_output([sys.executable, "deeplab/vis.py"])
              val_mean_iou = float(val_iou.decode("utf-8").split('\n')[-2])
              print("Mean IoU on validation dataset: ", val_mean_iou)
              sys.stdout.flush()

              if  val_mean_iou > best_val_mean_iou:
                dir_count+=1
                print("Validation Mean IoU: ", val_mean_iou)
                shutil.copytree('./xception/', './best_chkpt'+str(dir_count)+'/')
                best_val_mean_iou = val_mean_iou


if __name__ == '__main__':
  flags.mark_flag_as_required('train_logdir')
  flags.mark_flag_as_required('dataset_dir')
  tf.app.run()
