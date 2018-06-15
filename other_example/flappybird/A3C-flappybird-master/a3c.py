# -*- coding: utf-8 -*-
import tensorflow as tf
import threading
import numpy as np

import signal
import random
import math
import os
import time

from game_ac_network import GameACFFNetwork, GameACLSTMNetwork
from training_thread import A3CTrainingThread
from rmsprop_applier import RMSPropApplier
from statistics import Statistics

import display as DISPLAY
import visualize as VISUALIZE

flags = tf.app.flags

# MODE
flags.DEFINE_string('mode', 'train', 'Current mode to run [train, display, visualize] (default train)')

# EXPERIMENT
flags.DEFINE_string('experiment_name', 'flappybird', 'Name of the current experiment (for summary)')
flags.DEFINE_string('checkpoint_dir', 'checkpoints', 'Name of the directory for checkpoints')
flags.DEFINE_boolean('use_gpu', False, 'If GPU should be used to speed up the training process')

# AGENT
flags.DEFINE_integer('parallel_agent_size', 16, 'Number of parallel agents')
flags.DEFINE_integer('action_size', 2, 'Numbers of available actions')
flags.DEFINE_string('agent_type', 'FF', 'What type of A3C to train the agent with [FF, LSTM] (default FF)')

# TRAINING
flags.DEFINE_integer('max_time_step', 40000000, 'Maximum training steps')
flags.DEFINE_float('initial_alpha_low', -5, 'LogUniform low limit for learning rate (represents x in 10^x)')
flags.DEFINE_float('initial_alpha_high', -3, 'LogUniform high limit for learning rate (represents x in 10^x)') 
flags.DEFINE_float('gamma', 0.99, 'Discount factor for rewards')
flags.DEFINE_float('entropy_beta', 0.01, 'Entropy regularization constant')
flags.DEFINE_float('grad_norm_clip', 40.0, 'Gradient norm clipping')
flags.DEFINE_integer('random_seed', 1, 'Random seed to use during training')

# OPTIMIZER
flags.DEFINE_float('rmsp_alpha', 0.99, 'Decay parameter for RMSProp')
flags.DEFINE_float('rmsp_epsilon', 0.1, 'Epsilon parameter for RMSProp')
flags.DEFINE_integer('local_t_max', 256, 'Repeat step size')

# LOG
flags.DEFINE_string('log_level', 'FULL', 'Log level [NONE, FULL]')
flags.DEFINE_integer('average_summary', 25, 'How many episodes to average summary over')
flags.DEFINE_integer('performance_log_interval', 1000, 'How often to print current performance (in steps/s)')

# DISPLAY
flags.DEFINE_integer('display_episodes', 50, 'Numbers of episodes to display')
flags.DEFINE_integer('display_time_sleep', 0, 'Sleep time in each state (seconds)')
flags.DEFINE_string('display_log_level', 'MID', 'Display log level - NONE prints end summary, MID prints episode summary and FULL prints for every state [NONE, MID, FULL]')
flags.DEFINE_boolean('display_save_log', False, 'If MID level log should be saved')
flags.DEFINE_boolean('show_max', True, 'If a screenshot of the high score should be plotted')

settings = flags.FLAGS

LOG_FILE = 'summaries/{}-{}'.format(settings.experiment_name, settings.agent_type)

random.seed(settings.random_seed)

def log_uniform(lo, hi, size):
  #logspace中，开始点和结束点是10的幂，生成等比数列,比如(-5,-3,3) 产生10^-5,10^-4,10^-3
  return np.logspace(lo, hi, size)

def train_function(parallel_index):
  global global_t #全局step数
  
  worker = training_threads[parallel_index]
  # set start_time
  start_time = time.time() - wall_t
  worker.set_start_time(start_time)

  #无限训练:直到有停止请求,或者work总step数大于  max_time_step =4000W
  while True:
    if stop_requested:
      break
    if global_t > settings.max_time_step:
      break

    diff_global_t = worker.process(sess, global_t, statistics)
    global_t += diff_global_t
    
    
def signal_handler(signal, frame):
  global stop_requested
  print('You pressed Ctrl+C!')
  stop_requested = True

def write_checkpoint(saver, start_time):
  global global_t
  global settings

  if not os.path.exists(settings.checkpoint_dir):
    os.mkdir(settings.checkpoint_dir)
  if not os.path.exists(settings.checkpoint_dir + '/' + settings.experiment_name + '-' + settings.agent_type):
    os.mkdir(settings.checkpoint_dir + '/' + settings.experiment_name + '-' + settings.agent_type)

  # write wall time
  wall_t = time.time() - start_time
  wall_t_fname = settings.checkpoint_dir + '/' + settings.experiment_name + '-' + settings.agent_type + '/' + 'wall_t.' + str(
    global_t)
  with open(wall_t_fname, 'w') as f:
    f.write(str(wall_t))

  saver.save(sess,
             settings.checkpoint_dir + '/' + settings.experiment_name + '-' + settings.agent_type + '/' 'checkpoint',
             global_step=global_t)



if not settings.mode == 'display' and not settings.mode == 'visualize':
  device = "/cpu:0"
  if settings.use_gpu:
    device = "/gpu:0"

  #初始化N_worker个学习率              # low = -5  high = -3  parallel_agent_size =16
  initial_learning_rates = log_uniform(settings.initial_alpha_low,
                                        settings.initial_alpha_high,
                                        settings.parallel_agent_size)
  global_t = 0

  stop_requested = False
  #===================================
  #实例化global网络,定义ac网络结构
  if settings.agent_type == 'LSTM':
    global_network = GameACLSTMNetwork(settings.action_size, -1, device)
  else:
    global_network = GameACFFNetwork(settings.action_size, -1, device)




  learning_rate_input = tf.placeholder("float")

  #初始化优化器,这里自己定义了一个优化器
  grad_applier = RMSPropApplier(learning_rate = learning_rate_input,
                                decay = settings.rmsp_alpha, #0.99
                                momentum = 0.0,
                                epsilon = settings.rmsp_epsilon, #0.1
                                clip_norm = settings.grad_norm_clip, #40
                                device = device)

  #初始化worker网络,每个worker的学习率不同
  training_threads = []
  for i in range(settings.parallel_agent_size):
    training_thread = A3CTrainingThread(i, 
                                        global_network, 
                                        initial_learning_rates[i],
                                        learning_rate_input, 
                                        grad_applier, 
                                        settings.max_time_step, #40000000, 'Maximum training steps'
                                        device,
                                        settings.action_size,
                                        settings.gamma,
                                        settings.local_t_max,  #256, 'Repeat step size'
                                        settings.entropy_beta,
                                        settings.agent_type, #FF
                                        settings.performance_log_interval,
                                        settings.log_level,
                                        settings.random_seed)

    training_threads.append(training_thread)

  # prepare session
  sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                          allow_soft_placement=True))

  init = tf.global_variables_initializer()
  sess.run(init)

  # Statistics summary writer
  summary_writer = tf.summary.FileWriter(LOG_FILE, sess.graph)
  statistics = Statistics(sess, summary_writer, settings.average_summary)

  if settings.agent_type == 'LSTM':
    agent = settings.agent_type
  else:
    agent = 'FF'

  # init or load checkpoint with saver
  saver = tf.train.Saver()
  checkpoint = tf.train.get_checkpoint_state(settings.checkpoint_dir + '/' + settings.experiment_name + '-' + settings.agent_type)
  if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("checkpoint loaded:", checkpoint.model_checkpoint_path)
    tokens = checkpoint.model_checkpoint_path.split("-")
    # set global step
    global_t = int(tokens[2])
    print(">>> global step set: ", global_t)
    # set wall time
    wall_t_fname = settings.checkpoint_dir + '/' + settings.experiment_name + '-' + settings.agent_type + '/' + 'wall_t.' + str(global_t)
    with open(wall_t_fname, 'r') as f:
      wall_t = float(f.read())
    print "Continuing experiment {} with agent type {} at step {}".format(settings.experiment_name, agent, global_t)

  else:
    print("Could not find old checkpoint")
    # set wall time
    wall_t = 0.0

    print "Starting experiment {} with agent type {}".format(settings.experiment_name, agent)

  train_threads = []
  #启动工作线程,函数train_function,传入线程索引i.
  for i in range(settings.parallel_agent_size):
    train_threads.append(threading.Thread(target=train_function, args=(i,)))
    
  signal.signal(signal.SIGINT, signal_handler)

  # set start time
  start_time = time.time() - wall_t

  for t in train_threads:
    t.start()

  print('Press Ctrl+C to stop')
  signal.pause()

  print('Now saving data. Please wait')
    
  for t in train_threads:
    t.join()

  write_checkpoint(saver=saver, start_time=start_time)

elif settings.mode == 'display':
  DISPLAY.display(settings.experiment_name,
                  settings.rmsp_alpha,
                  settings.rmsp_epsilon,
                  settings.grad_norm_clip,
                  settings.agent_type,
                  settings.action_size,
                  settings.random_seed,
                  settings.checkpoint_dir + '/' + settings.experiment_name + '-' + settings.agent_type,
                  settings.display_time_sleep,
                  settings.display_episodes,
                  settings.display_log_level,
                  settings.display_save_log,
                  settings.show_max)

elif settings.mode == 'visualize':
  print 'viz'
  VISUALIZE.visualize(settings.experiment_name,
                      settings.rmsp_alpha,
                      settings.rmsp_epsilon,
                      settings.grad_norm_clip,
                      settings.agent_type,
                      settings.action_size,
                      settings.random_seed,
                      settings.checkpoint_dir + '/' + settings.experiment_name + '-' + settings.agent_type)

