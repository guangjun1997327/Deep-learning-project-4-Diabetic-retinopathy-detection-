import logging
import gin
import wandb
import ray
from ray import tune

from input_pipeline.datasets import load
from models.vgg_likemodel import vgg_like
from train import Trainer
from utils import utils_params, utils_misc

wandb.login()
# def train_func(config):
#     # Hyperparameters
#     bindings = []
#     for key, value in config.items():
#         bindings.append(f'{key}={value}')
#
#     # generate folder structures
#     run_paths = utils_params.gen_run_folder(','.join(bindings))
#
#     # set loggers
#     utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)
#
#     # gin-config
#     gin.parse_config_files_and_bindings(['/absolute/path/to/configs/config.gin'], bindings) # change path to absolute path of config file
#     utils_params.save_config(run_paths['path_gin'], gin.config_str())
#
#     # setup pipeline
#     ds_train, ds_val, ds_test, ds_info = load()
#
#     # model
#     model = vgg_like(input_shape=ds_info.features["image"].shape, n_classes=ds_info.features["label"].num_classes)
#
#     trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)
#     for val_accuracy in trainer.train():
#         tune.report(val_accuracy=val_accuracy)
#
#
# ray.init(num_cpus=10, num_gpus=1)
# analysis = tune.run(
#     train_func, num_samples=2, resources_per_trial={"cpu": 10, "gpu": 1},
#     config={
#         "Trainer.total_steps": tune.grid_search([1e4]),
#         "vgg_like.base_filters": tune.choice([8, 16]),
#         "vgg_like.n_blocks": tune.choice([2, 3, 4, 5]),
#         "vgg_like.dense_units": tune.choice([32, 64]),
#         "vgg_like.dropout_rate": tune.uniform(0, 0.9),
#     })
#
# print("Best config: ", analysis.get_best_config(metric="val_accuracy", mode="max"))
#
# # Get a dataframe for analyzing trial results.
# df = analysis.dataframe()
def tune():
  run = wandb.init(project='sweep')
  learning_rate  =  wandb.config.lr
  total_step = wandb.config.total_step
  drop_rate = wandb.config.drop_rate
  #model parameter
  dense = wandb.config.dense
  channel_4 = wandb.config.channel_4
  channel_3 = wandb.config.channel_3
  channel_2 = wandb.config.channel_2
  channel_1 = wandb.config.channel_1
  layer_4 = wandb.config.layer_4
  layer_3 = wandb.config.layer_3
  archt = ((2, 3, channel_1), (2, 3, channel_2), (layer_3, 3, channel_3), (layer_4, 3, channel_4))
  modelv = vgg_like((256, 256, 3), archt, 2, dense,drop_rate)
  trainer = Trainer(modelv, trainds, valds, info, total_step, 100,learning_rate)
  for _ in trainer.train():
      continue


sweep_configuration = {
  'method': 'grid',
  'name': 'sweep',
  'metric': {'goal': 'maximize', 'name': 'val_acc'},
  'parameters':
    {
      'total_step': {'values': [1500, 2000, 2500, 4000]},
      'lr': {'values': [0.001, 0.0001]},
      'dense': {'values': [64, 128]},
      'drop_rate': {'values': [0.4, 0.5, 0.6]},
      'channel_1': {'values': [8, 16]},
      'channel_2': {'values': [16, 32]},
      'channel_3': {'values': [32, 64]},
      'channel_4': {'values': [64, 128]},
      'layer_3': {'values': [2, 3]},
      'layer_4': {'values': [2, 3]}

    }
}

sweep_1 = wandb.sweep(sweep=sweep_configuration, project='sweep')
wandb.agent(sweep_1, function=tune, count=50)