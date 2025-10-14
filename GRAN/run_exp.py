import os
import sys
import torch
import logging
import traceback
import numpy as np
from pprint import pprint
import time
import json

from runner import *
from utils.logger import setup_logging
from utils.arg_helper import parse_arguments, get_config
torch.set_printoptions(profile='full')


def main():
  args = parse_arguments()
  config = get_config(args.config_file, is_test=args.test)
  np.random.seed(config.seed)
  torch.manual_seed(config.seed)
  torch.cuda.manual_seed_all(config.seed)
  config.use_gpu = config.use_gpu and torch.cuda.is_available()

  # ensure save_dir exists
  os.makedirs(config.save_dir, exist_ok=True)

  # log info
  log_file = os.path.join(config.save_dir, "log_exp_{}.txt".format(config.run_id))
  logger = setup_logging(args.log_level, log_file)
  logger.info("Writing log file to {}".format(log_file))
  logger.info("Exp instance id = {}".format(config.run_id))
  logger.info("Exp comment = {}".format(args.comment))
  logger.info("Config =")
  print(">" * 80)
  pprint(config)
  print("<" * 80)

  # Run the experiment
  try:
    runner = eval(config.runner)(config)
    if not args.test:
      # === timer: training wall time ===
      t0 = time.perf_counter()
      runner.train()
      if config.use_gpu and torch.cuda.is_available():
        torch.cuda.synchronize()
      train_wall_seconds = time.perf_counter() - t0
      logger.info(f"[timer] train_wall_seconds={train_wall_seconds:.3f}")

      # save to file
      with open(os.path.join(config.save_dir, "train_time.json"), "w") as f:
        json.dump({"train_wall_seconds": round(train_wall_seconds, 3)}, f)

      if hasattr(config, "test") and getattr(config.test, "run_after_train", False):
        import gc
        del runner
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(0.5)
        config.test.test_model_dir = config.save_dir
        config.test.test_model_name = "model_last.pth"
        logger.info(f"[after-train] run test() with {config.test.test_model_dir}/{config.test.test_model_name}")

        runner = eval(config.runner)(config)
        runner.test()
    else:
      runner.test()
  except Exception:
    logger.error(traceback.format_exc())
    raise

  sys.exit(0)



if __name__ == "__main__":
  main()
