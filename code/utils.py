import os
import joblib
import random
import numpy as np
import time
import datetime as dt
from pathlib import Path
import logging
from contextlib import contextmanager


LOGDIR = Path("./out") / f"{dt.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
LOGDIR.mkdir(exist_ok=True, parents=True)


def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def init_logger(path=LOGDIR/"train.log"):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    log_fmt = logging.Formatter(
        "%(asctime)s %(lineno)d [%(levelname)s] %(message)s",
        datefmt='%Y/%m/%d %H:%M:%S'
    )

    handler1 = logging.StreamHandler()
    handler1.setLevel(logging.INFO)
    handler1.setFormatter(log_fmt)
    handler2 = logging.FileHandler(path)
    handler2.setLevel(logging.DEBUG)
    handler2.setFormatter(log_fmt)

    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


@contextmanager
def timer(name, logger):
    s = time.time()
    logger.info(f"[{name}] start.")
    yield
    logger.info(f"[{name}] done in {time.time() - s:.3f} seconds.")