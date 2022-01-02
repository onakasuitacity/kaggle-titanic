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
    

def df_info(df):
    tmp = pd.concat({
        "unique": df.nunique(),
        "missing": df.isnull().sum(),
        "dtype": df.dtypes
    }, axis=1)
    return pd.concat([tmp, df.describe().T], axis=1)


def reduce_memory_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != 'object':
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    pass
        else:
            df[col] = df[col].astype('category')
    
    return df