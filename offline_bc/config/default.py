from typing import List, Optional, Union

import numpy as np

from habitat.config import Config as CN

CONFIG_FILE_SEPARATOR = ','

# -----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# -----------------------------------------------------------------------------
_C = CN()
_C.SEED = 100
_C.CMD_TRAILING_OPTS = []  # store command line options as list of strings

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.MAX_EPISODE_STEPS = 1000

# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.MAX_EPISODE_STEPS = 1000
# _C.DATASET.num_ft_views = 1

def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    :ref:`config_paths` and overwritten by options from :ref:`opts`.

    Args:
        config_paths: List of config paths or string that contains comma
        separated list of config paths.
        opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example, ``opts = ['FOO.BAR',
        0.5]``. Argument can be used for parameter sweeping or quick tests.
    """
    config = _C.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)
            
    if opts:
        config.CMD_TRAILING_OPTS = config.CMD_TRAILING_OPTS + opts
        config.merge_from_list(config.CMD_TRAILING_OPTS)

    config.freeze()
    return config
