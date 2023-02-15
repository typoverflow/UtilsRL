import os
import sys
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

from UtilsRL.monitor import Monitor

from typing import Any, Optional, List, Union, Callable, Dict, Sequence


def log2pd(
    log_dir: str, 
    tag_filter: Callable[[str], bool] = lambda x: True, 
    step_interval: int = 1, 
    max_step: Optional[int] = None, 
    size_guidance: Optional[Dict[str, int]] = None
) -> pd.DataFrame:
    """
    Reads a log and returns a dataframe.
    
    Parameters
    ----------
    log_dir :  Directory to read from. Must be one event file in the log directory.
    tag_filter :  Filter to apply to tags. If this function returns True the event will be filtered out.
    step_interval :  Interval between step values. Default is 1.
    max_step :  Maximum step value to consider. Default is None which means no limit.
    size_guidance :  Used by EventAccumulator.
    
    Returns
    -------
    A pandas.DataFrame containing the events
    """
    file_name = [f for f in os.listdir(log_dir) if "tfevents" in f]
    if len(file_name) != 1:
        raise ValueError(f"There should be only one event file in the log directory, got {file_name}")
    event_path = os.path.join(log_dir, file_name[0])
    ea = event_accumulator.EventAccumulator(event_path, size_guidance=size_guidance)
    ea.Reload()
    filtered_tags = [k for k in ea.Tags()["scalars"] if tag_filter(k)]
    
    ret_pd = pd.DataFrame()
    for tag in filtered_tags:
        event_list = ea.Scalars(tag)
        step = [x.step for x in event_list]
        value = [x.value for x in event_list]
        if "step" in ret_pd and (ret_pd["step"] != step).any():
            raise ValueError(f"Steps are not consistent in the log file, try filter tags with tag_filter.")
        else:
            if "step" not in ret_pd:
                ret_pd["step"] = step
        ret_pd[tag] = value
    ret_pd = ret_pd.iloc[::step_interval]
    if max_step:
        ret_pd = ret_pd[ret_pd["step"] <= max_step]
    return ret_pd


def logs2pd(
    log_dirs: Union[Dict[str, Union[Sequence[str], str]], Sequence[str], str], 
    tag_filter: Callable[[str], bool] = lambda x: True,
    step_interval: int = 1,
    max_step: Optional[int] = None, 
    size_guidance: Optional[Dict[str, int]] = None
) -> pd.DataFrame:
    """
    Extracts tensorboard events from log directories and returns a pandas DataFrame. This function is useful for converting a list of log directories into a pandas DataFrame.
    
    Parameters
    ----------
    log_dirs :  A string or list of strings containing the names of the log directories to be extracted.
    tag_filter :  A function that takes a string and returns True if the string should be included in the output DataFrame.
    step_interval :  The interval between step events. Default is 1.
    max_step :  The maximum number of steps to be extracted. Default is None which means no limit.
    size_guidance :  Used by EventAccumulator.
    
    Returns
    -------
    A unioned pandas DataFrame
    """
    dfs = list()
    name_log_mapping = dict()
    if isinstance(log_dirs, str):
        log_dirs = [log_dirs]
    if isinstance(log_dirs, Sequence):
        name_log_mapping["0"] = log_dirs
    elif isinstance(log_dirs, dict):
        name_log_mapping = log_dirs
    else:
        raise TypeError(f"log_dirs should be a str, list or dict, got {type(log_dirs)}.")
    
    for name, log_dir in Monitor("extracting events").listen(name_log_mapping.items()):
        if isinstance(log_dir, str):
            log_dir = [log_dir]
        for path in log_dir:
            df = log2pd(path, tag_filter, step_interval, max_step, size_guidance)
            df["name"] = name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)
        
        
