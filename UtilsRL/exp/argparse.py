import argparse
from collections import OrderedDict
from types import ModuleType, FrameType
from typing import Optional, Union, Dict, Any

from numpy import isin, nested_iters
    
from UtilsRL.misc.namespace import NameSpace, NameSpaceMeta
from UtilsRL.misc.chore import safe_eval
from UtilsRL.logger import ColoredLogger


# argparse callbacks
argparse_callbacks = {}
def register_argparse_callback(key, func):
    """
    Register a callback function which is specific to UtilsRL use. 
    """
    argparse_callbacks[key] = func
    
def get_key(_key):
    if _key.startswith("--"):
        return _key[2:]
    elif _key.startswith("-"):
        return _key[1:]
    else:
        return _key
    
def parse_cmd_args(convert=True):
    cmd_parser = argparse.ArgumentParser()
    _, cmd_args = cmd_parser.parse_known_args()
    num = len(cmd_args)//2
    cmd_args = dict(zip([get_key(cmd_args[2*i]) for i in range(num)], [cmd_args[2*i+1] for i in range(num)]))
    args = NameSpace("cmd_args", {}, nested=True) if convert else {}
    for key, value in cmd_args.items():
        keys = key.split(".")
        this = args
        for subkey in keys[:-1]:
            if convert:
                this[subkey] = this.get(subkey, NameSpace(subkey, {}, nested=True))
            else:
                this[subkey] = this.get(subkey, {})
            this = this[subkey]
        this[keys[-1]] = safe_eval(value)
    return args
    
def parse_args(path: Optional[Union[str, dict, ModuleType]] = None, convert=True) -> Union[NameSpace, Dict[str, Any]]:
    """
    Parse args from json file, yaml, python file or plain old dict. 
    Command-line argumnets will be parsed as well, and will be used to overwrite
        the init arguments. 
    
    Args:
        args: can be string or python module object. If it is string, it will be 
            interpreted as the path to the config file. If it is a python module, 
            then its attributes will be extracted to from an argument dict. 
        convert: whether to convert the final argument dict to a Namespace class, 
            which brings lots of convenience. Default to True. 
    """
    cmd_args = parse_cmd_args(convert=convert)
    
    # parse args from config files or modules
    if isinstance(path, str):
        if path.endswith(".json"):
            import json
            with open(path, "r") as fp:
                file_args = json.load(fp)
        elif path.endswith(".yaml") or path.endswith(".yml"):
            import yaml
            with open(path, "r") as fp:
                file_args = yaml.safe_load(fp)
        elif path.endswith(".py"):
            import importlib.util
            spec = importlib.util.spec_from_file_location("config_module", path)
            foo = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(foo)
            file_args = _parse_args_from_module(foo)
    elif isinstance(path, ModuleType):
        file_args = _parse_args_from_module(path)
    elif path is None:
        file_args = {}
    
    file_args = NameSpace("args", file_args, nested=True) if convert else file_args
        
    # update with cmd args
    clogger = ColoredLogger()
    def traverse_add(old, new, current_key=""):
        for new_key, new_value in new.items():
            if new_key not in old:
                clogger.log_str(f"parse_args: key {current_key + new_key} is not in the config file, setting it to {new_value}", type="WARNING")
                old[new_key] = new_value
            elif type(old[new_key]) != type(new_value):
                clogger.log_str(f"parse_args: overwriting key {current_key + new_key} with {new_value}", type="WARNING")
                old[new_key] = new_value
            else:
                if (convert and isinstance(new_value, NameSpaceMeta)) \
                    or (not convert and isinstance(new_value, dict)):
                    traverse_add(old[new_key], new_value, current_key=current_key+new_key+".")
                else:
                    clogger.log_str(f"parse_args: overwriting key {current_key + new_key} with {new_value}", type="WARNING")
                    old[new_key] = new_value
                
    traverse_add(file_args, cmd_args)
    
    # check if there is a callback
    for key in argparse_callbacks:
        _args = file_args
        _keys = key.split(".")
        for k in _keys:
            _args = _args.get(k, None)
            if _args is None:
                break
        else:
            ret = argparse_callbacks[key](_args)
            file_args = update_args(file_args, ret)
    
    return file_args
    
def update_args(args, new_args: Optional[Union[dict, list]] = None):
    """Update the arguments with a flat dict or list of key-value pairs.

    Args:
        args: argument object, which can be dict or `NameSpace`.
        new_args: new (command line) arguments. Can be a list or a dict. 

    Returns:
        The updated argument object. 
    """
    if new_args is None:
        return args
    if isinstance(new_args, list):
        num = len(new_args)//2
        new_args = dict(zip([new_args[2*i] for i in range(num)], [new_args[2*i+1] for i in range(num)]))
    for key, value in new_args.items():
        for _ in range(2):
            if key[0] == "-":
                key = key[1:]
        _key = key.split(".")
        _final = args
        for k in _key[:-1]:
            if k not in _final:
                _final[k] = {}
            _final = _final[k]
        _final[_key[-1]] = safe_eval(value)
    return args
            
    
def _parse_args_from_module(module: ModuleType):
    """
    parse args from a given module, without parsing functions and modules. 
    """
    args = [ i for i in dir(module) if not i.startswith("__") ]
    config = OrderedDict()
    for key in args:
        val = getattr(module, key)
        if type(val).__name__ in ["function", "module"]:
            continue
        else:
            config[key] = val
    
    return config
