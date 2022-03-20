import argparse
from collections import OrderedDict
from types import ModuleType, FrameType
from typing import Optional, OrderedDict, Union, Dict, Any
    
from UtilsRL.misc.namespace import NameSpace, NameSpaceMeta
from UtilsRL.misc.chore import safe_eval


# argparse callbacks
argparse_callbacks = {}
def register_argparse_callback(key, func):
    """
    Register a callback function which is specific to UtilsRL use. 
    """
    argparse_callbacks[key] = func

def parse_args(args: Optional[Union[str, dict, ModuleType]] = None, convert=True) -> Union[NameSpace, Dict[str, Any]]:
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
    if isinstance(args, str):
        if args.endswith(".json"):
            import json
            with open(args, "r") as fp:
                args = json.load(fp)
        elif args.endswith(".yaml") or args.endswith(".yml"):
            import yaml
            with open(args, "r") as fp:
                args = yaml.safe_load(fp)
        elif args.endswith(".py"):
            import importlib.util
            spec = importlib.util.spec_from_file_location("config_module", args)
            foo = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(foo)
            args = _parse_args_from_module(foo)
    elif isinstance(args, ModuleType):
        args = _parse_args_from_module(args)
    elif args is None:
        args = {}
        
    # update with command line arguments
    cmd_parser = argparse.ArgumentParser()
    _, unknown = cmd_parser.parse_known_args()
    args = update_args(args, unknown)
    
    # convert or not
    if not isinstance(args, dict):
        raise TypeError("Unsupported args type: {}".format(type(args)))
    
    # check if there is a callback
    for key in argparse_callbacks:
        _args = args
        _keys = key.split(".")
        for k in _keys:
            _args = _args.get(k, None)
            if _args is None:
                break
        else:
            ret = argparse_callbacks[key](_args)
            args = update_args(args, ret)
    
    if convert: 
        return NameSpace("args", args, nested=convert)
    else:
        return args
    
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
