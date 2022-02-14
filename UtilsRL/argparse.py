from collections import OrderedDict
from types import ModuleType, FrameType
from typing import Optional, OrderedDict, Union, Dict, Any
    
from UtilsRL.misc import NameSpace, NameSpaceMeta

def parse_args(args: Optional[Union[str, dict, ModuleType]], convert=True) -> Dict[str, Any]:
    """
    parse args from json file, yaml, python file or plain old dict. 
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
    elif isinstance(args, ModuleType):
        args = _parse_args_from_module(args)
        
    if not isinstance(args, dict):
        raise TypeError("Unsupported args type: {}".format(type(args)))
    
    if convert: 
        return NameSpace("args", args, nested=convert)
    else:
        return args
    
def update_args(args, new_args):
    for key, value in new_args.items():
        _key = key.split("/")
        _final = args
        for k in _key[:-1]:
            _final = _final[k]
        _final[_key[-1]] = value
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
