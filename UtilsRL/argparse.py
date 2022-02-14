from UtilsRL.misc import NameSpace

from collections import OrderedDict
from types import ModuleType, FrameType
from typing import Optional, OrderedDict, Union, Dict, Any

class ParserError(Exception):
    pass

    
def parse_args(args: Optional[Union[str, dict, ModuleType]]) -> Dict[str, Any]:
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
        raise ParserError("Unsupported args type: {}".format(type(args)))
    
    return NameSpace("args", args)
    
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

if __name__ == "__main__":
    import tests.test as test
    file = "./tests/test.json"
    print(parse_args(file))