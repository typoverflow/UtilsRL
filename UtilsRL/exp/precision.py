import os
import torch
import numpy as np
from UtilsRL.misc.namespace import NameSpace

from typing import Optional, Dict, Union

def set_precision(args: Optional[Union[torch.dtype, str, int]]):
    if args is None:  # default precision is float32
        args = 32
    if args in {16, 32, 64}:
        if args == 16: prec = "float16"
        elif args == 32: prec = "float32"
        elif args == 64: prec = "float64"
    elif args.lower() in {"float32", "float16", "float32", "float64", "double"}:
        args = args.lower()
        if args == "double":
            prec = "float64"
        elif args == "float": 
            prec = "float32"
        else:
            prec = args
    elif args in {torch.float16, torch.float32, torch.float64}:
        if args == torch.float16: prec = "float16"
        elif args == torch.float32: prec = "float32"
        elif args == torch.float64: prec = "float64"
    else:
        e = f"""
        When setting precision, input format should be either
        - int, 16 or 32 or 64; 
        - str, float16 or float or float32 or float64 or double; 
        - torch.dtype object.
        But got {type(args)} {args}.
        """
        raise TypeError(e)

    np_ftype, torch_ftype = {
        "float16": [np.float16, torch.float16], 
        "float32": [np.float32, torch.float32], 
        "float64": [np.float64, torch.float64]
    }.get(prec)
    torch.set_default_dtype(torch_ftype)
    if prec == "float64":
        torch.set_default_tensor_type(torch.DoubleTensor)
    
    return {
        "UtilsRL.numpy_fp": np_ftype, 
        "UtilsRL.torch_fp": torch_ftype, 
        "UtilsRL.precision": prec
    }
    
    
        