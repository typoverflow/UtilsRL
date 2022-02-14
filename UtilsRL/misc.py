import inspect
import sys
from typing import Any, Dict

__all__ = (
    "get_var_name", "_is_descriptor", "_is_dunder", "_is_sunder", 
    "NameSpace"
)

def get_var_name(var):
    """Get the name of a variable.
    """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]
        
def _is_descriptor(obj):
    return (
        hasattr(obj, '__get__') or
        hasattr(obj, '__set__') or
        hasattr(obj, '__delete__')
    )
    
def _is_dunder(name):
    return (
        len(name) > 4 and
        name[:2] == name[-2:] == '__' and
        name[2] != '_' and
        name[-3] != '_'
    )
    
def _is_sunder(name):
    return (
        len(name) > 2 and
        name[0] == name[-1] == '_' and
        name[1:2] != '_' and
        name[-2:-1] != '_'
    )

class NameSpaceMeta(type):
    def __new__(cls, name, bases, dct):
        x = super(NameSpaceMeta, cls).__new__(cls, name, bases, dct)
        x._data_ = {k:v for k, v in dct.items() if not _is_dunder(k) and not _is_sunder(k)}
        return x
    
    def __call__(cls, name: str, maps: Dict[str, Any], *, module=None, qualname=None, type=None):
        meta_cls = cls.__class__
        bases = (cls, ) if type is None else (type, cls)
        classdict = {}
        for key, value in maps.items():
            if _is_dunder(key) or _is_sunder(key):
                raise KeyError("NameSpace does not support for dunder keys or sunder keys: {}".format(key))
            if isinstance(value, dict):
                classdict[key] = NameSpace.__call__(key, value, module=module, qualname=qualname, type=type)
            else:
                classdict[key] = value
        new_namespace_cls = NameSpaceMeta.__new__(meta_cls, name, bases, classdict)
        
        if module is None:
            try:
                module = sys._getframe(2).f_globals["__name__"]
            except (AttributeError, ValueError, KeyError) as exc:
                pass
        new_namespace_cls.__module__ = module
        if not qualname is None:
            new_namespace_cls.__qualname__ = qualname
        
        return new_namespace_cls
        
    def __repr__(cls) -> str:
        return "<{}: {}>".format(cls.__base__.__name__, cls.__name__)
    
    def __str__(cls, indent=0) -> str:
        substr = "\n".join([
            "{}|{}:\t{}".format("\t"*(indent), k, NameSpaceMeta.__str__(v, indent+(len(k)+2+7)//8)) if isinstance(v, NameSpaceMeta) \
                else "{}|{}: {}".format("\t"*(indent), k, v) for k, v in cls._data_.items() 
        ]) 
        return "{}\n".format(NameSpaceMeta.__repr__(cls)) + substr
    
    def __setattr__(cls, __name: str, __value: Any) -> None:
        if _is_dunder(__name) or _is_sunder(__name):
            return type.__setattr__(cls, __name, __value)
        else:
            cls._data_[__name] = __value
            
    def __setitem__(cls, __name: str, __value: Any) -> None:
        if _is_dunder(__name) or _is_sunder(__name):
            return type.__setitem__(cls, __name, __value)
        else:
            cls._data_[__name] = __value
    
    def __getattr__(cls, __name: str) -> Any:
        if _is_dunder(__name) or _is_sunder(__name):
            return type.__getattribute__(cls, __name)
        else:
            return cls._data_[__name]
    
    def __getitem__(cls, __name: str) -> Any:
        if _is_dunder(__name) or _is_sunder(__name):
            return type.__getattribute__(cls, __name)
        else:
            return cls._data_[__name]


class NameSpace(metaclass = NameSpaceMeta):
    pass

