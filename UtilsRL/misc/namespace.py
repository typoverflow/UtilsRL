import sys
import inspect
from typing import Any, Dict

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
    """Meta class for NameSpace. """
    
    def __new__(cls, name, bases, dct):
        x = type.__new__(cls, name, bases, {})
        x._data_ = {k:v for k, v in dct.items() if not _is_dunder(k) and not _is_sunder(k)}
        return x
    
    def __call__(cls, name: str, maps: Dict[str, Any], nested=True, *, module=None, qualname=None, type=None):
        meta_cls = cls.__class__
        bases = (cls, ) if type is None else (type, cls)
        classdict = {}
        for key, value in maps.items():
            if _is_dunder(key) or _is_sunder(key):
                raise KeyError("NameSpace does not support for dunder keys or sunder keys: {}".format(key))
            if nested and isinstance(value, dict):
                classdict[key] = NameSpaceMeta.__call__(key, value, module=module, qualname=qualname, type=type)
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
    
    def __str__(cls, indent=0, sep=[]) -> str:
        # prefix = "".join(["|\t" if i in sep else "\t" for i in range(indent)])
        if len(cls._data_) == 0:
            substr = ""
        else:
            substr = "\n"+"\n".join([
                "{}|{}:\t{}".format("".join(["|\t" if i in sep else "\t" for i in range(indent)]), k, NameSpaceMeta.__str__(v, indent+(len(k)+2+7)//8, sep+[indent])) if isinstance(v, NameSpaceMeta) \
                    else "{}|{}: {}".format("".join(["|\t" if i in sep else "\t" for i in range(indent)]), k, v) for k, v in cls._data_.items() 
            ])
        return "{}".format(NameSpaceMeta.__repr__(cls)) + substr
    
    def __setattr__(cls, __name: str, __value: Any) -> None:
        if _is_dunder(__name) or _is_sunder(__name):
            return type.__setattr__(cls, __name, __value)
        else:
            cls._data_[__name] = __value
            
    def __setitem__(cls, __name: str, __value: Any) -> None:
        if _is_dunder(__name) or _is_sunder(__name):
            raise KeyError("NameSpace does not support for dunder keys or sunder keys: {}".format(__name))
        else:
            cls._data_[__name] = __value
    
    def __getattr__(cls, __name: str) -> Any:
        if __name in ["items", "keys", "values"]:
            return type.__getattribute__(cls._data_, __name)
        if _is_dunder(__name) or _is_sunder(__name):
            return type.__getattribute__(cls, __name)
        else:
            return cls._data_[__name]
    
    def __getitem__(cls, __name: str) -> Any:
        if _is_dunder(__name) or _is_sunder(__name):
            return type.__getattribute__(cls, __name)
        else:
            return cls._data_[__name]
        
    def __eq__(cls, __obj: Any) -> bool:
        if type(cls) != NameSpaceMeta or type(cls) != type(__obj):
            return False
        if cls._data_.keys() != __obj._data_.keys():
            return False
        for key, value in cls._data_.items():
            if value != __obj._data_[key]:
                return False
        return True
        
    def __contains__(cls, __obj):
        return __obj in cls._data_
    
    def __hash__(cls):
        return hash(cls.__module__ + cls.__name__)

    def __add__(cls, __obj):
        if not isinstance(__obj, NameSpaceMeta):
            raise TypeError("unsupported operand type(s) for +: '{}' and '{}'".format(type(cls), type(__obj)))
        ret = cls.__bases__[0]("unnamed", {}, nested=True)
        for k, v in cls._data_.items():
            ret[k] = v
        for k, v in __obj._data_.items():
            ret[k] = v
        return ret

    def keys(cls):
        return cls._data_.keys()
    
    def values(cls):
        return cls._data_.values()
    
    def items(cls):
        return cls._data_.items()

    def get(cls, __key, __default=None):
        return cls._data_.get(__key, __default)
    

class NameSpace(metaclass = NameSpaceMeta):
    """So that we can inherit from this class, rather than
        designating the meta class to NameSpaceMeta for each scope. 
    """
    pass
