import sys
from typing import Any, Dict
from collections.abc import Mapping

# In this module we implement a custom meta class `NameSpace`, 
# and this meta class overwrites several magic methods, like 
# `__repr__`, `__setattr__`, `__setitem__` to provide a easier 
# access/modification/visualization of the arguments. 
# 
# In theory such meta-programming can be substitute by pure inheritance. 
# For example, we define a class named `NameSpace` and add class methods
# and inherits new classes from `NameSpace`. The defined class methods 
# will be inherited as well. However, one critical flaw of this technique 
# is that it fails for the megic methods -- for example, even if we define 
# a class method `__str__` for `NameSpace`, when we call `str(S)` where `S`
# subclasses `NameSpace`, python will not invoke the user-defined `__str__`, 
# but rather `type.__str__`. So we have to use the metaclass method, even if 
# it means sacrifice of readability. 

def is_dunder(name):
    return (
        len(name) > 4 and
        name[:2] == name[-2:] == '__' and
        name[2] != '_' and
        name[-3] != '_'
    )

class NameSpaceMeta(type): 
    """
    NameSpaceMeta is a metaclass which we defined for simplified argument managing. 
    Classes which uses NameSpaceMeta as meta class or inherits from NameSpace supports: 
    - Both dick-like and attribute-like argument accessing and setting
    - structured and prettified visualizing and printing
    - dict-style manipulation, for example `.key() .values() .items()` and element unpacking. 
    """
    
    def __call__(cls, name: str, maps: Dict[str, Any], nested=True, *, module=None, qualname=None, type=None):
        meta_cls = cls.__class__
        bases = (cls, ) if type is None else (type, cls)
        classdict = {}
        for key, value in maps.items():
            if nested and isinstance(value, dict):
                classdict[key] = NameSpaceMeta.__call__(cls, key, value, module=module, qualname=qualname, type=type)
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

    def __data_as_dict__(cls):
        return {_key: _value for _key, _value in cls.__dict__.items() if not is_dunder(_key)}
        
    def __repr__(cls) -> str:
        return "<{}: {}>".format(cls.__base__.__name__, cls.__name__)

    def __str__(cls, indent=0, sep=[]) -> str:
        _data = {_key: _value for _key, _value in cls.__dict__.items() if not is_dunder(_key)}
        if len(_data) == 0:
            substr = ""
        else:
            substr = "\n"+"\n".join([
                "{}|{}:\t{}".format("".join(["|\t" if i in sep else "\t" for i in range(indent)]), k, NameSpaceMeta.__str__(v, indent+(len(k)+2+7)//8, sep+[indent])) if type(v) is NameSpaceMeta\
                    else "{}|{}: {}".format("".join(["|\t" if i in sep else "\t" for i in range(indent)]), k, v) for k, v in _data.items() 
            ])
        return "{}".format(NameSpaceMeta.__repr__(cls)) + substr

    def __getitem__(cls, __name: str) -> Any:
        return cls.__dict__[__name]
    
    def __setitem__(cls, __name: str, __value: Any) -> None:
        return type.__setattr__(cls, __name, __value)  # we use type.__setattr__ because normally cls.__dict__ doesn't support writing
    
    def __hash__(cls):
        return hash(cls.__module__ + cls.__name__)
    
    def __len__(cls):
        return len(cls.__data_as_dict__())
    
    def __iter__(cls):
        for _key in cls.__data_as_dict__():
            yield _key

    def __contains__(cls, key):
        try:
            cls[key]
        except KeyError:
            return False
        else:
            return True
        
    def __eq__(cls, other):
        return dict(cls.items()) == dict(other.items())
    
    def get(cls, key, default=None):
        try:
            return cls[key]
        except KeyError:
            return default

    def keys(cls):
        return cls.__data_as_dict__().keys()

    def items(cls):
        return cls.__data_as_dict__().items()

    def values(cls):
        return cls.__data_as_dict__().values()
            
    def as_dict(cls):
        def as_dict_helper(v):
            if isinstance(v, NameSpaceMeta):
                return {_key: as_dict_helper(_value) for _key, _value in v.__dict__.items() if not is_dunder(_key)}
            else:
                return v
        return as_dict_helper(cls)
            
        
class NameSpace(metaclass = NameSpaceMeta):
    """
    A sugar which avoids designating metaclass when creating new namespaces. 
    """
    pass
    
