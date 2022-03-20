import os

from UtilsRL.misc.namespace import NameSpace
from UtilsRL.logger import ColoredLogger
from UtilsRL.exp.argparse import argparse_callbacks, register_argparse_callback

from typing import Optional, Dict, Union

def make_snapshot(args: Optional[Union[Dict, NameSpace, str]]):
    prefix = branch = "/".join(["snapshot", args])
    suffix = 0
    while os.system(f"git --no-pager branch | grep {branch} > /dev/null 2>&1") == 0:
        suffix += 1
        branch = prefix + f"-{suffix}"
    cmd = f"git add -A >/dev/null 2>&1 && \
            git stash >/dev/null 2>&1 && \
            git switch -c {branch} >/dev/null 2>&1 && \
            git stash apply >/dev/null 2>&1 && \
            git add -A >/dev/null 2>&1 && \
            git commit -m \"snapshot: {branch}\" >/dev/null 2>&1 && \
            git switch - >/dev/null 2>&1 && \
            git stash pop >/dev/null 2>&1 "
    ColoredLogger().log_str(f"saving snapshot to {branch}")
    os.system(cmd)
    
    return {
        "UtilsRL.snapshot_branch": branch
    }