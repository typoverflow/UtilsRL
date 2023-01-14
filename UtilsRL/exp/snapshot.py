import os

from UtilsRL.misc.namespace import NameSpaceMeta
from UtilsRL.logger import logger

from typing import Optional, Dict, Union

def make_snapshot(args: Optional[Union[Dict, NameSpaceMeta, str]]):
    if args is None:
        return {
            "UtilsRL.snapshot_branch": None
        }
        
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
    logger.info(f"UtilsRL.snapshot: saving code snapshot to {branch}.")
    os.system(cmd)
    
    return {
        "UtilsRL.snapshot_branch": branch
    }