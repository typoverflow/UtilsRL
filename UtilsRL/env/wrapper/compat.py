def _parse_reset_result(reset_result):
    contains_info = (
        isinstance(reset_result, tuple) and len(reset_result) == 2
        and isinstance(reset_result[1], dict)
    )
    if contains_info:
        return reset_result[0], reset_result[1], contains_info
    return reset_result, {}, contains_info

def _format_reset_result(reset_result, reset_info, contains_info):
    if contains_info:
        return reset_result, reset_info
    else:
        return reset_result
    
def _parse_step_result(step_result):
    if len(step_result) == 4:
        new_api = False
        obs, rew, done, info = step_result
        trunc = None
    else:
        new_api = True
        obs, rew, done, trunc, info = step_result
    return obs, rew, done, trunc, info, new_api

def _format_step_result(obs, rew, done, trunc, info, new_api):
    if new_api:
        return obs, rew, done, trunc, info
    else:
        return obs, rew, done, info