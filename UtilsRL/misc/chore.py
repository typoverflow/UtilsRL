def safe_eval(expr: str):
    if not isinstance(expr, str):
        raise TypeError("Expr for safe eval must be string.")
    try:
        ret = eval(expr)
    except Exception as e:
        ret = expr
    return ret