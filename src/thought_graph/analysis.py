from .invoke import _query_ctx
import functools
import sys
from .instrument import tracer_inst, DECORATED_CODE_OBJECTS
from .trace import global_trace

def analysis(func):
    # [关键] 注册原始函数的代码对象，标记为"白盒"
    # Tracer 会根据此集合决定是否 Step-Into
    DECORATED_CODE_OBJECTS.add(func.__code__)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 1. 上下文管理：增加深度
        is_root = (_query_ctx._depth == 0)
        _query_ctx.enter()
        
        # 2. 插桩管理：仅在最外层设置 sys.settrace
        previous_trace = None
        if is_root:
            global_trace.clear()  # 新的执行流，清空旧 Trace
            previous_trace = sys.gettrace()
            sys.settrace(tracer_inst.trace_callback)
        
        try:
            # 执行原函数
            # 注意：如果是嵌套调用，tracer_inst 已经在运行，
            # 它会检测到 func.__code__ 在 DECORATED_CODE_OBJECTS 中，从而自动进入
            result = func(*args, **kwargs)
            
            # 3. 返回值处理
            if is_root:
                # [目标 4] 最外层返回 Trace
                return result, global_trace
            else:
                # [目标 3] 内部调用返回原始值，保持逻辑透明
                return result
                
        finally:
            # 4. 退出逻辑
            if is_root:
                sys.settrace(previous_trace) # 恢复之前的 Tracer
            
            _query_ctx.exit() # 减少深度

    return wrapper