from typing import Any, List, Dict, Optional, Tuple
from collections import deque
from .trace import global_trace, KeyType

def _find_source_index(target_val: Any, source_pool: List[Tuple[int, Any]]) -> Optional[int]:
    """
    在源池中寻找与 target_val 匹配的元素索引。
    优先匹配 identity (is)，其次匹配 equality (==)。
    匹配成功后从池中移除，以支持重复元素。
    """
    # 1. 尝试 Identity 匹配 (最强一致性)
    for i, (idx, src_val) in enumerate(source_pool):
        if src_val is target_val:
            source_pool.pop(i)
            return idx
            
    # 2. 尝试 Equality 匹配 (值相同)
    for i, (idx, src_val) in enumerate(source_pool):
        if src_val == target_val:
            source_pool.pop(i)
            return idx
            
    return None

def handle_sorted(scope_id: str, def_name: str, def_obj: Any, use_name: str, use_obj: Any) -> bool:
    """
    处理 sorted 函数的依赖追踪。
    逻辑：candidates = sorted(next_states)
    建立 candidates[i] <- next_states[j] 的点对点依赖。
    """
    if not isinstance(def_obj, (list, tuple)) or not isinstance(use_obj, (list, tuple)):
        return False

    # 构建源池：[(index, value), ...]
    # 使用 list 而不是 dict，以处理非 hashable 元素和重复元素
    source_pool = list(enumerate(use_obj))
    
    # 遍历输出列表
    for out_idx, out_val in enumerate(def_obj):
        # 在输入中寻找对应项
        in_idx = _find_source_index(out_val, source_pool)
        
        if in_idx is not None:
            # 找到匹配！建立精确的元素级依赖
            # DEF: candidates[out_idx]
            # USE: next_states[in_idx]
            
            # 这里我们传入 val_obj，保持 snapshot 能力
            def_node = global_trace.new_def_node(
                scope_id, def_name, out_idx, val_obj=out_val
            )
            
            use_node = global_trace.get_current_node(scope_id, use_name, in_idx)
            
            # 使用特殊的事件类型，表明这是重排序
            global_trace.add_event(def_node, [use_node], 'reorder')
        else:
            # 没找到对应项（理论上 sorted 不应该发生，除非 key 改变了值本身且无法还原）
            # Fallback 到普通处理：依赖整体
            pass 

    return True

# 注册表
SPECIAL_FUNCTION_HANDLERS = {
    'sorted': handle_sorted,
    # 未来可添加 'reversed', 'filter' 等
}