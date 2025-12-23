import re
import os
from openai import OpenAI
from typing import Tuple, Union

from thought_graph.utils import get_logger


logger = get_logger(__name__)

class QueryContext:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE_URL"))
        self.model = os.getenv("OPENAI_API_MODEL", "gpt-4o")
        self.prompt_buffer: list[str] = []
        self._depth: int = 0  # [NEW] 引用计数器

    @property
    def is_active(self) -> bool:
        return self._depth > 0

    def enter(self):
        """进入上下文：增加深度"""
        if self._depth == 0:
            # 只有最外层需要重置 buffer
            self.prompt_buffer.clear()
        self._depth += 1

    def exit(self):
        """退出上下文：减少深度"""
        self._depth -= 1
        if self._depth == 0:
            # 只有最外层退出时才完全清理 (可根据需求决定是否在此处断开连接，通常保留 client)
            self.prompt_buffer.clear()


_query_ctx = QueryContext()

def query(prompt_template: str, *args) -> Union[str, Tuple[str, ...], None]:
    formatted_text = prompt_template.format(*args)
    capture_targets = re.findall(r'\[(.*?)\]', prompt_template)
    
    if not capture_targets:
        _query_ctx.prompt_buffer.append(formatted_text)
        return None
    
    full_context = "\n".join(_query_ctx.prompt_buffer + [formatted_text])
    
    output_sample = prompt_template.replace('[', '<').replace(']', '>').format(*args)
    
    # 按照你的建议：手动插入提示 Tag 强制输出格式
    # 明确告知 LLM 在 <OUTPUT> 之后按顺序填入值
    
    logger.debug(f"Formatted text {formatted_text}")  # 调试信息
    
    instruction = (
        "\n\nPlease complete the thought. "
        "Finally, provide the values for each placeholder in the EXACT order they appear, "
        "Only outputs the values that covered by the brackets [*], carrying brackets <>, and"
        f"under a tag '<OUTPUT>', using the format {output_sample}, e.g.,"
        f"<OUTPUT>\n{output_sample}"
    )

    response = _query_ctx.client.chat.completions.create(
        model=_query_ctx.model,
        messages=[
            {"role": "system", "content": "You are a reasoning assistant. You must use the <OUTPUT> tag to finalize your answer."},
            {"role": "user", "content": full_context + instruction}
        ],
        temperature=0
    )
    
    llm_output = response.choices[0].message.content
    assert llm_output is not None
    
    logger.debug(f"LLM output: {llm_output}")  # 调试信息
    
    # 核心修复：只在 <OUTPUT> 标记之后的部分寻找捕获值
    if "<OUTPUT>" in llm_output:
        result_part = llm_output.split("<OUTPUT>")[-1]
    else:
        result_part = llm_output # 退而求其次
        
    captured_values = re.findall(r'\<(.*?)\>', result_part)
    
    logger.debug(f"captured_values: {captured_values}")  # 调试信息
    
    # 清空缓冲区以保证下一次调用独立 [cite: 231]
    _query_ctx.prompt_buffer.clear()
    
    # 验证捕获数量是否与模板一致，防止解包失败 [cite: 134, 254]
    if len(captured_values) != len(capture_targets):
        # 也可以选择抛出更具体的异常或进行重试逻辑
        pass 

    if len(captured_values) == 0: return None
    if len(captured_values) == 1: return captured_values[0]
    return tuple(captured_values)
