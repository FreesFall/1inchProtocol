# GLM5 缠论分析提示词 — 接入指南

## 1. API 调用结构

在 GLM5 API 中，消息按以下顺序组装：

```python
messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT       # system_prompt.md 全文
    },
    {
        "role": "user",
        "content": FEW_SHOT_USER_1     # 示例 1 的用户输入
    },
    {
        "role": "assistant",
        "content": FEW_SHOT_ASSISTANT_1 # 示例 1 的助手回复
    },
    {
        "role": "user",
        "content": FEW_SHOT_USER_2     # 示例 2 的用户输入（可选）
    },
    {
        "role": "assistant",
        "content": FEW_SHOT_ASSISTANT_2 # 示例 2 的助手回复（可选）
    },
    {
        "role": "user",
        "content": REAL_USER_INPUT      # 用户实际的 K 线数据和分析请求
    }
]
```

## 2. Python 示例代码

```python
from zhipuai import ZhipuAI

client = ZhipuAI(api_key="your-api-key")

# 读取提示词文件
with open("system_prompt.md", "r", encoding="utf-8") as f:
    system_prompt = f.read()

with open("few_shot_examples.md", "r", encoding="utf-8") as f:
    few_shot_content = f.read()

# 从 few_shot_examples.md 中提取用户和助手的对话对
# （实际使用时建议拆分为独立的 user/assistant 消息对）
few_shot_user_1 = """请分析以下 K 线数据（30分钟级别）：

| 编号 | 日期时间          | 开盘  | 最高  | 最低  | 收盘  |
|------|-------------------|-------|-------|-------|-------|
| K1   | 2025-03-01 10:00 | 100.0 | 103.0 | 99.0  | 102.0 |
| K2   | 2025-03-01 10:30 | 102.0 | 104.0 | 101.0 | 103.5 |
...（完整数据）
"""

few_shot_assistant_1 = """## 1. K 线合并
...（从 few_shot_examples.md 复制完整回复）
"""

# 用户的真实请求
user_input = """请分析以下日线 K 线数据：

| 编号 | 日期       | 开盘  | 最高  | 最低  | 收盘  | 成交量   |
|------|------------|-------|-------|-------|-------|----------|
| K1   | 2025-04-01 | 15.20 | 15.80 | 15.10 | 15.65 | 1200000 |
...（用户实际数据）
"""

# 发送请求
response = client.chat.completions.create(
    model="glm-5",  # 使用 GLM5 模型
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": few_shot_user_1},
        {"role": "assistant", "content": few_shot_assistant_1},
        {"role": "user", "content": user_input},
    ],
    temperature=0.1,   # 低温度，确保严谨推理
    top_p=0.7,         # 适度采样
    max_tokens=8192,    # 留足空间给完整分析
)

print(response.choices[0].message.content)
```

## 3. 关键参数建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `temperature` | 0.1 ~ 0.2 | 缠论分析需要严格推理，避免高温度带来的随机性 |
| `top_p` | 0.7 | 平衡确定性和表达多样性 |
| `max_tokens` | 8192+ | 完整的 8 步分析输出较长，需要足够的 token 空间 |
| `do_sample` | True | 配合低 temperature 使用 |

## 4. 提升准确率的技巧

### 4.1 分步调用（推荐）

当 K 线数据量大时，将 8 步分析拆分为多次 API 调用：

```python
# 第一次调用：K 线合并 + 分型识别
step1_prompt = f"""
{user_input}

请只执行 步骤 1（K 线合并）和 步骤 2（分型识别），暂不进行后续分析。
"""

step1_response = call_glm5(system_prompt, step1_prompt)

# 第二次调用：基于第一步结果继续
step2_prompt = f"""
以下是上一步的分析结果：

{step1_response}

请基于以上合并 K 线序列和分型结果，继续执行 步骤 3（笔的划分）和 步骤 4（线段的划分）。
"""

step2_response = call_glm5(system_prompt, step2_prompt)

# 以此类推...
```

### 4.2 自检回调

在每步结果返回后，程序化检查自检清单，如果模型标注了 ✗，自动要求修正：

```python
if "✗" in step_response:
    correction_prompt = f"""
    你上一步的分析中存在自检未通过的项目：

    {step_response}

    请重新检查标注为 ✗ 的项目，修正后重新输出该步骤的完整结果。
    """
    corrected = call_glm5(system_prompt, correction_prompt)
```

### 4.3 数据预处理

在调用模型前，先用程序完成 K 线合并（减少模型计算负担）：

```python
def merge_klines(klines):
    """程序化处理 K 线包含关系合并"""
    merged = [klines[0]]
    direction = 0  # 0=未确定, 1=上, -1=下

    for i in range(1, len(klines)):
        curr = klines[i]
        prev = merged[-1]

        # 检查包含关系
        if (prev['high'] >= curr['high'] and prev['low'] <= curr['low']) or \
           (curr['high'] >= prev['high'] and curr['low'] <= prev['low']):
            # 存在包含关系，按方向合并
            if direction >= 0:  # 向上合并
                merged[-1] = {
                    'high': max(prev['high'], curr['high']),
                    'low': max(prev['low'], curr['low']),
                    'merged_from': prev.get('merged_from', [prev]) + [curr]
                }
            else:  # 向下合并
                merged[-1] = {
                    'high': min(prev['high'], curr['high']),
                    'low': min(prev['low'], curr['low']),
                    'merged_from': prev.get('merged_from', [prev]) + [curr]
                }
        else:
            # 无包含关系，更新方向
            if curr['high'] > prev['high']:
                direction = 1
            elif curr['high'] < prev['high']:
                direction = -1
            merged.append(curr)

    return merged
```

然后将合并后的数据和原始数据一起传给模型：

```python
user_input = f"""
原始 K 线数据：
{format_klines(raw_klines)}

程序预处理的合并 K 线序列（请验证后使用）：
{format_klines(merged_klines)}

请从 步骤 1 开始验证合并结果，然后继续完整分析。
"""
```

### 4.4 多级别分析

对于多级别联动分析，建议为每个级别单独调用：

```python
# 先分析 5 分钟级别
min5_result = analyze(system_prompt, klines_5min)

# 再分析 30 分钟级别，并参考 5 分钟结果
min30_prompt = f"""
以下是 30 分钟级别 K 线数据：
{klines_30min}

参考信息 — 5 分钟级别分析结果：
{min5_result}

请分析 30 分钟级别，并确保与 5 分钟级别的结构一致。
"""
min30_result = call_glm5(system_prompt, min30_prompt)
```

## 5. 开场澄清协议的使用

系统提示词第六部分启用了"开场澄清协议"：当用户的请求缺少级别、起点、数据源或结构假设时，模型会先反问 3–4 个关键问题，再进入 8 步分析。

### 5.1 期望行为

- 用户发送笼统请求 → 模型输出编号化的澄清问题（不立即分析）
- 用户补全前提后 → 模型先输出 `## 0. 前提确认`，然后依次进入步骤 1–8

### 5.2 绕过澄清（直接分析）

如果你的业务场景希望跳过澄清，直接把完整前提塞进用户消息即可，模型会检测到"所有必要前提已给出"从而略过提问：

```text
【分析前提 — 已给齐，请直接进入步骤 0 前提确认并开始 8 步分析】
- 级别：日线
- 起点：K1 起全部
- 数据：下方表格
- 结构假设：无预设
- 分析目标：完整 8 步
- 多级别联动：否
- 规则：缺口成笔=关，MACD 辅助=开

<K 线表格>
```

### 5.3 自动填充默认值

如果不希望模型反问，可以在 user 消息末尾附加：

```text
请不要反问。未指定的前提请使用第六部分 §6.4 的默认假设，并在 步骤 0 显式声明。
```

## 6. 常见问题

### Q：模型经常在 K 线合并步骤出错怎么办？

A：这是最容易出错的步骤。建议：
1. 用程序预处理合并，只让模型验证
2. K 线数量控制在 30-50 根以内
3. 提供明确的合并方向初始值

### Q：模型跳步直接给结论怎么办？

A：在用户消息末尾加入硬性约束：

```
【重要】请严格按照 8 个步骤逐步输出，每步包含自检清单。
不得跳过任何步骤。如果某步因数据不足无法完成，请明确标注原因。
```

### Q：分析结果与实际走势不一致怎么办？

A：
1. 检查输入数据是否准确（OHLC 是否正确）
2. 确认分析级别与 K 线级别匹配
3. 缠论分析的是已完成走势的结构，不是预测未来
4. 最后一根未收盘 K 线的分析可能随收盘价变化

### Q：Token 消耗太大怎么优化？

A：
1. 用分步调用代替一次性输出
2. 程序预处理合并步骤
3. 减少 few-shot 示例数量（保留 1 个最相关的即可）
4. 对简单走势可以省略详细的合并过程，只要求输出合并结果

### Q：如何处理实时行情的增量分析？

A：维护一个对话上下文，将之前的分析结果作为历史传入：

```python
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "之前的分析（截至K50）"},
    {"role": "assistant", "content": previous_analysis},
    {"role": "user", "content": "新增K线 K51-K55，请更新分析"},
]
```
