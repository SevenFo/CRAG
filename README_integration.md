# CRAG 集成指南：将纠正性检索增强生成应用于现有RAG系统

## 项目概述

CRAG (Corrective Retrieval Augmented Generation，纠正性检索增强生成) 是一种提高RAG系统鲁棒性的方法，特别是在处理检索质量不佳的情况时。本文档详细说明如何将CRAG的核心组件整合到现有的RAG系统中，以实现从简单的"retrieve-LLM"架构转变为更强大的"retrieve-CRAG-refined_retrieve_result-LLM"架构。

## CRAG的核心组件

通过分析CRAG项目代码，我们可以确定以下几个关键组件：

### 1. 检索评估器 (Retrieval Evaluator)

检索评估器是CRAG的核心，用于评估检索结果的质量。它基于T5模型实现，在`train_evaluator.py`中定义。

评估器的主要功能：
- 为每对(查询, 文档)分配相关性分数
- 根据分数确定整体检索质量
- 触发适当的知识获取行动

### 2. 行动触发器 (Action Trigger)

根据评估器的结果，CRAG可以触发三种不同的行动：

- **正确 (Correct)**: 当至少有一个文档高度相关时，使用内部知识精炼
- **错误 (Incorrect)**: 当所有文档都不相关时，转向外部知识获取（网页搜索）
- **模糊 (Ambiguous)**: 当相关性不确定时，结合内部和外部知识源

这部分逻辑主要在`CRAG_Inference.py`的`process_flag`函数中实现。

### 3. 知识精炼 (Knowledge Refinement)

当检索到的文档质量较好时，CRAG通过分解和重组文档来提取最相关部分：

- 文档分解：支持多种模式（`fixed_num`, `excerption`, `selection`）
- 相关性评估：评估每个文档片段的相关性
- 重组：选择最相关的片段并组合

这部分功能在`internal_knowledge_preparation.py`中实现。

### 4. 外部知识获取 (External Knowledge)

当内部知识库检索结果不佳时，CRAG会通过网页搜索获取外部知识：

- 关键词提取：使用OpenAI API从查询中提取关键词
- 网页搜索：通过搜索API获取网页信息
- 内容提取和评估：访问网页，提取内容并评估相关性

这部分功能在`external_knowledge_preparation.py`中实现。

### 5. 知识组合 (Knowledge Combination)

在"模糊"情况下，CRAG会结合内部和外部知识：

- 合并两种知识源以提供更全面的信息
- 标记不同的知识来源

这部分逻辑在`combined_knowledge_preparation.py`中实现。

## 将CRAG整合到现有RAG系统

### 步骤 1: 准备检索评估器

首先，我们需要训练或加载一个检索评估器：

```python
from transformers import T5Tokenizer, T5ForSequenceClassification
import torch

# 加载评估器
evaluator_path = "your_evaluator_model_path"
tokenizer = T5Tokenizer.from_pretrained(evaluator_path)
model = T5ForSequenceClassification.from_pretrained(evaluator_path, num_labels=1)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
```

如果您需要训练自己的评估器，可以参考`train_evaluator.py`中的代码。

### 步骤 2: 检索评估函数

实现一个函数来评估检索结果的质量：

```python
def evaluate_retrieval(query, docs, tokenizer, model, device):
    """评估检索到的文档相关性"""
    scores = []
    for doc in docs:
        input_text = query + " [SEP] " + doc
        inputs = tokenizer(input_text, return_tensors="pt", 
                         padding="max_length", max_length=512)
        with torch.no_grad():
            outputs = model(inputs["input_ids"].to(device), 
                          attention_mask=inputs["attention_mask"].to(device))
        score = float(outputs["logits"].cpu())
        scores.append(score)
    return scores
```

### 步骤 3: 行动触发函数

基于评估分数确定行动：

```python
def determine_action(scores, upper_threshold=0.592, lower_threshold=-0.995):
    """根据评估分数确定行动类型"""
    if max(scores) >= upper_threshold:
        return "correct"  # 使用内部知识精炼
    elif max(scores) >= lower_threshold:
        return "ambiguous"  # 使用组合知识
    else:
        return "incorrect"  # 使用外部知识获取
```

### 步骤 4: 知识精炼函数

实现文档分解和重组：

```python
def extract_strips_from_psg(psg, mode="selection"):
    """将文档分解成更小的片段"""
    # 基于scripts/internal_knowledge_preparation.py中的实现
    if mode == 'selection':
        return [psg]  # 最简单的模式，直接返回原始文档
    elif mode == 'fixed_num':
        # 按固定单词数分解
        final_strips = []
        window_length = 50
        words = psg.split(' ')
        buf = []
        for w in words:
            buf.append(w)
            if len(buf) == window_length:
                final_strips.append(' '.join(buf))
                buf = []
        if buf:
            if len(buf) < 10:
                final_strips[-1] += (' ' + ' '.join(buf))
            else:
                final_strips.append(' '.join(buf))
        return final_strips
    elif mode == 'excerption':
        # 按句子分解
        strips = []
        # 实现省略，可参考internal_knowledge_preparation.py
        return strips

def select_relevants(strips, query, tokenizer, model, device, top_n=5):
    """选择最相关的文档片段"""
    strips_data = []
    for i, p in enumerate(strips):
        if len(p.split()) < 4:
            scores = -1.0
        else:
            input_content = query + " [SEP] " + p
            inputs = tokenizer(input_content, return_tensors="pt", 
                            padding="max_length", truncation=True, max_length=512)
            try:
                with torch.no_grad():  
                    outputs = model(inputs["input_ids"].to(device), 
                                attention_mask=inputs["attention_mask"].to(device))
                scores = float(outputs["logits"].cpu())
            except:
                scores = -1.0
        strips_data.append((scores, p, i))
    
    # 按分数排序
    sorted_results = sorted(strips_data, key=lambda x: x[0], reverse=True)
    selected_strips = [s[1] for s in sorted_results[:top_n]]
    return '; '.join(selected_strips)

def refine_internal_knowledge(query, docs, tokenizer, model, device):
    """内部知识精炼"""
    all_strips = []
    for doc in docs:
        all_strips.extend(extract_strips_from_psg(doc, mode="selection"))
    refined_knowledge = select_relevants(
        all_strips, query, tokenizer, model, device, top_n=5
    )
    return refined_knowledge
```

### 步骤 5: 外部知识获取函数

如果需要从网络获取知识：

```python
def get_external_knowledge(query, openai_api_key, search_api_key):
    """通过网页搜索获取外部知识"""
    # 1. 从查询中提取关键词
    import openai
    openai.api_key = openai_api_key
    
    # 此处简化了提取关键词的过程，实际中应使用更健壮的方法
    # 参考external_knowledge_preparation.py中的extract_keywords函数
    keywords = query
    
    # 2. 网页搜索
    # 示例代码，实际中可能需要使用特定的搜索API
    search_results = web_search(keywords, search_api_key)
    
    # 3. 提取网页内容并评估相关性
    web_contents = []
    for result in search_results[:5]:
        content = extract_web_content(result["link"])
        if content:
            web_contents.append(content)
    
    # 4. 使用同样的评估器筛选内容
    if web_contents:
        refined_external = select_relevants(
            web_contents, query, tokenizer, model, device, top_n=5
        )
    else:
        refined_external = ""
        
    return refined_external
```

### 步骤 6: 组合知识函数

实现内部和外部知识的组合：

```python
def combine_knowledge(internal_knowledge, external_knowledge):
    """组合内部和外部知识"""
    # 简单的组合方式，可以根据需要自定义更复杂的组合逻辑
    return f"Knowledge1: {internal_knowledge} [sep] Knowledge2: {external_knowledge}"
```

### 步骤 7: 整合CRAG到RAG流程

将所有组件整合到一个完整的流程中：

```python
def retrieve_crag_llm(query, retriever, llm, tokenizer, model, device, 
                     openai_api_key=None, search_api_key=None):
    """完整的CRAG-RAG流程"""
    # 1. 原始检索
    retrieved_docs = retriever(query)
    
    # 2. 评估检索结果
    scores = evaluate_retrieval(query, retrieved_docs, tokenizer, model, device)
    
    # 3. 确定行动
    action = determine_action(scores)
    
    # 4. 根据行动类型处理知识
    if action == "correct":
        # 内部知识精炼
        refined_knowledge = refine_internal_knowledge(query, retrieved_docs, tokenizer, model, device)
        
    elif action == "incorrect" and openai_api_key and search_api_key:
        # 外部知识获取
        refined_knowledge = get_external_knowledge(query, openai_api_key, search_api_key)
        
    elif action == "ambiguous" and openai_api_key and search_api_key:
        # 组合内部和外部知识
        internal_knowledge = refine_internal_knowledge(query, retrieved_docs, tokenizer, model, device)
        external_knowledge = get_external_knowledge(query, openai_api_key, search_api_key)
        refined_knowledge = combine_knowledge(internal_knowledge, external_knowledge)
        
    else:
        # 降级处理，使用简单过滤的检索结果
        sorted_docs = [doc for _, doc in sorted(zip(scores, retrieved_docs), reverse=True)]
        refined_knowledge = '; '.join(sorted_docs[:3])
    
    # 5. 使用LLM生成回答
    prompt = format_prompt(query, refined_knowledge)
    response = llm(prompt)
    
    return response
```

### 步骤 8: 辅助函数

最后，我们需要一个格式化提示的函数：

```python
def format_prompt(question, paragraph):
    """格式化提示模板"""
    # 可以根据需要自定义提示模板
    prompt = f"Refer to the following documents, follow the instruction and answer the question.\n\nDocuments: {paragraph}\n\nInstruction: Answer the question: {question}"
    return prompt
```

## 简化版集成方案

如果您只想实现CRAG的核心功能，而不需要完整的网页搜索功能，可以使用这个简化版本：

```python
def simple_crag_integration(query, retriever, llm, tokenizer, model, device):
    # 1. 检索文档
    retrieved_docs = retriever(query)
    
    # 2. 评估和筛选
    scores = evaluate_retrieval(query, retrieved_docs, tokenizer, model, device)
    
    # 3. 简化版知识精炼
    if max(scores) >= 0:  # 使用简单阈值
        # 仅保留正分数的文档
        good_docs = [doc for score, doc in zip(scores, retrieved_docs) if score >= 0]
        if good_docs:
            refined_knowledge = '; '.join(good_docs)
        else:
            refined_knowledge = '; '.join(retrieved_docs[:2])  # 降级处理
    else:
        # 所有文档都不太相关，但仍使用前两个
        refined_knowledge = '; '.join(retrieved_docs[:2])
    
    # 4. 生成回答
    prompt = format_prompt(query, refined_knowledge)
    response = llm(prompt)
    
    return response
```

## 调整和优化建议

1. **评估器质量**:
   - 评估器的质量对CRAG效果至关重要
   - 建议在特定领域的数据上微调评估器
   - 参考`train_evaluator.py`中的训练代码

2. **阈值调整**:
   - `upper_threshold`和`lower_threshold`需要根据评估器和数据集特点调整
   - 可以通过验证集调整这些阈值

3. **知识精炼策略**:
   - 根据文档类型选择合适的分解模式（`selection`、`fixed_num`或`excerption`）
   - 对于短文档，`selection`通常效果好；对于长文档，`excerption`或`fixed_num`可能更合适

4. **外部知识集成**:
   - 如果不需要外部知识获取功能，可以简化流程，仅使用内部知识精炼
   - 如果集成外部知识获取，请确保有稳定的API访问

## 结论

将CRAG整合到现有的RAG系统中，可以显著提高系统在处理检索质量不佳情况下的鲁棒性。通过评估检索结果、精炼知识和灵活选择知识源，CRAG能够提供更相关、更准确的信息给大语言模型，从而生成更高质量的回答。

CRAG的核心思想是"纠正"和"优化"检索过程，而不是简单地依赖初始检索结果，这使得整个系统更加智能和可靠。