#  [Large Language Models：A Survey](https://arxiv.org/pdf/2402.06196)

论文整体架构为以下五方面：

1. 讨论突出的LLMs（eg.GPT，LLamA、PaLM）的特点、贡献、局限；
2. 概述构建和增强LLMs的相关技术；
3. 调查在训练LLMs和微调（fine-tuning）时受欢迎的数据集；
4. 介绍LLMs的评估指标；
5. 最后讨论开放的挑战和未来研究方向；

> Key Word：

##  1.主流LLMs

###  1.1Bert

![image-20250121114854115](./Large Language Models：A Survey 2402.06196v2.assets/image-20250121114854115.png)

**Bert**属于 Coder-Only 模型，有三个模块构成：（1）嵌入模块`an
embedding module that converts input text into a sequence
of embedding vectors`；（2）Transformer 编码块`a stack of Transformer encoders that converts embedding vectors into contextual representation vectors`；（3）分类器模块，将输出层的表示向量转换为 One-hot编码`a fully connected layer that converts the representation vectors (at the final layer) to one-hot vectors`.

预训练的 Bert 模型可以添加对应的分类器层进行微调来适应不同的语言理解任务。

> Bert 变体：
>
> * **RoBERTa**：使用一组模型设计选择和训练策略（model design choices and training strategies），例如：修改一些关键的超参数，删除下一个句子的预训练目标以及使用更大的 mini-batches 和 learning rates
> * **ALBERT**：使用两种参数约简技术来降低内存消耗并提高 BERT的训练速度-（1）将嵌入矩阵拆分为两个较小的矩阵；（2）在 groups 中使用 repeating layers 来拆分
> * **DeBERTa**：使用两种新技术改进了 BERT 和 RoBERTa 模型-（1）disentangled attention mechanism：每个单词分别使用两个向量来表示，分别编码其内容和位置，以及单词之间的注意权重分别使用解纠缠矩阵 disentangled matrices 对其内容和相对位置进行计算（2）使用增强的掩码解码器在解码层中合并绝对位置，以预测模型预训练中的 masked tokens。

###  1.2GPT1&GPT2

**GPT1&GPT2**是最广泛使用的两个纯解码器预训练语言模型（decoder-only PLMs）。

GPT-1 首次证明，在各种未标记文本语料库上，以**自监督**学习方式（即 next world/token prediction），仅使用解码器转换器模型的生成式预训练(GPT)可以在广泛的自然语言任务上获得良好的性能。

GPT-2 表明，在由数百万个网页组成的大型 WebText 数据集上训练时，语言模型能够在没有任何明确监督的情况下学习执行特定的自然语言任务。GPT-2 模型遵循 GPT-1 的模型设计，并进行了一些修改：将层归一化移动到
每个子块的输入，在最终的 self-attention 块之后添加额外的层归一化，修改初始化以考虑残差路径上的积累和残差层的权重缩放，词汇量扩展到50、25，并且上下文大小从512个 token 增加到1024个 token。

##  2. Large Language Model Families

回顾三个LLM家族: GPT、LLaMA 和 PaLM。

![image-20250125103927377](./Large Language Models：A Survey 2402.06196v2.assets/image-20250125103927377.png)

### 2.1 The GPT Family

Generative Pre-trained Transformers (GPT)是由OpenAI开发的**仅基于解码器**的转换器语言模型家族。

#### 2.1.1 GPT-3

GPT-3 是一个预训练的自回归语言模型，具有1750亿个参数。GPT-3被广泛认为是第一个 LLM，因为它不仅比以前的 plm 大得多，而且第一次展示了在以前较小的 plm 中没有观察到的应急能力。GPT-3 显示了上下文学习的紧急能力，这意味着GPT-3 可以应用于任何下游任务，而无需任何梯度更新或微调，任务和少量演示完全通过与模型的文本交互指定。

####  2.1.2 CODEX

CODEX 是一种通用编程模型，可以解析自然语言并生成响应代码。CODEX 是GPT-3的后代，对从 GitHub 收集的代码语料库进行了微调，支持微软的GitHub Copilot。

####  2.1.3 WebGPT

WebGPT 是 GPT-3 的另一个后代，经过微调，可以使用基于文本的网络浏览器回答开放式问题，方便用户搜索和浏览网络。具体来说，WebGPT的训练分为三个步骤。

1. WebGPT学习使用人类演示数据模拟人类浏览行
   为。
2. 学习奖励函数来预测人类的偏好。
3. 对WebGPT进行改进，通过 **reinforcement learning** 和 **rejection sampling** 来优化奖励函数。

####  2.1.4 GPT-4

GPT-4 是一个多模态LLM，可以将图像和文本作为输入，并产生文本输出。GPT-4先在大型文本语料库预训练来预测  next tokens，然后使用 **RLHF**（reinforcement learning human feedback） 进行微调，使模型行为与人类期望的行为保持一致。

###  2.2 The LLaMA Family

LLaMA家族：LLaMA 是一个基础语言模型的集合，由 Meta 发布。与GPT 模型不同，LLaMA 模型是开源的。

第一套 LLaMA 模型于2023年2月发布，参数范围从 7B 到 65B。LLaMA使用GPT-3的 transformer 架构，并对架构进行了一些小的修改，包括:

(1) 使用SwiGLU激活函数代替ReLU;

(2) 使用 rotary positional embeddings 代替 absolute positional embedding;

(3) 使用 root-mean-squared layer-normalization 代替 standard layer-normalization

开源的 LLaMA-13B 模型在大多数基准测试中优于专有的 GPT-3(175B) 模型，使其成为LLM研究的良好基准。