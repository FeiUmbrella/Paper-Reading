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

![image-20250121114854115](Large Language Models：A Survey 2402.06196v2.assets/image-20250121114854115.png)

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