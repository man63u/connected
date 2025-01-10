# Transformer 原理四部曲

## **1. 总体架构概览**

### **简介**
Transformer 是一种基于注意力机制的深度学习模型，广泛应用于自然语言处理任务（如机器翻译、文本生成）。它的核心模块包括：

- **Embedding（嵌入层）**：将输入词转化为向量。
- **Attention（注意力机制）**：捕获序列中单词间的关系。
- **MLP/前馈网络（Feed-Forward Network, FFN）**：提取和整合特征。
- **Unembedding（输出层）**：将特征向量转化为词概率分布。

### **核心模块概览**
1. **Embedding（嵌入层）**
2. **Attention（注意力机制）**
3. **MLP/前馈网络（Feed-Forward Network）**
4. **Unembedding（输出层）**

---

## **2. 各模块详细说明**

### **2.1 Embedding（嵌入层）**

#### **位置**
- Transformer 的开头部分，位于编码器和解码器的最前面。

#### **内容**
- **Tokenization（词元化）**：
  - 将文本分解为单词、子词或字符。
  - 框架支持：
    - TensorFlow 和 PyTorch 的内置 `Embedding` 层。
    - Hugging Face 的 `tokenizer` 和 `embedding` 方法。

- **Tokens 转化为向量**：
  - 嵌入向量维度为 \( d_{model} \)，是固定大小的数值表示。
  - **随机初始化** 或 **预训练嵌入**（如 Word2Vec、GloVe）。

#### **示例**
```python
# PyTorch 的嵌入层示例
embedding = nn.Embedding(vocab_size, d_model)
embedded_tokens = embedding(token_ids)
```

---

### **2.2 Attention（注意力机制）**

#### **位置**
- Transformer 的核心，位于编码器和解码器的每一层。

#### **内容**
- **目的**：
  1. **更新嵌入向量**：通过 `Q`（查询）、`K`（键）、`V`（值）的交互更新单词的表示。
  2. **建立语义相关性**：捕捉单词间的上下文联系。

- **流程**：
  1. **计算 \( Q, K \) 点积**：得出注意力分数，表示某个词对其他词的相关性。
  2. **Softmax 函数归一化**：分数转化为注意力权重。
  3. **加权求和**：用权重加权 \( V \)，得到上下文感知的嵌入表示。

- **多头注意力**：
  - 将 \( Q, K, V \) 划分为 \( h \) 个头，每个头的维度为 \( d_k = d_{model} / h \)。
  - 每个头独立计算注意力，捕获不同子空间中的特征。
  - 最终将所有头的输出拼接起来，得到整体的注意力输出。

#### **公式**
1. 单头注意力：
   \[
   Attention(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
   \]

2. 多头注意力：
   \[
   MultiHead(Q, K, V) = \text{Concat}(head_1, \dots, head_h)W_O
   \]

#### **示例**
```python
# Scaled Dot-Product Attention 的伪代码
scores = Q @ K^T / math.sqrt(d_k)  # 计算点积注意力分数
weights = torch.softmax(scores, dim=-1)  # 对分数进行归一化
output = weights @ V  # 加权求和值
```

---

### **2.3 MLP（前馈网络）**

#### **位置**
- 编码器和解码器中，每层多头注意力后连接一个前馈网络。

#### **内容**
- **结构**：
  - 两层全连接网络：
    \[
    FFN(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2
    \]
  - 第一层扩展维度（如 \( d_{ff} = 2048 \)），第二层将维度缩回 \( d_{model} \)。

- **作用**：
  - **非线性变换**：通过激活函数引入非线性。
  - **特征提取与整合**：通过全连接层提取并组合注意力输出的特征。

#### **示例**
```python
# PyTorch 实现
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
```

---

### **2.4 Unembedding（输出层）**

#### **位置**
- 解码器最后一层，预测词的概率分布。

#### **内容**
- **过程**：
  1. 将解码器的输出通过线性变换映射回词汇表大小。
  2. 使用 Softmax 得到每个词的概率分布。

- **作用**：
  - 将隐藏表示转化为词概率。
  - 实现文本生成任务。

#### **示例**
```python
# PyTorch 的输出层示例
logits = nn.Linear(d_model, vocab_size)(decoder_output)
probs = torch.softmax(logits, dim=-1)
```

---

## **3. 其他模块**

### **残差连接和 LayerNorm**
- **残差连接**：
  - 减轻梯度消失问题，帮助信息直接流动。
  - 每个子层的输出与输入相加。

- **LayerNorm**：
  - 标准化每个位置的向量，提升训练稳定性。

### **掩码机制（Masking）**
- **填充掩码（Padding Mask）**：
  - 避免填充位置对计算的干扰。

- **未来掩码（Look-ahead Mask）**：
  - 防止解码器看到未来信息。

#### **示例**
```python
# 填充掩码
def create_padding_mask(seq, pad_token=0):
    return (seq == pad_token).unsqueeze(1).unsqueeze(2)

# 未来掩码
def create_look_ahead_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask
```

