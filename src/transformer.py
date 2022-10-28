import torch
import torch.nn as nn
# nn.functional工具包装载了网络层中那些只进行计算, 而没有参数的层
import torch.nn.functional as F
# torch中变量封装函数Variable.
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
import numpy as np
import copy


# 定义Embeddings类来实现文本嵌入层, 这里s说明代表两个一模一样的嵌入层, 他们共享参数.
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        """
        d_model: 词嵌入的维度
        vocab:   指词表的大小
        """
        # 接着就是使用super的方式指明继承nn.Module的初始化函数
        super(Embeddings, self).__init__()
        # 定义Embedding层
        self.lut = nn.Embedding(vocab, d_model)
        # 将参数传入类中
        self.d_model = d_model

    def forward(self, x):
        """
        当传给该类的实例化对象参数时, 自动调用该类函数
        x: 代表输入给模型的文本通过词汇映射后的张量
        """
        return self.lut(x) * math.sqrt(self.d_model)

# 词嵌入维度是512维
d_model = 512
# 词表大小是1000
vocab = 1000

# 输入x是一个使用Variable封装的长整型张量, 形状是2 x 4
x = Variable(torch.LongTensor([[100, 2, 421, 508],[491, 998, 1, 221]]))

emb = Embeddings(d_model, vocab)
embr = emb(x)
# print("embr:", embr)
# # 2*4*512
# print(embr.shape)

# 构建位置编码器类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        """
        d_model: 词嵌入维度,
        dropout: 置0比率
        max_len: 每个句子的最大长度
        """
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        # 初始化位置编码矩阵
        pe = torch.zeros(max_len, d_model)

        # 初始化绝对位置矩阵, 用它的索引去表示.
        # 首先使用arange方法获得一个连续自然数向量
        # 然后再使用unsqueeze方法拓展向量维度使其成为 max_len*1 的矩阵
        position = torch.arange(0, max_len).unsqueeze(1)

        # 绝对位置矩阵初始化之后, 接下来就是考虑如何将这些位置信息加入到位置编码矩阵中
        # 最简单思路就是先将 max_len*1 的绝对位置矩阵, 变换成 max_len*d_model 形状, 然后覆盖原来的初始位置编码矩阵即可
        # 要做这种矩阵变换, 就需要一个 1*d_model 形状的变换矩阵div_term, 我们对这个变换矩阵的要求除了形状外
        # 还希望它能够将自然数的绝对位置编码缩放成足够小的数字, 有助于在之后的梯度下降过程中更快的收敛
        # 首先使用arange获得一个自然数矩阵,    但是细心的同学们会发现,    我们这里并没有按照预计的一样初始化一个 1*d_model 的矩阵
        # 而是有了一个跳跃, 只初始化了一半即 1*d_model/2 的矩阵. 其实这里并不是真正意义上的初始化了一半的矩阵
        # 我们可以把它看作是初始化了两次, 而每次初始化的变换矩阵会做不同的处理, 第一次初始化的变换矩阵分布在正弦波上, 第二次初始化的变换矩阵分布在余弦波上
        # 并把这两个矩阵分别填充在位置编码矩阵的偶数和奇数位置上, 组成最终的位置编码矩阵
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 这样我们就得到了位置编码矩阵pe, pe现在还只是一个二维矩阵, 要想和embedding的输出 (一个三维张量) 相加, 就必须拓展一个维度
        pe = pe.unsqueeze(0)

        # 最后把pe位置编码矩阵注册成模型的buffer, 什么是buffer呢
        # 我们把它认为是对模型效果有帮助的, 但是却不是模型结构中超参数或者参数, 不需要随着优化步骤进行更新的增益对象.
        # 注册之后我们就可以在模型保存后重加载时和模型结构与参数一同被加载.
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x: 表示文本序列的词嵌入表示"""
        # 在相加之前我们对pe做一些适配工作, 将这个三维张量的第二维也就是句子最大长度的那一维将切片到与输入的x的第二维相同即x.size(1)
        # 因为我们默认max_len为5000一般来讲实在太大了, 很难有一条句子包含5000个词汇, 所以要进行与输入张量的适配
        # 最后使用Variable进行封装, 使其与x的样式相同, 但是它是不需要进行梯度求解的, 因此把requires_grad设置成false
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        # 最后使用self.dropout对象进行'丢弃'操作, 并返回结果.
        return self.dropout(x)

d_model = 512
dropout = 0.1
max_len=60

# 输入x是Embedding层的输出的张量, 形状是 2*4*512
x = embr

pe = PositionalEncoding(d_model, dropout, max_len)
pe_result = pe(x)
# print("pe_result:", pe_result)
# print(pe_result.shape)


# # 创建一张15 x 5大小的画布
# plt.figure(figsize=(15, 5))

# # 实例化PositionalEncoding类得到pe对象
# pe = PositionalEncoding(20, 0)

# # 然后向pe传入被Variable封装的tensor, 这样pe会直接执行forward函数,
# # 且这个tensor里的数值都是0, 被处理后相当于位置编码张量
# y = pe(Variable(torch.zeros(1, 100, 20)))

# # 然后定义画布的横纵坐标, 横坐标到100的长度, 纵坐标是某一个词汇中的某维特征在不同长度下对应的值
# # 因为总共有20维之多, 我们这里只查看4, 5, 6, 7维的值.
# plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())

# # 在画布上填写维度提示信息
# plt.legend(["dim %d"%p for p in [4,5,6,7]])
# plt.show()

def subsequent_mask(size):
    """
    生成向后遮掩的掩码张量
    size: 掩码张量最后两个维度, 形成一个方阵
    """
    attn_shape = (1, size, size)

    # 使用np.ones()先构建全1张量, 然后利用np.triu()形成上三角矩阵, 最后为了节约空间
    # 使其中的数据类型变为无符号8位整形unit8
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    # 反转三角矩阵
    return torch.from_numpy(1 - subsequent_mask)

# size = 5
# sm = subsequent_mask(size)
# print("sm:", sm)

# plt.figure(figsize=(5,5))
# plt.imshow(subsequent_mask(20)[0])
# plt.show()

def attention(query, key, value, mask=None, dropout=None):
    """注意力机制的实现"""
    # 首先取query的最后一维大小, 代表词嵌入的维度
    d_k = query.size(-1)
    # 按照注意力公式, 将query与key的转置进行矩阵乘法, 然后除以缩放系数
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 接着判断是否使用掩码张量
    if mask is not None:
        # 利用tensor.masked_fill()方法, 将掩码张量和scores张量每个位置一一比较
        # 如果掩码张量处为0, 则对应的scores张量用-1e9这个值来替换
        scores = scores.masked_fill(mask == 0, -1e9)

    # 对scores的最后一维进行softmax操作
    p_attn = F.softmax(scores, dim=-1)

    # 判断是否使用dropout
    if dropout is not None:
        p_attn = dropout(p_attn)

    # 返回query的注意力表示和注意力张量
    return torch.matmul(p_attn, value), p_attn

query = key = value = pe_result
# 令mask为一个 2*4*4 的零张量
mask = Variable(torch.zeros(2, 4, 4))
attn, p_attn = attention(query, key, value, mask=mask)
# print("attn:", attn)
# print(attn.shape)
# print("p_attn:", p_attn)
# print(p_attn.shape)

# 用于深度拷贝的copy工具包
import copy

# 首先需要定义克隆函数, 因为在多头注意力机制的实现中, 用到多个结构相同的线性层.
# 我们将使用clone函数将他们一同初始化在一个网络层列表对象中. 之后的结构中也会用到该函数.
def clones(module, N):
        """
        用于生成相同网络层的克隆函数
        module: 要克隆的目标网络层
        N:      代表需要克隆的数量
        """
        # 通过for循环对module进行N次深度拷贝, 使其每个module成为独立的层
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        """
        head:          头数
        embedding_dim: 词嵌入的维度
        dropout:       进行dropout操作时置0比率, 默认是0.1
        """
        super(MultiHeadedAttention, self).__init__()
        # 需要确认一个事实, 多头的数量head能被词嵌入的维度embedding_dim整除
        assert embedding_dim % head == 0

        # 得到每个头获得的分割词向量维度d_k
        self.d_k = embedding_dim // head

        self.head = head
        self.embedding_dim = embedding_dim

        # 实例化4个线性层, 分别是Q, K, V以及最终的输出线性层
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)

        # 初始化注意力张量, 现在还没有结果所以为None.
        self.attn = None

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # 判断是否使用掩码张量
        if mask is not None:
            # 使用unsqueeze拓展维度, 代表多头中的第n头
            mask = mask.unsqueeze(0)

        # 得到batch_size, 他是query尺寸的第1维大小, 代表有多少条样本
        batch_size = query.size(0)

        # 多头处理环节
        # 首先利用zip和for循环将线性层和三个输入QKV组到一起
        # 做完线性变换后, 开始为每个头分割输入. 使用view()方法对线性变换的结果进行维度重塑, 多加了一个维度h, 代表头数. 因此每个头可以获得一部分词特征组成的句子
        # 为了让代表句子长度维度和词向量维度能够相邻, 对第二维和第三维进行转置操作, 这样注意力机制才能找到词义与句子位置的关系
        # 从attention函数中可以看到, 利用的是原始输入的倒数第一和第二维
        query, key, value = [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
                for model, x in zip(self.linears, (query, key, value))]

        # 将每个头的输出传入attention层
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 得到每个头计算结果组成的4维张量, 需要将其转换为输入的形状以方便后续的计算
        # 进行第一步处理环节的逆操作, 先对第2和第3维进行转置, 然后使用contiguous()方法
        # 注意: 经历了transpose()方法后, 必须使用contiguous()方法,不然无法使用view()方法
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)

        # 最后将x输入线性层列表中的最后一个线性层中进行处理, 得到最终的多头注意力结构输出
        return self.linears[-1](x)

head = 8
embedding_dim = 512
dropout = 0.2

# 假设输入的Q, K, V仍然相等
query = value = key = pe_result
mask = Variable(torch.zeros(8, 4, 4))

mha = MultiHeadedAttention(head, embedding_dim, dropout)
mha_result = mha(query, key, value, mask)
# print(mha_result)
# print(mha_result.shape)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        d_model: 线性层的输入维度也是第二个线性层的输出维度 (希望维度不变)
        d_ff:    第二个线性层的输入维度和第一个线性层的输出维度
        dropout: 置0比率
        """
        super(PositionwiseFeedForward, self).__init__()

        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))

d_model = 512
d_ff = 64
dropout = 0.2

# 输入参数x可以是多头注意力机制的输出, 2*4*512
x = mha_result

ff = PositionwiseFeedForward(d_model, d_ff, dropout)
ff_result = ff(x)
# 2*4*512
# print(ff_result)
# print(ff_result.shape)

# 通过LayerNorm实现规范化层的类
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        """
        features: 词嵌入的维度,
        eps:      一个足够小的数, 在规范化公式的分母中出现, 防止分母为0. 默认是1e-6
        """
        super(LayerNorm, self).__init__()

        # 根据features的形状初始化两个参数张量a2, 和b2, 第一个初始化为1张量
        # 第二个初始化为0张量, 这两个张量就是规范化层的参数
        # 因为直接对上一层得到的结果做规范化公式计算, 将改变结果的正常表征, 因此就需要有参数作为调节因子
        # 使其即能满足规范化要求, 又能不改变针对目标的表征.最后使用nn.parameter封装, 代表他们是模型的参数
        self.a2 = nn.parameter.Parameter(torch.ones(features))
        self.b2 = nn.parameter.Parameter(torch.zeros(features))

        self.eps = eps

    def forward(self, x):
        # 对输入变量x求其最后一个维度的均值, 并保持输出维度与输入维度一致
        # 求最后一个维度的标准差, 用x减去均值除以标准差获得规范化的结果
        # 对结果乘以缩放参数a2, 加上位移参数b2
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a2 * (x - mean) / (std + self.eps) + self.b2

features = d_model = 512
eps = 1e-6

# 输入x来自前馈全连接层的输出
x = ff_result

ln = LayerNorm(features, eps)
ln_result = ln(x)
# print(ln_result)
# print(ln_result.shape)

# 使用SublayerConnection来实现子层连接结构的类
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        """
        size:    词嵌入维度的大小
        dropout: 随机置0比率
        """
        super(SublayerConnection, self).__init__()

        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        """
        sublayer: 该子层连接中的子层函数
        """
        return x + self.dropout(sublayer(self.norm(x)))

size = 512
dropout = 0.2
head = 8
d_model = 512

# 令x为位置编码器的输出
x = pe_result
mask = Variable(torch.zeros(8, 4, 4))

# 假设子层中装的是多头注意力层, 实例化这个类
self_attn =    MultiHeadedAttention(head, d_model)

# 使用lambda获得一个函数类型的子层
sublayer = lambda x: self_attn(x, x, x, mask)

sc = SublayerConnection(size, dropout)
sc_result = sc(x, sublayer)
# print(sc_result)
# print(sc_result.shape)

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        """
        size:         词嵌入维度的大小, 也即编码器层的大小
        self_attn:    多头自注意力子层实例化对象
        feed_froward: 前馈全连接层实例化对象
        """
        super(EncoderLayer, self).__init__()

        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # 编码器层中有两个子层连接结构, 使用clones函数克隆
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

size = 512
head = 8
d_model = 512
d_ff = 64
dropout = 0.2
x = pe_result

self_attn = MultiHeadedAttention(head, d_model)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
mask = Variable(torch.zeros(8, 4, 4))

el = EncoderLayer(size, self_attn, ff, dropout)
el_result = el(x, mask)
# print(el_result)
# print(el_result.shape)

class Encoder(nn.Module):
    def __init__(self, layer, N):
        """
        layer: 编码器层实例对象
        N:     编码器层个数
        """
        super(Encoder, self).__init__()
        # 首先使用clones()函数克隆N个编码器层放在self.layers中
        self.layers = clones(layer, N)
        # 再初始化一个规范化层, 它将用在编码器的最后面.
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

size = 512
head = 8
d_model = 512
d_ff = 64
dropout = 0.2

attn = MultiHeadedAttention(head, d_model)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
# 因为编码器层中的子层是不共享的, 因此需要使用深度拷贝各个对象
c = copy.deepcopy
layer = EncoderLayer(size, c(attn), c(ff), dropout)

# 编码器中编码器层的个数N
N = 8
x = pe_result
mask = Variable(torch.zeros(8, 4, 4))

en = Encoder(layer, N)
en_result = en(x, mask)
# print(en_result)
# print(en_result.shape)

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward,
            dropout):
        """
        size:         词嵌入的维度大小, 也即码器层的尺寸
        self_attn:    多头自注意力对象 (Q=K=V)
        src_attn:     多头注意力对象 (Q!=K=V)
        feed_forward: 前馈全连接层对象
        droupout:     置0比率
        """
        super(DecoderLayer, self).__init__()

        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward

        # 克隆三个子层连接对象
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, source_mask, target_mask):
        """
        x:           上一层的输入张量
        mermory:     编码器层的语义存储变量
        source_mask: 源数据的掩码张量
        target_mask: 目标数据的掩码张量
        """
        # 将memory表示成m方便之后使用
        m = memory

        # 第一个子层使用自注意力机制, Q=K=V=x
        # target_mask对目标数据进行遮掩, 因为此时模型可能还没有生成任何目标数据
        # 比如在解码器准备生成第一个字符或词汇时, 我们其实已经传入了第一个字符以便计算损失
        # 但是我们不希望在生成第一个字符时模型能利用这个信息, 因此我们会将其遮掩, 同样生成第二个字符或词汇时
        # 模型只能使用第一个字符或词汇信息, 第二个字符以及之后的信息都不允许被模型使用
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))

        # 第二个子层使用常规的注意力机制, Q=x, k=v=memory
        # source_mask对源数据进行遮掩, 原因并非是抑制信息泄漏
        # 而是遮蔽掉对结果没有意义的字符而产生的注意力值, 以此提升模型效果和训练速度.
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, source_mask))

        # 最后为前馈全连接子层
        return self.sublayer[2](x, self.feed_forward)

# 类的实例化参数与解码器层类似, 相比多出了src_attn, 但是和self_attn是同一个类.
head = 8
size = 512
d_model = 512
d_ff = 64
dropout = 0.2
self_attn = src_attn = MultiHeadedAttention(head, d_model, dropout)

# 前馈全连接层也和之前相同
ff = PositionwiseFeedForward(d_model, d_ff, dropout)

# x是来自目标数据的词嵌入表示, 但形式和源数据的词嵌入表示相同
x = pe_result

# memory是来自编码器的输出
memory = en_result

# 实际中source_mask和target_mask并不相同, 为了方便计算令他们都为mask
mask = Variable(torch.zeros(8, 4, 4))
source_mask = target_mask = mask

dl = DecoderLayer(size, self_attn, src_attn, ff, dropout)
dl_result = dl(x, memory, source_mask, target_mask)
# print(dl_result)
# print(dl_result.shape)

# 使用类Decoder来实现解码器
class Decoder(nn.Module):
    def __init__(self, layer, N):
        """
        layer: 解码器层实例化对象
        N:     解码器层的个数
        """
        super(Decoder, self).__init__()

        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, source_mask, target_mask):
        """
        x:           目标数据的嵌入表示
        memory:      编码器层的输出,
        source_mask: 代表源数据的掩码张量
        target_mask: 目标数据的掩码张量
        """
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)

size = 512
d_model = 512
head = 8
d_ff = 64
dropout = 0.2
c = copy.deepcopy
attn = MultiHeadedAttention(head, d_model)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
layer = DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout)
N = 8

x = pe_result
memory = en_result
mask = Variable(torch.zeros(8, 4, 4))
source_mask = target_mask = mask

de = Decoder(layer, N)
de_result = de(x, memory, source_mask, target_mask)
# print(de_result)
# print(de_result.shape)

class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        """
        d_model:    词嵌入维度
        vocab_size: 词表大小
        """
        super(Generator, self).__init__()
        self.project = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # 在这里之所以使用log_softmax是因为和我们这个pytorch版本的损失函数实现有关, 在其他版本中将修复
        # log_softmax就是对softmax的结果又取了对数, 因为对数函数是单调递增函数
        # 因此对最终我们取最大的概率值没有影响
        return F.log_softmax(self.project(x), dim=-1)

d_model = 512
vocab_size = 1000
x = de_result

gen = Generator(d_model, vocab_size)
gen_result = gen(x)
# print(gen_result)
# print(gen_result.shape)

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, source_embed, target_embed, generator):
            """
            encoder:      编码器对象
            decoder:      解码器对象
            source_embed: 源数据嵌入函数
            target_embed: 目标数据嵌入函数
            generator:    输出部分的类别生成器对象
            """
            super(EncoderDecoder, self).__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.src_embed = source_embed
            self.tgt_embed = target_embed
            self.generator = generator

    def encode(self, source, source_mask):
            return self.encoder(self.src_embed(source), source_mask)

    def decode(self, memory, source_mask, target, target_mask):
            return self.decoder(self.tgt_embed(target), memory, source_mask, target_mask)

    def forward(self, source, target, source_mask, target_mask):
            """
            source: 		源数据
            target: 		目标数据
            source_mask:	源数据的掩码张量
            target_mask:	目标数据的掩码张量
            """
            return self.decode(self.encode(source, source_mask), source_mask, target, target_mask)

vocab_size = 1000
d_model = 512
encoder = en
decoder = de
source_embed = nn.Embedding(vocab_size, d_model)
target_embed = nn.Embedding(vocab_size, d_model)
generator = gen

# 假设源数据与目标数据相同, 实际中并不相同
source = target = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))

# 假设src_mask与tgt_mask相同, 实际中并不相同
source_mask = target_mask = Variable(torch.zeros(8, 4, 4))

ed = EncoderDecoder(encoder, decoder, source_embed, target_embed, generator)
ed_result = ed(source, target, source_mask, target_mask)
# print(ed_result)
# print(ed_result.shape)

def make_model(source_vocab, target_vocab, N=6,
               d_model=512, d_ff=2048, head=8, dropout=0.1):
    """
    source_vocab: 源数据特征(词汇)总数
    target_vocab: 目标数据特征(词汇)总数
    N:            编码器和解码器堆叠数
    d_model:      词向量映射维度
    d_ff:         前馈全连接网络中变换矩阵的维度
    head:         多头注意力结构中的多头数
    dropout:      置零比率
    """
    c = copy.deepcopy

    # 实例化多头注意力类
    attn = MultiHeadedAttention(head, d_model)

    # 实例化前馈全连接类
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    # 实例化位置编码类
    position = PositionalEncoding(d_model, dropout)

    # 编码器层, 解码器层, 源数据Embedding层和位置编码组成的有序结构
    # 目标数据Embedding层和位置编码组成的有序结构, 以及类别生成器层
    # 在编码器层中有attention子层以及前馈全连接子层
    # 在解码器层中有两个attention子层以及前馈全连接层
    model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            nn.Sequential(Embeddings(d_model, source_vocab), c(position)),
            nn.Sequential(Embeddings(d_model, target_vocab), c(position)),
            Generator(d_model, target_vocab))

    # 初始化模型中的参数, 比如线性层中的变换矩阵
    # 如果参数的维度大于1, 则将其初始化成一个服从均匀分布的矩阵
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

source_vocab = 11
target_vocab = 11
N = 6
# 其他参数都使用默认值

# if __name__ == '__main__':
#     res = make_model(source_vocab, target_vocab, N)
#     print(res)

# --------------------------------------------------------

# 导入工具包Batch, 它能够对原始样本数据生成对应批次的掩码张量
from pyitcast.transformer_utils import Batch
# get_std_opt用于获得标准的针对Transformer模型的优化器
# 该标准优化器基于Adam优化器, 使其对序列到序列的任务更有效
from pyitcast.transformer_utils import get_std_opt
# 标签平滑工具包LabelSmoothing用于标签平滑, 作用是小幅度的改变原有标签值的值域
# 因为在理论上即使是人工的标注数据也并非完全正确, 会受到一些外界因素的影响而产生微小偏差
# 使用标签平滑来弥补这种偏差, 减少模型对某一条规律的绝对认知, 以防止过拟合
from pyitcast.transformer_utils import LabelSmoothing
# 损失计算工具包SimpleLossCompute能够使用标签平滑后的结果进行损失计算 (交叉熵损失函数)
from pyitcast.transformer_utils import SimpleLossCompute
# 单轮训练工具包run_epoch对模型使用给定的损失函数计算方法进行单轮参数更新， 并打印每轮参数更新的损失结果
from pyitcast.transformer_utils import run_epoch
# 贪婪解码工具包greedy_decode对最终结进行贪婪解码, 即每次预测都选择概率最大的结果作为输出
# 这不一定能获得全局最优, 但却拥有最高的执行效率
from pyitcast.transformer_utils import greedy_decode


def data_generator(V, batch_size, num_batch):
    """
    用于随机生成copy任务的数据,
    V:          随机生成数字的最大值+1
    batch_size: 每次输送给模型更新一次参数的数据量, 经历这些样本训练后进行一次参数的更新
    num_batch:  一共输送num_batch次完成一轮
    """
    for i in range(num_batch):
        # 使用np的random.randint()方法随机生成[1, V)的整数
        # 分布在(batch_size, 10)形状的矩阵中
        data = torch.from_numpy(np.random.randint(1, V, size=(batch_size, 10)))

        # 将数据的的第一列置为1, 作为起始标志列
        # 当解码器进行第一次解码的时候, 会使用起始标志列作为输入
        data[:, 0] = 1

        # 因为是copy任务, 所有source与target是完全相同的, 且数据样本作用变量不需要求梯度
        source = Variable(data, requires_grad=False)
        target = Variable(data, requires_grad=False)

        # 使用Batch()对source和target进行对应批次的掩码张量生成, 使用yield返回
        yield Batch(source, target)

V = 11
batch = 20
num_batch = 30

if __name__ == '__main__':
    res = data_generator(V, batch, num_batch)
    print(res)

# 使用make_model()获得模型的实例化对象
model = make_model(V, V, N=2)
# 使用get_std_opt()获得模型优化器
model_optimizer = get_std_opt(model)
# 使用LabelSmoothing()获得标签平滑对象
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
# 使用SimpleLossCompute()获得利用标签平滑结果的损失计算方法
loss = SimpleLossCompute(model.generator, criterion, model_optimizer)

# # 使用LabelSmoothing实例化crit对象.
# # size: 目标数据的词汇总数, 即模型最后一层所得张量的最后一维大小
# # padding_idx: 将tensor中的数字替换成0, padding_idx=0表示不进行替换
# # smoothing: 标签的平滑程度. 如原来标签值为1, 则平滑后的值域变为[1-smoothing, 1+smoothing]
# crit = LabelSmoothing(size=5, padding_idx=0, smoothing=0.5)

# # 假定一个任意的模型最后输出预测结果和真实结果
# predict = Variable(torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
#                                       [0, 0.2, 0.7, 0.1, 0],
#                                       [0, 0.2, 0.7, 0.1, 0]]))

# target = Variable(torch.LongTensor([2, 1, 0]))
# crit(predict, target)
# # 绘制标签平滑图像
# plt.imshow(crit.true_dist)
# plt.show()

def run(model, loss, epochs=10):
    """
    模型训练函数
    model:  将要进行训练的模型
    loss:   使用的损失计算方法
    epochs: 模型训练轮次数
    """
    for epoch in range(epochs):
        # 使用训练模式, 所有参数将被更新
        model.train()
        # 训练时, 传入的batch_size为20
        run_epoch(data_generator(V, 8, 20), model, loss)
        # 使用评估模式, 参数将固定不变
        model.eval()
        # 评估时, 传入的batch_size为5
        run_epoch(data_generator(V, 8, 5), model, loss)

    model.eval()

    # 初始化一个输入张量
    source = Variable(torch.LongTensor([[1,3,2,5,4,6,7,8,9,10]]))

    # 初始化源数据掩码张量, 全1代表没有任何遮掩
    source_mask = Variable(torch.ones(1, 1, 10))

    # max_len: 解码的最大长度限制, 默认为10
    # start_symbol: 起始标志数字, 默认为1
    result = greedy_decode(model, source, source_mask, max_len=10, start_symbol=1)
    print(result)

# model和loss都是来自上一步的结果
if __name__ == '__main__':
    run(model, loss)
