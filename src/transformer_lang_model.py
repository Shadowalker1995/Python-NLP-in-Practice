import math
import time
import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# torch中经典文本数据集有关的工具包
import torchtext
# 英文分词工具get_tokenizer
from torchtext.data.utils import get_tokenizer
# 已经构建完成的TransformerModel
from pyitcast.transformer import TransformerModel


# 创建语料域, 存放语料的数据结构
# tokenize: 使用get_tokenizer("basic_english")获得一个按照文本为基础英文进行分割的分割器对象, 代表给存放语料（或称作文本）施加的作用
# init_token: 对文本施加的起始符
# eos_token: 对文本施加的终止符
# lower: 存放的文本字母全部小写
# 最终获得一个Field对象
TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)
# print(TEXT)

# 使用torchtext的数据集方法导入WikiText2数据
# 切分为对应训练文本, 验证文本, 测试文本, 并对这些文本施加刚刚创建的语料域
train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
# print(test_txt.examples[0].text[:10])
# 将训练集文本数据构建一个vocab对象. 后续可以调用vocab对象的stoi()方法统计文本共包含的不重复词汇总数
TEXT.build_vocab(train_txt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batchify(data, bsz):
    """
    构建批次数据的函数, 用于将文本数据映射成连续数字, 并转换成指定的样式
    data: 之前得到的文本数据 (train_txt, val_txt, test_txt)
    bsz:  batch_size
    """
    # 使用TEXT的numericalize()方法将单词映射成对应的连续数字
    data = TEXT.numericalize([data.examples[0].text])
    # print(data)

    # 遍历完所有数据所需要的batch数量
    nbatch = data.size(0) // bsz
    # 使用narrow()方法删除不规整的剩余数据
    # 第一个参数: 横轴删除(0)还是纵轴删除(1)
    # 第二个和第三个参数: 切割的起始位置和终止位置, 类似于切片
    data = data.narrow(0, 0, nbatch * bsz)
    # print(data)

    # 使用view()方法对data进行矩阵变换 (形状为[None, bsz]), 紧接着进行转置操作
    # 如果输入是训练数据, 形状为[104335, 20], 即data的列数等于bsz
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)
# batchify(test_txt, 20)

batch_size = 20
eval_batch_size = 10

train_data = batchify(train_txt, batch_size)
val_data = batchify(val_txt, eval_batch_size)
test_data = batchify(test_txt, eval_batch_size)

# 设定句子的最大长度
bptt = 35

def get_batch(source, i):
    """
    获得每个批次合理大小的源数据和目标数据
    source: train_data/val_data/test_data
    i:      具体的批次数
    """
    # 确定句子长度, 取bptt和len(source)-1-i中最小值
    # 实质上, 前面的批次中都会是bptt的值, 只不过最后一个批次中, 句子长度
    # 可能不够bptt的35个, 因此会变为len(source)-1-i
    seq_len = min(bptt, len(source)-1-i)

    data = source[i:i+seq_len]

    # 根据语言模型训练的语料规定, 目标数据是源数据向后移动一位
    # 因为最后目标数据的切片会越界, 使用view(-1)来保证形状正常
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

source = test_data
i = 1
x, y = get_batch(source, i)
# print(x)
# print(y)

# 通过TEXT.vocab.stoi()方法获得不重复词汇总数
ntokens = len(TEXT.vocab.stoi)
# 词嵌入大小为200
emsize = 200
# 前馈全连接层的节点数
nhid = 200
# 编码器层的数量
nlayers = 2
# 多头注意力机制的头数
nhead = 2
dropout = 0.2

# 将参数输入到TransformerModel中实例化模型
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()
lr = 5.0
# SGD优化器
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# 学习率调整器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

def train():
    model.train()
    total_loss = 0
    log_interval = 200
    start_time = time.time()

    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        # 通过get_batch()方法获得源数据和目标数据
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        # 使用nn自带的clip_grad_norm_()方法进行梯度规范化, 防止出现梯度消失或爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()

    if batch % log_interval == 0 and batch > 0:
        cur_loss = total_loss / log_interval
        elapsed = time.time() - start_time
        # 打印轮数, 当前批次, 总批次, 当前学习率, 训练速度(每豪秒处理多少批次),
        # 平均损失, 以及困惑度. 困惑度是衡量语言模型的重要指标, 其计算方法是
        # 对交叉熵平均损失取自然对数的幂
        print('| epoch {:3d} | {:5d}/{:5d} batches | '
              'lr {:02.2f} | ms/batch {:5.2f} | '
              'loss {:5.2f} | ppl {:8.2f}'.format(
                  epoch, batch, len(train_data) // bptt, scheduler.get_last_lr()[0],
                  elapsed * 1000 / log_interval,
                  cur_loss, math.exp(cur_loss)))

        total_loss = 0
        start_time = time.time()

def evaluate(eval_model, data_source):
    """
    eval_model:  每轮训练产生的模型
    data_source: 验证或测试数据集
    """
    eval_model.eval()
    total_loss = 0

    with torch.no_grad():
        # 与训练过程相同, 但是因为过程不需要打印信息, 因此不需要batch数
        for i in range(0, data_source.size(0) - 1, bptt):
            # 通过get_batch()方法获得源数据和目标数据
            data, targets = get_batch(data_source, i)
            output = eval_model(data)
            # 对输出形状扁平化, 变为全部词汇的概率分布
            output_flat = output.view(-1, ntokens)
            total_loss += criterion(output_flat, targets).item()
            cur_loss = total_loss / ((data_source.size(0) - 1) / bptt)

    return cur_loss

best_val_loss = float("inf")
epochs = 3
best_model = None
best_model_path = './models/transformer_lang.pth'

if not os.path.exists(best_model_path):
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(model, val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 使用深拷贝, 拷贝最优模型
            best_model = copy.deepcopy(model)
            torch.save(model.state_dict(), best_model_path)
    # 每轮都会对优化方法的学习率做调整
            scheduler.step()
else:
    model.load_state_dict(torch.load(best_model_path))

best_model = model
test_loss = evaluate(best_model, test_data)

print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
