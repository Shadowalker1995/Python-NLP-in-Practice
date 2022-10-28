# 从io工具包导入open方法
from io import open
# 用于字符规范化
import unicodedata
# 用于正则表达式
import re
import os
# 用于随机生成数据
import random
# 时间和数学工具包
import time
import math
# 导入plt以便绘制损失曲线
import matplotlib.pyplot as plt
# 用于构建网络结构和函数的torch工具包
import torch
import torch.nn as nn
import torch.nn.functional as F
# torch中预定义的优化方法工具包
from torch import optim
# 设备选择, 我们可以选择在cuda或者cpu上运行你的代码
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 起始标志
SOS_token = 0
# 结束标志
EOS_token = 1

class Lang():
    def __init__(self, name):
        """
        name: 所传入某种语言的名字
        """
        self.name = name
        # 初始化词汇对应自然数值的字典
        self.word2index = {}
        # 初始化自然数值对应词汇的字典, 其中0, 1对应的SOS和EOS已经在里面了
        self.index2word = {0: "SOS", 1: "EOS"}
        # 初始化词汇对应的自然数索引, 这里从2开始, 因为0, 1已经被开始和结束标志占用了
        self.n_words = 2

    def addSentence(self, sentence):
        """添加句子函数, 将整个句子中所有的单词依次添加到字典中"""
        # 根据一般国家的语言特性(我们这里研究的语言都是以空格分个单词), 直接进行分词就可以
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        """添加词汇函数, 即将词汇转化为对应的数值, 输入参数word是一个单词"""
        # 首先判断word是否已经在self.word2index字典的key中
        if word not in self.word2index:
            # 如果不在, 则将这个词加入其中, 并为它对应一个数值, 即self.n_words
            self.word2index[word] = self.n_words
            # 同时也将它的反转形式加入到self.index2word中
            self.index2word[self.n_words] = word
            # self.n_words一旦被占用之后, 逐次加1, 变成新的self.n_words
            self.n_words += 1

name = "eng"
sentence = "hello I am Jay"

engl = Lang(name)
engl.addSentence(sentence)
# print("word2index:", engl.word2index)
# print("index2word:", engl.index2word)
# print("n_words:", engl.n_words)

# 将unicode转为Ascii, 去掉一些语言中的重音标记
def unicodeToAscii(s):
    return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn')

def normalizeString(s):
    """字符串规范化函数, 参数s代表传入的字符串"""
    # 使字符变为小写并去除两侧空白符, z再使用unicodeToAscii去掉重音标记
    s = unicodeToAscii(s.lower().strip())
    # 在.!?前加一个空格
    s = re.sub(r"([.!?])", r" \1", s)
    # 使用正则表达式将字符串中不是大小写字母和正常标点的都替换成空格
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

s = "Are you kidding me?"
res = normalizeString(s)
# print(res)

data_path = './data/eng-fra.txt'

def readLangs(lang1, lang2):
    """
    读取语言函数
    lang1: 源语言的名字
    lang2: 目标语言的名字
    返回对应的class Lang对象, 以及语言对列表
    """
    # 从文件中读取语言对并以/n划分存到列表lines中
    lines = open(data_path, encoding='utf-8').read().strip().split('\n')
    # 对lines列表中的句子进行标准化处理, 并以\t进行再次划分, 形成子列表, 也就是语言对
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    # 然后分别将语言名字传入Lang类中, 获得对应的语言对象, 返回结果
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)
    return input_lang, output_lang, pairs

lang1 = "eng"
lang2 = "fra"

input_lang, output_lang, pairs = readLangs(lang1, lang2)
# print("input_lang:", input_lang)
# print("output_lang:", output_lang)
# print("pairs中的前五个:", pairs[:5])

# 设置组成句子中单词或标点的最多个数
MAX_LENGTH = 10

# 选择带有指定前缀的语言特征数据作为训练数据
eng_prefixes = (
        "i am ", "i m ",
        "he is", "he s ",
        "she is", "she s ",
        "you are", "you re ",
        "we are", "we re ",
        "they are", "they re ")

def filterPair(pair):
    """
    语言对过滤函数
    pair: 输入的语言对, 如['she is afraid.', 'elle malade.']
    """
    # pair[0]代表英文源语句, 它的长度应小于最大长度MAX_LENGTH并且要以指定的前缀开头
    # pair[1]代表法文源语句, 它的长度应小于最大长度MAX_LENGTH
    return len(pair[0].split(' ')) < MAX_LENGTH and \
            pair[0].startswith(eng_prefixes) and \
            len(pair[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    """
    对多个语言对列表进行过滤
    pairs: 语言对组成的列表, 简称语言对列表
    """
    # 函数中直接遍历列表中的每个语言对并调用filterPair()即可
    return [pair for pair in pairs if filterPair(pair)]

# 输入参数pairs使用readLangs函数的输出结果pairs
fpairs = filterPairs(pairs)
# print("过滤后的pairs前五个:", fpairs[:5])

def prepareData(lang1, lang2):
    """
    数据准备函数, 作用是将所有字符串数据向数值型数据的映射以及过滤语言对
    lang1, lang2分别代表源语言和目标语言的名字
    """
    input_lang, output_lang, pairs = readLangs(lang1, lang2)
    pairs = filterPairs(pairs)
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepareData('eng', 'fra')
# print("input_n_words:", input_lang.n_words)
# print("output_n_words:", output_lang.n_words)
# print(random.choice(pairs))

def tensorFromSentence(lang, sentence):
    """
    将句子文本转换为张量
    lang:     Lang的实例化对象
    sentence: 预转换的句子
    """
    # 遍历句子中的每一个词汇, 并取得其在lang.word2index()中对应的索引
    indexes = [lang.word2index[word] for word in sentence.split(' ')]
    # 加入句子结束标志
    indexes.append(EOS_token)
    # 使用torch.tensor封装成n*1张量, 方便后续计算
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    """将语言对转换为张量对"""
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

pair = pairs[0]

pair_tensor = tensorsFromPair(pair)
# print(pair_tensor)

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        input_size: 解码器的输入尺寸 (源语言词表大小)
        hidden_size: 隐层节点数. 即GRU层的输入尺寸 或 词嵌入维度
        """
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # 实例化nn中预定义的Embedding层, input_size为源语言词表大小,
        # hidden_size为源语言词嵌入维度
        self.embedding = nn.Embedding(input_size, hidden_size).to(device)
        # 实例化nn中预定义的GRU层, 由于上层的输出维度是hidden_size
        # 因此GRU层的输入维度也是hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size).to(device)

    def forward(self, x, hidden):
        """
        x:      源语言的Embedding层输入张量
        hidden: 编码器gru层的初始隐层张量
        """
        # 理论上, 编码器每次只以一个词作为输入, embedding后的尺寸应该是 1*embedding_size
        # 而torch中预定义gru必须使用三维张量作为输入, 因此需要拓展一个维度
        output = self.embedding(x).view(1, 1, -1)
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        """初始化隐层张量函数"""
        return torch.zeros(1, 1, self.hidden_size, device=device)

hidden_size = 25
input_size = 20

# pair_tensor[0]为源语言即英文的句子, pair_tensor[0][0]为句子中的第一个词
x = pair_tensor[0][0]
# 按1*1*hidden_size的全0张量初始化第一个隐层张量
hidden = torch.zeros(1, 1, hidden_size, device=device)

encoder = EncoderRNN(input_size, hidden_size)
encoder_output, hidden = encoder(x, hidden)
# print(encoder_output)
# # 1*1*25
# print(encoder_output.shape)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        """
        hidden_size: 隐层节点数. 即GRU层的输入尺寸
        output_size: 解码器的输出尺寸 (目标语言词表大小)
        """
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # 实例化nn中预定义的Embedding层, output_size为目标语言的词表大小
        # hidden_size为目标语言词嵌入维度
        self.embedding = nn.Embedding(output_size, hidden_size).to(device)
        # 实例化nn中预定义的GRU层, 由于上层的输出维度是hidden_size
        # 因此GRU层的输入维度也是hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size).to(device)
        # 实例化线性层, output_size为希望的输出尺寸
        self.out = nn.Linear(hidden_size, output_size).to(device)
        self.softmax = nn.LogSoftmax(dim=-1).to(device)

    def forward(self, x, hidden):
        """
        x:      目标语言的Embedding层输入张量
        hidden: 解码器GRU层的初始隐层张量
        """
        # 对输入张量进行embedding操作
        output = self.embedding(x).view(1, 1, -1)
        # 根据relu函数的特性, 将使Embedding矩阵更稀疏, 以防止过拟合
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        # GRU层的输出output为三维张量, 第一维没有意义, 可以通过output[0]来降维
        # 再经线性层变换, 传入softmax层处理以便于分类
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        """初始化隐层张量函数"""
        return torch.zeros(1, 1, self.hidden_size, device=device)

hidden_size = 25
output_size = 10

# pair_tensor[1]为目标语言即法文的句子, pair_tensor[1][0]为句子中的第一个词
x = pair_tensor[1][0]
# 按1*1*hidden_size的全0张量初始化第一个隐层张量
hidden = torch.zeros(1, 1, hidden_size, device=device)

decoder = DecoderRNN(hidden_size, output_size)
output, hidden = decoder(x, hidden)
# print(output)
# # 1*10
# print(output.shape)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        """
        hidden_size: 隐层节点数. 即GRU层的输入尺寸
        output_size: 解码器的输出尺寸 (目标语言词表大小)
        dropout_p:   置零比率, 默认0.1
        max_length:  句子的最大长度
        """
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        # 实例化nn中预定义的Embedding层
        self.embedding = nn.Embedding(output_size, hidden_size).to(device)

        # 根据attention的QKV理论, attention的输入参数为三个, Q, K, V
        # ### 第一步 ###
        # 使用Q与K进行attention权值计算得到权重矩阵, 再与V做矩阵乘法, 得到V的注意力表示结果
        # 常见的计算方式有三种:
        # 1. 将Q, K进行纵轴拼接, 做一次线性变化, 最后使用softmax处理获得结果最后与V做张量乘法
        # 2. 将Q, K进行纵轴拼接, 做一次线性变化后使用tanh函数激活, 然后进行内部求和, 最后使用softmax处理获得结果再与V做张量乘法
        # 3. 将Q与K的转置做点积运算, 然后除以一个缩放系数, 最后使用softmax处理获得结果最后与V做张量乘法
        # 说明：当注意力权重矩阵和V都是三维张量且第一维代表为batch条数时, 则做bmm运算
        # ### 第二步 ###
        # 根据第一步采用的计算方法, 如果是拼接方法, 则需要将Q与第一步的计算结果再进行拼接
        # 如果是转置点积, 一般是自注意力 (Q与V相同), 则不需要进行与Q的拼接. 因此第二步的计算方式与第一步采用的全值计算方法有关
        # ### 第三步 ###
        # 最后为了使整个attention结构按照指定尺寸输出, 使用线性层在第二步的结果上做一个线性变换, 得到最终对Q的注意力表示

        # 这里使用第一种计算方式, 因此需要实例化一个线性变换的矩阵
        # 由于输入是Q, K的纵轴拼接, 所以nn.Linear()的第一个参数为hidden_size*2, 第二个参数为max_length
        # Q为解码器Embedding层的输出. K为解码器GRU的隐层输出, 因为首次隐层还没有任何输出, 会使用编码器的隐层输出
        # V为编码器层的输出
        self.attn = nn.Linear(hidden_size*2, max_length).to(device)
        # 实例化nn.Dropout层, 并传入self.dropout_p
        self.dropout = nn.Dropout(dropout_p).to(device)

        # 实例化另外一个线性层, 用于规范输出尺寸, 其输入来自第三步的结果 (将Q与第二步的结果进行拼接)
        # 输入维度是self.hidden_size*2
        self.attn_combine = nn.Linear(hidden_size*2, hidden_size).to(device)
        # 实例化nn.GRU层, 它的输入和隐层尺寸都是self.hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size).to(device)
        # 实例化gru后面的线性层, 也即解码器输出层
        self.out = nn.Linear(hidden_size, output_size).to(device)

    def forward(self, x, hidden, encoder_outputs):
        """
        x:      目标语言的Embedding层输入张量, [1]
        hidden: 解码器GRU层的初始隐层张量, 1*1*hidden_size
        encoder_outputs: 解码器的输出张量, max_length*hidden_size
        """
        # 对输入张量进行embedding操作
        # 1 -> hidden_size -> 1*1*hidden_size
        embedded = self.embedding(x).view(1, 1, -1)
        # 使用dropout进行随机丢弃, 防止过拟合
        embedded = self.dropout(embedded)

        # ### 第一步 ###
        # 使用第一种计算方式进行attention的权重计算
        # 将Q, K进行纵轴拼接, 并做一次线性变化, 最后使用softmax处理获得结果
        # attn_weights: 1*hidden_size cat 1*hidden_size -> 1*2xhidden_size -> 1*max_length
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=-1)
        # 将得到的权重矩阵与V做矩阵乘法计算, 当二者都是三维张量且第一维代表为batch条数时, 则做bmm运算
        # 1*1*max_length bmm 1*max_length*hidden_size -> 1*1*hidden_size
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        # ### 第二步 ###
        # 通过取[0]降维, 根据第一步采用的计算方法, 将Q与第一步的计算结果再进行拼接
        # 1*hidden_size cat 1*hidden_size -> 1*2xhidden_size
        output = torch.cat((embedded[0], attn_applied[0]), 1)

        # ### 第三步 ###
        # 在第二步的输出结果上做一个线性变换并扩展维度, 得到输出
        # 1*2xhidden_size -> 1*hidden_size -> 1*1*hidden_size
        output = self.attn_combine(output).unsqueeze(0)

        # attention结构的结果使用relu激活
        output = F.relu(output)
        # 将激活后的结果作为gru的输入和hidden一起传入其中
        output, hidden = self.gru(output, hidden)
        # 将结果降维并使用softmax处理得到最终的结果
        # 1*1*hidden_size -> 1*hidden_size -> 1*output_size
        output = F.log_softmax(self.out(output[0]), dim=-1)

        # 返回解码器结果, 最后的隐层张量以及注意力权重张量
        return output, hidden, attn_weights

    def iniHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

hidden_size = 25
output_size = 10

x = pair_tensor[1][0]
hidden = torch.zeros(1, 1, hidden_size, device=device)
# encoder_outputs需要是encoder中每一个时间步的输出堆叠而成
# 它的形状应该是max_length*hidden_size, 10*25
encoder_outputs = torch.randn(10, 25, device=device)

decoder = AttnDecoderRNN(hidden_size, output_size)
output, hidden, attn_weights = decoder(x, hidden, encoder_outputs)
# print(output)
# # 1*10
# print(output.shape)
# # 1*1*25
# print(hidden.shape)
# # 1*10
# print(attn_weights.shape)

teacher_forcing_ratio = 0.5
def train(input_tensor, target_tensor, encoder, decoder,
        encoder_optimizer, decoder_optimizer, criterion,
        max_length=MAX_LENGTH):
    """
    input_tensor:      源语言输入张量
    target_tensor:     目标语言输入张量
    encoder, decoder:  编码器和解码器实例化对象
    encoder_optimizer: 编码器优化方法
    decoder_optimizer: 解码器优化方法
    criterion:         损失函数计算方法
    max_length:        句子的最大长度
    """
    # 初始化隐层张量
    encoder_hidden = encoder.initHidden()
    # 编码器和解码器的优化器梯度归0
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    # 根据源文本和目标文本张量获得对应的长度
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    # 以max_length*encoder.hidden_size的全0张量初始化编码器输出张量
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    # 初始设置损失
    # loss = 0
    loss = criterion(torch.tensor([[0.]], device=device), torch.tensor([0], device=device))

    for ei in range(input_length):
        # 根据索引从input_tensor取出对应单词的张量表示, 和初始化隐层张量一同传入encoder对象中
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        # encoder_output为三维张量, 使用[0]降维变成向量依次存入到encoder_outputs
        # encoder_outputs的每一行为对应的句子中每个单词通过编码器的输出结果 (1*1*encoder.hidden_size)
        encoder_outputs[ei] = encoder_output[0]

    # 初始化解码器的第一个输入, 即起始符
    decoder_input = torch.tensor([SOS_token], device=device)
    # 初始化解码器的隐层张量即编码器的隐层输出
    decoder_hidden = encoder_hidden
    # 根据随机数与teacher_forcing_ratio对比判断是否使用teacher_forcing
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # 循环遍历目标张量索引
        for di in range(target_length):
            # 将decoder_input, decoder_hidden, encoder_outputs即attention中的QKV传入解码器对象
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            # 由于使用了teacher_forcing, 无论解码器输出的decoder_output是什么
            # 都只使用 '正确答案', 即target_tensor[di]来计算损失
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            # 从decoder_output取出答案
            topv, topi = decoder_output.topk(1)
            # 损失计算仍然使用decoder_output和target_tensor[di]
            loss += criterion(decoder_output, target_tensor[di])
            # 如果输出值是终止符, 则循环停止
            if topi.squeeze().item() == EOS_token:
                break
            # 否则, 对topi降维并分离赋值给decoder_input以便进行下次运算
            # detach的分离作用使得这个decoder_input与模型构建的张量图无关, 相当于全新的外界输入
            decoder_input = topi.squeeze().detach()

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    # 返回平均损失
    return loss.item() / target_length

def timeSince(since):
    "获得每次打印的训练耗时, since是训练开始时间"
    # 获得当前时间
    now = time.time()
    # 获得时间差, 就是训练耗时
    s = now - since
    # 将秒转化为分钟, 并取整
    m = math.floor(s / 60)
    # 计算剩下不够凑成1分钟的秒数
    s -= m * 60
    # 返回指定格式的耗时
    return '%dm %ds' % (m, s)

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    """
    训练迭代函数
    encoder, decoder: 编码器和解码器对象
    n_iters:          总迭代步数
    print_every:      打印日志间隔
    plot_every:       绘制损失曲线间隔
    learning_rate:    学习率
    """
    ENCODER_PATH = './models/translator_eng2fra_encoder.pth'
    DECODER_PATH = './models/translator_eng2fra_decoder.pth'
    if not (os.path.exists(ENCODER_PATH) and os.path.exists(DECODER_PATH)):
        # 获得训练开始时间戳
        start = time.time()
        # 每个损失间隔的平均损失保存列表, 用于绘制损失曲线
        plot_losses = []
        # 每个打印日志间隔的总损失, 初始为0
        print_loss_total = 0
        # 每个绘制损失间隔的总损失, 初始为0
        plot_loss_total = 0

        # 使用预定义的SGD作为优化器, 将参数和学习率传入其中
        encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

        # 选择损失函数
        criterion = nn.NLLLoss()

        for iter in range(1, n_iters+1):
            # 每次从语言对列表中随机取出一条作为训练语句
            training_pair = tensorsFromPair(random.choice(pairs))
            # 分别从training_pair中取出输入张量和目标张量
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            # 通过train函数获得模型运行的损失
            loss = train(input_tensor, target_tensor, encoder, decoder,
                    encoder_optimizer, decoder_optimizer,
                    criterion)
            # 将损失进行累和
            print_loss_total += loss
            plot_loss_total = loss

            if iter % print_every == 0:
                # 通过总损失除以间隔得到平均损失
                print_loss_avg = print_loss_total / print_every
                # 总损失归0
                print_loss_total = 0
                # 打印日志, 日志内容分别是: 训练耗时, 当前迭代步, 当前进度百分比, 当前平均损失
                print('%s (%d %d%%) %.4f' % (timeSince(start),
                    iter, iter / n_iters * 100, print_loss_avg))

            if iter % plot_every == 0:
                # 通过总损失除以间隔得到平均损失
                plot_loss_avg = plot_loss_total / plot_every
                # 将平均损失装进plot_losses列表
                plot_losses.append(plot_loss_avg)
                # 总损失归0
                plot_loss_total = 0

        torch.save(encoder.state_dict(), ENCODER_PATH)
        torch.save(attn_decoder.state_dict(), DECODER_PATH)

        # 绘制损失曲线
        plt.figure()
        plt.plot(plot_losses)
        # 保存到指定路径
        plt.savefig("./figures/s2s_loss.png")
    else:
        encoder.load_state_dict(torch.load(ENCODER_PATH))
        attn_decoder.load_state_dict(torch.load(DECODER_PATH))

# 设置隐层大小为256 , 也是词嵌入维度
hidden_size = 256
# 通过input_lang.n_words获取输入词汇总数
encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
# 通过output_lang.n_words获取目标词汇总数
attn_decoder = AttnDecoderRNN(hidden_size, output_lang.n_words,
        dropout_p=0.1).to(device)

n_iters = 75000
print_every = 5000

# 调用trainIters进行模型训练
trainIters(encoder, attn_decoder, n_iters, print_every=print_every)

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    """
    评估函数
    encoder, decoder: 编码器和解码器对象,
    sentence:         需要评估的句子
    max_length:       句子的最大长度
    """
    # 评估阶段不进行梯度计算
    with torch.no_grad():
        # 对输入的句子进行张量表示
        input_tensor = tensorFromSentence(input_lang, sentence)
        # 获得输入的句子长度
        input_length = input_tensor.size(0)
        # 初始化编码器隐层张量
        encoder_hidden = encoder.initHidden()

        # 以max_length*encoder.hidden_size的全0张量初始化编码器输出张量
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            # 根据索引从input_tensor取出对应单词的张量表示, 和初始化隐层张量一同传入encoder对象中
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            # encoder_output为三维张量, 使用[0]降两维变成向量依次存入到encoder_outputs
            # encoder_outputs的每一行为对应的句子中每个单词通过编码器的输出结果 (1*1*encoder.hidden_size)
            encoder_outputs[ei] = encoder_output[0]

        # 初始化解码器的第一个输入, 即起始符
        decoder_input = torch.tensor([SOS_token], device=device)
        # 初始化解码器的隐层张量即编码器的隐层输出
        decoder_hidden = encoder_hidden

        # 初始化预测的词汇列表
        decoded_words = []
        # 初始化attention张量
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            # 取所有的attention结果存入初始化的attention张量中
            decoder_attentions[di] = decoder_attention.data
            # 从解码器输出中获得概率最高的值及其索引对象
            topv, topi = decoder_output.topk(1)
            # 从索引对象中取出它的值与结束标志值作对比
            if topi.item() == EOS_token:
                # 如果是结束标志值, 则将结束标志装进decoded_words列表, 代表翻译结束
                decoded_words.append('<EOS>')
                break
            else:
                # 否则, 根据索引找到它在输出语言的index2word字典中对应的单词装进decoded_words
                decoded_words.append(output_lang.index2word[topi.item()])
            # 最后将本次预测的索引降维并分离赋值给decoder_input, 以便下次进行预测
            decoder_input = topi.squeeze().detach()
        # 返回decoded_words, 以及完整注意力张量, 把没有用到的部分切掉
        return decoded_words, decoder_attentions[:di+1]

def evaluateRandomly(encoder, decoder, n=6):
    """随机测试函数"""
    for i in range(n):
        pair = random.choice(pairs)
        # > 代表输入
        print('>', pair[0])
        # = 代表正确的输出
        print('=', pair[1])
        # 调用evaluate进行预测
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        # 将结果连成句子
        output_sentence = ' '.join(output_words)
        # < 代表模型的输出
        print('>', output_words)
        print('')

evaluateRandomly(encoder, attn_decoder)

sentence = "we re both teachers ."
output_words, attentions = evaluate(encoder, attn_decoder, sentence)
print(output_words)
plt.matshow(attentions.numpy())
plt.savefig("./figures/s2s_attn.png")
