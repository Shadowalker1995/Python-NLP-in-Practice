from io import open
import glob
import os
import string
import unicodedata
import random
import time
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# 所有常用字符包括字母和常用标点
all_letters = string.ascii_letters + " .,;'"
# 常用字符数量
n_letters = len(all_letters)
# n_letter: 57
# print("n_letter:", n_letters)

# 去掉一些语言中的重音标记
# 如: Ślusàrski ---> Slusarski
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in all_letters)

data_path = "./data/names/"

def readLines(filename):
    """从文件中读取每一行加载到内存中形成列表"""
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

# filename = data_path + "Chinese.txt"
# result = readLines(filename)
# ['Ang', 'AuYong', 'Bai', 'Ban', 'Bao', 'Bei', 'Bian', 'Bui', 'Cai',
# 'Cao', 'Cen', 'Chai', 'Chaim', 'Chan', 'Chang', 'Chao', 'Che',
# 'Chen', 'Cheng', 'Cheung']
# print(result[:20])

# 构建人名类别与具体人名对应关系的字典. 形如：{"English":["Lily", "Susan", "Kobe"], "Chinese":["Zhang San", "Xiao Ming"]}
category_lines = {}
# 构建所有类别的列表. 形如： ["English",...,"Chinese"]
all_categories = []

# 读取指定路径下的txt文件, glob支持正则表达式
for filename in glob.glob(data_path + "*.txt"):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    # 读取每个文件的内容, 形成名字列表
    lines = readLines(filename)
    # 按照对应的类别, 将名字列表写入到category_lines字典中
    category_lines[category] = lines

n_categories = len(all_categories)

# n_categories 18
# print("n_categories", n_categories)

# ['Abandonato', 'Abatangelo', 'Abatantuono', 'Abate', 'Abategiovanni',
# 'Abatescianni', 'Abba', 'Abbadelli', 'Abbascia', 'Abbatangelo']
# print(category_lines['Italian'][:10])

def lineToTensor(line):
    """将人名转化为对应onehot张量表示, 参数line是输入的人名"""
    # 初始化一个形状为len(line)*1*n_letters的全0张量
    # 代表人名中的每个字母用一个1*n_letters的张量表示
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        # 使用字符串的find()方法得到每个字符在all_letters中的索引
        # 也是所生成onehot张量中1的索引位置
        tensor[li][0][all_letters.find(letter)] = 1

    return tensor

# line = "Bai"
# line_tensor = lineToTensor(line)
# print("line_tensor:", line_tensor)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
            num_layers=1):
        """
        input_size:     RNN输入的最后一个维度
        hidden_size:    RNN隐藏层的最后一个维度
        output_size:    RNN网络最后线性层的输出维度
        num_layers:     RNN网络的层数
        """
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # 实例化预定义的nn.RNN
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        # 实例化全连接线性层, 将nn.RNN的输出维度转化为指定的输出维度
        self.linear = nn.Linear(hidden_size, output_size)
        # 实例化nn中预定的Softmax层, 用于从输出层获得类别结果
        self.softmax = nn.LogSoftmax(dim=-1)

    def initHidden(self):
        """初始化隐层张量"""
        # self.num_layers*1*self.hidden_size
        return torch.zeros(self.num_layers, 1, self.hidden_size)

    def forward(self, x, hidden):
        """
        x:      人名分类器中的输入张量 (1*n_letters)
        hidden: RNN的隐藏层张量 (self.num_layers*1*self.hidden_size)
        """
        # nn.RNN要求输入是三维张量
        x = x.unsqueeze(0)
        # 如果num_layers=1, rr恒等于hn
        rr, hn = self.rnn(x, hidden)
        # 返回hn作为后续RNN的输入
        return self.softmax(self.linear(rr)), hn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
            num_layers=1):
        """
        input_size:     LSTM输入的最后一个维度
        hidden_size:    LSTM隐藏层的最后一个维度
        output_size:    LSTM网络最后线性层的输出维度
        num_layers:     LSTM网络的层数
        """
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # 实例化预定义的nn.LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        # 实例化全连接线性层, 将nn.LSTM的输出维度转化为指定的输出维度
        self.linear = nn.Linear(hidden_size, output_size)
        # 实例化nn中预定的Softmax层, 用于从输出层获得类别结果
        self.softmax = nn.LogSoftmax(dim=-1)

    def initHiddenAndC(self):
        """初始化函数不仅初始化hidden还要初始化细胞状态c, 它们形状相同"""
        c = hidden = torch.zeros(self.num_layers, 1, self.hidden_size)
        return hidden, c

    def forward(self, x, hidden, c):
        """注意: LSTM网络的输入有3个张量，尤其不要忘记细胞状态c"""
        x = x.unsqueeze(0)
        rr, (hn, c) = self.lstm(x, (hidden, c))
        return self.softmax(self.linear(rr)), hn, c


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
            num_layers=1):
        """
        input_size:     GRU输入的最后一个维度
        hidden_size:    GRU隐藏层的最后一个维度
        output_size:    GRU网络最后线性层的输出维度
        num_layers:     GRU网络的层数
        """
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)

    def forward(self, x, hidden):
        x = x.unsqueeze(0)
        rr, hn = self.gru(x, hidden)
        return self.softmax(self.linear(rr)), hn


# 因为是onehot编码, 输入张量最后一维的尺寸为n_letters
input_size = n_letters
# 定义隐层的最后一维尺寸
n_hidden = 128
# 输出尺寸为语言类别总数n_categories
output_size = n_categories
# num_layer使用默认值, num_layers = 1

# 假如以一个字母B作为RNN的首次输入, 它通过lineToTensor转为张量
# 而lineToTensor输出是三维张量, RNN类需要二维张量
# 因此需要使用squeeze(0)降低一个维度
x = lineToTensor('B').squeeze(0)
# 初始化一个三维的隐层0张量, 也是初始的细胞状态张量
hidden = c = torch.zeros(1, 1, n_hidden)

rnn = RNN(input_size, n_hidden, output_size)
lstm = LSTM(input_size, n_hidden, output_size)
gru = GRU(input_size, n_hidden, output_size)

rnn_output, next_hidden = rnn(x, hidden)
# print("rnn:", rnn_output)
# print("rnn_shape:", rnn_output.shape)   # 1*1*18
# print('**********')

lstm_output, next_hidden, c = lstm(x, hidden, c)
# print("lstm:", lstm_output)
# print("lstm_shape:", lstm_output.shape)   # 1*1*18
# print('**********')

gru_output, next_hidden = gru(x, hidden)
# print("gru:", gru_output)
# print("gru_shape:", gru_output.shape)   # 1*1*18
# print('**********')

def categoryFromOutput(output):
    """从输出结果中获得指定类别, 参数为输出张量output"""
    # 从输出张量中返回最大的值和索引对象
    top_n, top_i = output.topk(1)
    # top_i对象中取出索引的值
    category_i = top_i[0].item()
    # 根据索引值获得对应语言类别, 返回语言类别和索引值
    return all_categories[category_i], category_i

# category, category_i = categoryFromOutput(gru_output)
# print("category:", category)
# print("category_i:", category_i)

def randomTrainingExample():
    """该函数用于随机产生训练数据"""
    # 第一步使用random.choice()方法从all_categories中随机选择一个类别
    category = random.choice(all_categories)
    # 第二步通过category_lines字典取category类别对应的名字列表，从列表中随机取一个名字
    line = random.choice(category_lines[category])
    # 第三部将该类别在所有类别列表中的索引封装成tensor
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    # 第四步将随机取到的名字通过函数lineToTensor()转化为onehot张量表示
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

# for i in range(10):
#     category, line, category_tensor, line_tensor = randomTrainingExample()
#     print('category =', category, '/ line =', line, '/ category_tensor =', category_tensor)
# print(line_tensor)

# RNN的最后一层为nn.LogSoftmax(), 因此选定损失函数为nn.NLLLoss(), 两者的内部计算逻辑正好吻合
criterion = nn.NLLLoss()
learning_rate = 0.005

def trainRNN(category_tensor, line_tensor):
    """
    category_tensor:    类别的张量表示, 训练数据的标签,
    line_tensor:        名字的张量表示, 训练数据的标签
    """
    # 通过实例化对象rnn初始化隐层张量
    hidden = rnn.initHidden()
    # 关键的一步: 将模型结构中的梯度归0
    rnn.zero_grad()
    # 循环遍历训练数据line_tensor中的每个字符，逐个传入rnn之中, 并且迭代更新hidden
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    # 因为rnn对象由nn.RNN实例化得到, 最终输出形状是三维张量, 为了满足于category_tensor
    # 进行对比计算损失, 需要进行降维操作
    loss = criterion(output.squeeze(0), category_tensor)
    loss.backward()

    # 更新模型中所有的参数
    for p in rnn.parameters():
        # 将参数的张量表示与参数的梯度乘以学习率的结果相加以此来更新参数
        p.data.add_(p.grad.data, alpha=-learning_rate)
    # 返回结果和损失的值
    return output, loss.item()

# 与传统RNN相比多出细胞状态c
def trainLSTM(category_tensor, line_tensor):
    hidden, c = lstm.initHiddenAndC()
    lstm.zero_grad()
    for i in range(line_tensor.size()[0]):
        # 返回output, hidden以及细胞状态c
        output, hidden, c = lstm(line_tensor[i], hidden, c)
    loss = criterion(output.squeeze(0), category_tensor)
    loss.backward()

    for p in lstm.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)
    return output, loss.item()

def trainGRU(category_tensor, line_tensor):
    hidden = gru.initHidden()
    gru.zero_grad()
    for i in range(line_tensor.size()[0]):
        output, hidden= gru(line_tensor[i], hidden)
    loss = criterion(output.squeeze(0), category_tensor)
    loss.backward()

    for p in gru.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)
    return output, loss.item()

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

# 设置训练迭代次数
n_iters = 100000
# 设置结果的打印间隔
print_every = 5000
# 设置绘制损失曲线上的制图间隔
plot_every = 1000

def train(train_type_fn):
    """
    训练过程的日志打印函数
    train_type_fn: 代表选择哪种模型训练函数, 如trainRNN
    """
    # 初始化存储每个制图间隔损失的列表
    all_losses = []
    # 获得训练开始时间戳
    start = time.time()
    # 设置初始间隔损失为0
    current_loss = 0
    # 从1开始进行训练迭代, 共n_iters次
    for iter in range(1, n_iters+1):
        # 通过randomTrainingExample函数随机获取一组训练数据和对应的类别
        category, line, category_tensor, line_tensor = randomTrainingExample()
        # 将训练数据和对应类别的张量表示传入到train函数中
        output, loss = train_type_fn(category_tensor, line_tensor)
        # 计算制图间隔中的总损失
        current_loss += loss

        # 如果迭代数能够整除打印间隔
        if iter % print_every == 0:
            # 取该迭代步上的output通过categoryFromOutput函数获得对应的类别和类别索引
            guess, guess_i = categoryFromOutput(output)
            # 和真实的类别category做比较, 如果相同则打对号, 否则打叉号.
            correct = '✓' if guess == category else '✗ (%s)' % category
            # 打印迭代步, 迭代步百分比, 当前训练耗时, 损失, 该步预测的名字, 以及是否正确
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter/n_iters*100, timeSince(start), loss, line, guess, correct))

        # 如果迭代数能够整除制图间隔
        if iter % plot_every == 0:
            # 将保存该间隔中的平均损失到all_losses列表中
            all_losses.append(current_loss / plot_every)
            # 间隔损失重置为0
            current_loss = 0
    # 返回对应的总损失列表和训练耗时
    return all_losses, int(time.time() - start)

# 调用train函数, 分别进行RNN, LSTM, GRU模型的训练
# 并返回各自的全部损失, 以及训练耗时用于制图
RNN_PATH = './models/RNN_classifier.pth'
LSTM_PATH = './models/LSTM_classifier.pth'
GRU_PATH = './models/GRU_classifier.pth'
if not (os.path.exists(RNN_PATH) and os.path.exists(LSTM_PATH) and os.path.exists(GRU_PATH)):
    all_losses1, period1 = train(trainRNN)
    all_losses2, period2 = train(trainLSTM)
    all_losses3, period3 = train(trainGRU)

    torch.save(rnn.state_dict(), RNN_PATH)
    torch.save(lstm.state_dict(), LSTM_PATH)
    torch.save(gru.state_dict(), GRU_PATH)

    # 绘制损失对比曲线, 训练耗时对比柱张图
    # 创建画布0
    plt.figure(0)
    # 绘制损失对比曲线
    plt.plot(all_losses1, label="RNN")
    plt.plot(all_losses2, color="red", label="LSTM")
    plt.plot(all_losses3, color="orange", label="GRU")
    plt.legend(loc='upper left')
    plt.savefig("./figures/loss_compare.png")

    # 创建画布1
    plt.figure(1)
    x_data=["RNN", "LSTM", "GRU"]
    y_data = [period1, period2, period3]
    # 绘制训练耗时对比柱状图
    plt.bar(range(len(x_data)), y_data, tick_label=x_data)
    plt.savefig("./figures/period_compare.png")
else:
    rnn.load_state_dict(torch.load(RNN_PATH))
    lstm.load_state_dict(torch.load(LSTM_PATH))
    gru.load_state_dict(torch.load(GRU_PATH))

def evaluateRNN(line_tensor):
    """评估函数, 和训练函数逻辑相同, 参数是line_tensor代表名字的张量表示"""
    # 初始化隐层张量
    hidden = rnn.initHidden()
    # 将评估数据line_tensor的每个字符逐个传入rnn之中
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    # 获得输出结果
    return output.squeeze(0)

def evaluateLSTM(line_tensor):
    # 初始化隐层张量和细胞状态张量
    hidden, c = lstm.initHiddenAndC()
    # 将评估数据line_tensor的每个字符逐个传入lstm之中
    for i in range(line_tensor.size()[0]):
        output, hidden, c = lstm(line_tensor[i], hidden, c)
    return output.squeeze(0)

def evaluateGRU(line_tensor):
    hidden = gru.initHidden()
    # 将评估数据line_tensor的每个字符逐个传入gru之中
    for i in range(line_tensor.size()[0]):
        output, hidden = gru(line_tensor[i], hidden)
    return output.squeeze(0)

line = "Bai"
line_tensor = lineToTensor(line)

rnn_output = evaluateRNN(line_tensor)
lstm_output = evaluateLSTM(line_tensor)
gru_output = evaluateGRU(line_tensor)
print("rnn_output:", rnn_output)
print("lstm_output:", lstm_output)
print("gru_output:", gru_output)

def predict(input_line, evaluate, n_predictions=3):
    """
    input_line:     输入的字符串名字
    evaluate:       评估的模型函数, RNN, LSTM, GRU
    n_predictions:  需要取最有可能的top个结果
    """
    # 首先打印输入
    print('\n> %s' % input_line)

    # 以下操作的相关张量不进行求梯度
    with torch.no_grad():
        # 使输入的名字转换为张量表示, 并使用evaluate函数获得预测输出
        output = evaluate(lineToTensor(input_line))

        # 从预测的输出中取前3个最大的值及其索引
        topv, topi = output.topk(n_predictions, 1, True)
        # 创建盛装结果的列表
        predictions = []
        # 遍历n_predictions
        for i in range(n_predictions):
            # 从topv中取出的output值
            value = topv[0][i].item()
            # 取出索引并找到对应的类别
            category_index = topi[0][i].item()
            # 打印ouput的值, 和对应的类别
            print('(%.2f) %s' % (value, all_categories[category_index]))
            # 将结果装进predictions中
            predictions.append([value, all_categories[category_index]])
    return predictions

for evaluate_fn in [evaluateRNN, evaluateLSTM, evaluateGRU]:
    print("-"*18)
    predict('Dovesky', evaluate_fn)
    predict('Jackson', evaluate_fn)
    predict('Satoshi', evaluate_fn)
