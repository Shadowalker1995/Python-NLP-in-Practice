import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
# 导入数据集中的文本分类任务
from torchtext.datasets import text_classification
# 导入数据加载器的工具包
from torch.utils.data import DataLoader
# 导入数据的随机划分方法工具包
from torch.utils.data.dataset import random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 512

load_data_path = "./data"
if not os.path.isdir(load_data_path):
    os.mkdir(load_data_path)

# 选取torchtext包中的文本分类数据集'AG_NEWS', 即新闻主题分类数据
# 顺便将数据加载到内存中
train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](root=load_data_path)


class TextSentiment(nn.Module):
    """文本分类模型"""
    def __init__(self, vocab_size, embed_dim, num_class):
        """
        :param vocab_size: 代表整个语料包含的单词总数
        :param embed_dim:  代表词嵌入的维度
        :param num_class:  代表是文本分类的类别数
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        """初始化权重函数"""
        # 指定初始权重的取值范围数
        initrange = 0.5
        # 各层的权重使用均匀分布进行初始化
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        # 偏置初始化为0
        self.fc.bias.data.zero_()

    def forward(self, text):
        """
        :param text: 文本经数字化映射后的张量

        :return: 与类别数尺寸相同的张量, 用以判断文本类别
        """
        # m * 32, 其中m是BATCH_SIZE大小的数据中词汇总数
        embedded = self.embedding(text)
        # 将 m * 32 转化成 BATCH_SIZE * 32, 以便通过fc层后能计算相应的损失
        # 已知m的值远大于BATCH_SIZE, 用m整除BATCH_SIZE
        # 获得m中共包含c个BATCH_SIZE
        c = embedded.size(0) // BATCH_SIZE
        # 从embedded中取 BATCH_SIZE*c 个向量得到新的embedded
        # 新的embedded中的向量个数可以整除BATCH_SIZE
        embedded = embedded[:BATCH_SIZE*c]
        # 为了利用平均池化的方法求embedded中指定行数的列的平均数
        # 但平均池化方法是作用在行上的, 并且需要3维输入
        # 因此对新的embedded进行转置并拓展维度
        embedded = embedded.transpose(1, 0).unsqueeze(0)
        # 调用平均池化的方法, 并且核的大小为c, 即取每c的元素计算一次均值作为结果
        embedded = F.avg_pool1d(embedded, kernel_size=c)
        # 消除新增的维度, 然后转置回去输送给fc层
        return self.fc(embedded[0].transpose(1, 0))

# 获得整个语料包含的不同词汇总数
VOCAB_SIZE = len(train_dataset.get_vocab())
# 指定词嵌入维度
EMBED_DIM = 32
# 获得类别总数
NUM_CLASS = len(train_dataset.get_labels())
# 实例化模型
model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUM_CLASS).to(device)

def generate_batch(batch):
    """
    生成batch数据函数
    :param batch: 由样本张量和对应标签的元组组成的batch_size大小的列表
                  形如:
                  [(label1, sample1), (lable2, sample2), ..., (labelN, sampleN)]

    return: 样本张量和标签各自的列表形式(张量)
            形如:
            text = tensor([sample1, sample2, ..., sampleN])
            label = tensor([label1, label2, ..., labelN])
    """
    # 从batch中获得标签张量
    label = torch.tensor([entry[0] for entry in batch])
    # 从batch中获得样本张量
    text = [entry[1] for entry in batch]
    text = torch.cat(text)
    return text.to(device), label.to(device)

# 假设一个输入:
# batch = [(torch.tensor([3,23,2,8]), 1), (torch.tensor([3,45,21,6]), 0)]
# res = generate_batch(batch)
# print(res)

def train(train_data):
    """模型训练函数"""
    # 初始化训练损失和准确率为0
    train_loss = 0
    train_acc = 0

    # 使用数据加载器构建批次数据
    # 使用数据加载器生成BATCH_SIZE大小的数据进行批次训练
    # data为N多个generate_batch函数处理后的BATCH_SIZE大小的数据生成器
    data = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)

    # 对data进行循环遍历, 使用每个batch数据进行参数更新
    for i, (text, cls) in enumerate(data):
        # 设置优化器初始梯度为0
        optimizer.zero_grad()
        # 模型输入一个批次数据, 获得输出
        output = model(text)
        # 根据真实标签与模型输出计算损失
        try:
            loss = criterion(output, cls)
        except:
            continue
        # 将该批次的损失值累加到总损失中
        train_loss += loss.item()
        # 误差反向传播
        loss.backward()
        # 参数更新
        optimizer.step()
        # 计算该批次的准确率
        batch_acc = (output.argmax(1) == cls).sum().item()
        # 将该批次的准确率加到总准确率中
        train_acc += batch_acc
        if (i + 1) % 100 == 0:
            print('Batch {}, Acc: {}'.format(i + 1, 1.0 * batch_acc / BATCH_SIZE))

    # 调整优化器学习率
    scheduler.step()

    # 返回本轮训练的平均损失和平均准确率
    return train_loss / len(train_data), train_acc / len(train_data)

def valid(valid_data):
    """模型验证函数"""
    # 初始化验证损失和准确率为0
    valid_loss = 0
    valid_acc = 0

    # 和训练相同, 使用DataLoader获得训练数据生成器
    data = DataLoader(valid_data, batch_size=BATCH_SIZE, collate_fn=generate_batch)

    # 按批次取出数据验证
    for text, cls in data:
        # 验证阶段, 不再求解梯度
        with torch.no_grad():
            # 使用模型获得输出
            output = model(text)
            # 计算损失
            try:
                loss = criterion(output, cls)
            except:
                continue
            # 将损失和准确率加到总损失和准确率
            valid_loss += loss.item()
            valid_acc += (output.argmax(1) == cls).sum().item()

    # 返回本轮验证的平均损失和平均准确率
    return valid_loss / len(valid_data), valid_acc / len(valid_data)

# 指定训练轮数
N_EPOCHS = 20
# 定义初始的验证损失
min_valid_loss = float('inf')

# 选择预定义的交叉熵损失函数
criterion = torch.nn.CrossEntropyLoss().to(device)
# 选择随机梯度下降优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# 选择优化器步长调节方法StepLR, 用来衰减学习率
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

# 选择全部训练数据的95%作为训练集数据, 剩下的5%作为验证数据
train_len = int(len(train_dataset) * 0.95)
# print('train_len:', train_len)
# print('valid_len:', len(train_dataset) - train_len)
# 使用random_split进行乱序划分, 得到对应的训练集和验证集
sub_train_, sub_valid_ = random_split(train_dataset, [train_len, len(train_dataset) - train_len])

# 设定模型保存的路径
MODEL_PATH = './models/news_model.pth'
if not os.path.exists(MODEL_PATH):
    # 开始每一轮训练
    for epoch in range(N_EPOCHS):
        # 记录训练开始的时间
        start_time = time.time()
        # 调用train和valid函数得到训练和验证的平均损失, 平均准确率
        train_loss, train_acc = train(sub_train_)
        valid_loss, valid_acc = valid(sub_valid_)

        # 模型保存
        torch.save(model.state_dict(), MODEL_PATH)
        print('The model saved epoch {}'.format(epoch))

        # 计算训练和验证的总耗时(秒)
        secs = int(time.time() - start_time)
        # 用分钟和秒表示
        mins = secs / 60
        secs = secs % 60

        # 打印训练和验证耗时, 平均损失, 平均准确率
        print('Epoch: %d' % (epoch + 1), " | time in %d minites, %d seconds" % (mins, secs))
        print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
        print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')
else:
    # 如果未来要重新加载模型, 实例化model后执行以下命令
    model.load_state_dict(torch.load(MODEL_PATH))

# 打印从模型的状态字典中获得的Embedding矩阵
print('********************')
print(model.state_dict()['embedding.weight'])

