#构建数据集
import torch
import pickle
import config
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
from word2seq import Word2Sequence

word_sequence = pickle.load(open(config.word_sequence_dict,"rb")) #词典

class ChatDataset(Dataset):
    def __init__(self):
        super(ChatDataset,self).__init__()

        #读取内容
        data_path = config.data_path_txt
        self.data_lines = open(data_path,encoding='utf-8').readlines()

    #获取对应索引的问答
    def __getitem__(self, index):
        input = self.data_lines[index].strip().split()[:-1]
        target = self.data_lines[index].strip().split()[1:]
        #为空则默认读取下一条
        if len(input) == 0 or len(target)==0:
            input = self.data_lines[index+1].split()[:-1]
            target = self.data_lines[index+1].split()[1:]
        #此处句子的长度如果大于max_len，那么应该返回max_len
        return input,target,len(input),len(target)

    #获取数据长度
    def __len__(self):
        return len(self.data_lines)
    
# 整理数据————数据集处理方法
def collate_fn(batch):

    # 排序
    batch = sorted(batch,key=lambda x:x[2],reverse=True) #输入长度排序
    input, target, input_length, target_length = zip(*batch)

    max_len = max(input_length[0],target_length[0]) #这里只需要固定每个batch里面的样本长度一致就好，并不需要整个数据集的所有样本长度一致

    # 词变成词向量，并进行padding的操作
    input = torch.LongTensor([word_sequence.transform(i, max_len=max_len, add_eos=False) for i in input])
    target = torch.LongTensor([word_sequence.transform(i, max_len=max_len, add_eos=False) for i in target])

    input_length = torch.LongTensor(input_length)
    target_length = torch.LongTensor(target_length)

    return input, target

print("数据集装载...")
data_loader = DataLoader(dataset=ChatDataset(),batch_size=config.batch_size,shuffle=True,collate_fn=collate_fn,drop_last=True)

'''''
if __name__ == '__main__':
    for idx, (input, target) in enumerate(data_loader):
        print(idx)
        print(input)
        print(target)
'''''