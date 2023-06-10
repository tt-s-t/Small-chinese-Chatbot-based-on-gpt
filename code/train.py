import pickle
import config
import torch
import torch.utils.data
from gpt_model import *
from dataset import data_loader
from utils import AdamWarmup, LossWithLS, get_acc
import numpy as np
from torch.utils.tensorboard import SummaryWriter

summaryWriter = SummaryWriter("logs/log2")

# config
emb_dim = config.emb_dim
max_pos = config.max_pos
heads = config.heads
d_k = config.d_k
d_v = config.d_v
num_layers = config.num_layers
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
epochs = config.epochs

word_map = pickle.load(open(config.word_sequence_dict,"rb")) #词典
print(len(word_map.dict))

if config.load == False: #新的训练
    gpt = GPT(vocab_size=len(word_map.dict), d_model=emb_dim, max_pos=max_pos, n_heads= heads, d_k=d_k, d_v=d_v, n_layers=num_layers).to(device)
    adam_optimizer = torch.optim.Adam(gpt.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9) #Adam优化器
    #gpt_optimizer = AdamWarmup(model_size = emb_dim, warmup_steps = 4000, optimizer = adam_optimizer) #使用warmup策略
    #criterion = LossWithLS(len(word_map.dict), 0.1) #损失函数
    epoch_start = 0
else: #加载之前的模型接着训练
    checkpoint = torch.load('model.pth.rar')
    gpt = checkpoint['gpt']
    adam_optimizer = checkpoint['adam_optimizer']
    epoch_start = checkpoint['epoch'] + 1


criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)

def train(train_loader, gpt, criterion, epoch):
    
    gpt.train()
    sum_loss = 0
    count = 0
    sum_acc = 0

    for i, (question, reply) in enumerate(train_loader):

        torch.cuda.empty_cache() #释放缓存空间

        samples = question.shape[0]

        # Move to device
        question = question.to(device)
        reply = reply.to(device)

        # question的shape（batchsize,max_len），reply是（batchsize,max_len-1）

        # Get the gpt outputs
        out = gpt(question)

        # Compute the loss
        loss = criterion(out.view(-1, out.size(-1)), reply.view(-1))
        acc = get_acc(out, reply)
        
        # Backprop
        #gpt_optimizer.optimizer.zero_grad() 
        adam_optimizer.zero_grad() #存留梯度清零
        loss.backward() #反向传播计算梯度
        #gpt_optimizer.step()
        adam_optimizer.step() #根据梯度进行参数更新

        sum_loss += float(loss.item()) * samples
        sum_acc += acc.item() * samples
        count += samples
        
        if i % 100 == 0:
            print("Epoch [{}][{}/{}]\tLoss: {:.3f}\tAcc: {:.3f}".format(epoch, i, len(train_loader), sum_loss/count,sum_acc/count)) #输出累计情况下平均一个词的loss

    return sum_loss/count
            
print("train...")    
loss_max = 10000000000
for epoch in range(epoch_start, epochs):
    
    loss = train(data_loader, gpt, criterion, epoch)
    
    #tensorboard实时监控
    summaryWriter.add_scalars('epoch_metric', {'epoch_loss': loss }, epoch)

    if loss_max > loss: #选择性保存
        print("保存轮数：",epoch)
        loss_max = loss
    
        #state = {'epoch': epoch, 'gpt': gpt, 'gpt_optimizer': gpt_optimizer}
        state = {'epoch': epoch, 'gpt': gpt, 'adam_optimizer': adam_optimizer}

        torch.save(state, 'model.pth.rar') #记下每次最好的结果（为了防止中断程序后，啥都没保存）
    
    if epoch == epochs-1: #保存最后的结果
        torch.save(state, 'model_last.pth.rar')