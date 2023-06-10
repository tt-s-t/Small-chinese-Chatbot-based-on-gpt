import torch
import torch.nn as nn
import torch.utils.data

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class AdamWarmup:
    
    def __init__(self, model_size, warmup_steps, optimizer):
        
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer
        self.current_step = 0
        self.lr = 0
        
    def get_lr(self):
        return self.model_size ** (-0.5) * min(self.current_step ** (-0.5), self.current_step * self.warmup_steps ** (-1.5))
    
    # 获取学习率并更新参数
    def step(self):
        self.current_step += 1 #步数上升
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.lr = lr
        self.optimizer.step()   

class LossWithLS(nn.Module):

    def __init__(self, size, smooth): #smooth即为某个极小值
        super(LossWithLS, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=0).to(device) #ignore_index相当于mask掉了
        self.confidence = 1.0 - smooth
        self.smooth = smooth
        self.size = size
        
    def forward(self, prediction, target):
        """
        prediction: (batch_size, max_words, vocab_size)
        target and mask: (batch_size, max_words) mask屏蔽掉那些填充符
        """
        prediction = prediction.view(-1, prediction.size(-1)) #(batch_size * max_words, vocab_size)
        target = target.contiguous().view(-1) # (batch_size * max_words)
        labels = prediction.data.clone()
        labels.fill_(self.smooth / (self.size - 1)) #先全部填充为“0”的情况（ε/k-1)
        labels.scatter_(1, target.data.unsqueeze(1), self.confidence) #将src中数据根据index中的索引按照dim的方向填进input。输入参数：(dim,input,src)
        loss = self.criterion(prediction, labels) # (batch_size * max_words, vocab_size)
        return loss
    
def get_acc(out,target): #训练准确率计算
    pred = torch.argmax(out,dim = 2)
    mask = target!=0 #(batch_size, max_words)————屏蔽掉填充词
    result = (pred==target)
    return (result*mask).sum()/mask.sum()
