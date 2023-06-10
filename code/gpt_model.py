import torch
from torch import nn
import math
import torch.nn.functional as F

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

#生成掩码矩阵mask
def create_masks(input):
    #input的shape为：(batch_size, max_len)
    def subsequent_mask(size):
        mask = torch.tril(torch.ones(size, size).type(dtype=torch.uint8)) #torch.triu()返回一个下三角矩阵（让其只注意到前面的信息，后面注意不到，因为为0）
        return mask.unsqueeze(0) # shape: (1,size,size)
     
    input_mask = input!=0 #屏蔽掉填充词（填充符的标号为0）
    input_mask = input_mask.unsqueeze(1) # (batch_size, 1, max_len)
    input_mask = input_mask & subsequent_mask(input.size(-1)).type_as(input_mask.data) #统一成int形式再进行与操作，shape：(batch_size, max_len, max_len)
    
    return input_mask
    #(batch_size, 1, max_len, max_len)

#emb
class Embeddings(nn.Module):
    """
    实现词嵌入并添加它们的位置编码
    """
    def __init__(self, vocab_size, emb_dim, max_pos):
        super(Embeddings, self).__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim) #文字标号转emb词向量
        self.pos_emb = nn.Embedding(max_pos, emb_dim) #位置编码

    def forward(self, encoded_words):
        #输入shape：[batch_size, max_len]
        max_len = encoded_words.size(1)
        pos = torch.arange(max_len, dtype=torch.long,device=device)
        pos = pos.unsqueeze(0).expand_as(encoded_words)  # [max_len] -> [batch_size, max_len] —— 0为填充符
        embedding = self.embed(encoded_words) + self.pos_emb(pos)
        return embedding #[batch_size, max_len, emb_dim]

#多头注意力机制
class MultiHeadAttention(nn.Module):
    
    def __init__(self, heads, emb_dim, d_k, d_v):
        
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.heads = heads
        self.dropout = nn.Dropout(0.1)
        self.query = nn.Linear(emb_dim, d_k*heads)
        self.key = nn.Linear(emb_dim, d_k*heads)
        self.value = nn.Linear(emb_dim, d_v*heads)
        self.concat = nn.Linear(d_v*heads, emb_dim)
        
    def forward(self, query, key, value, mask):
        """
        query, key, value of shape: (batch_size, max_len, emb_dim)
        mask of shape: (batch_size, max_len, max_len)
        """
        
        query = self.query(query) #(batch_size, max_len, d_k*heads)
        key = self.key(key) #(batch_size, max_len, d_k*heads)     
        value = self.value(value) #(batch_size, max_len, d_v*heads)  
        
        query = query.view(query.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3) #(batch_size, h, max_len, d_k)
        key = key.view(key.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3) #(batch_size, h, max_len, d_k)
        value = value.view(value.shape[0], -1, self.heads, self.d_v).permute(0, 2, 1, 3) #(batch_size, h, max_len, d_v)
        
        # (batch_size, h, max_len, d_k) matmul (batch_size, h, d_k, max_len) --> (batch_size, h, max_len, max_len)
        scores = torch.matmul(query, key.permute(0,1,3,2)) / math.sqrt(self.d_k)

        mask = mask.unsqueeze(1).repeat(1, self.heads, 1,
                                                  1)  # attn_mask : [batch_size, n_heads, max_len, max_len]
   
        scores = scores.masked_fill(mask == 0, -1e9)    # (batch_size, h, max_len, max_len) masked_fill()函数主要用在transformer的attention机制中，在时序任务中，主要是用来mask掉当前时刻后面时刻的序列信息(即不为0的用-1e9替换)
       
        weights = F.softmax(scores, dim = -1) # (batch_size, h, max_len, max_len) 那么前面被mask掉的（即-1e9的概率就会很小很小，几乎为0，实现了屏蔽的效果）
        #weights = self.dropout(weights)
        # (batch_size, h, max_len, max_len) matmul (batch_size, h, max_len, d_v) --> (batch_size, h, max_len, d_v)
        context = torch.matmul(weights, value)
        #(batch_size, h, max_len, d_v) --> (batch_size, max_len, h, d_v) --> (batch_size, max_len, h * d_v)
        context = context.permute(0,2,1,3).contiguous().view(context.shape[0], -1, self.heads * self.d_v) #torch.contiguous()方法首先拷贝了一份张量在内存中的地址,否则直接view是不会重新创建一个的
        #(batch_size, max_len, h * d_v)
        interacted = self.concat(context)
        return interacted #(batch_size, max_len, emb_dim)

# FeedForward的输入是Multi-Head Attention的输出做了残差连接和Norm之后的数据，然后FeedForward做了两次线性线性变换，为的是更加深入的提取特征。
class FeedForward(nn.Module):
    # torch.nn.Linear的输入和输出的维度可以是任意的，默认对最后一维做全连接
    def __init__(self, emb_dim, middle_dim = 2048):
        super(FeedForward, self).__init__()
        
        self.fc1 = nn.Linear(emb_dim, middle_dim)
        self.fc2 = nn.Linear(middle_dim, emb_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(self.dropout(out))
        return out

class DecoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v):
        super(DecoderLayer, self).__init__()
        self.layernorm = nn.LayerNorm(d_model)
        self.self_multihead = MultiHeadAttention(n_heads, d_model, d_k, d_v)
        self.feed_forward = FeedForward(d_model)

    def forward(self, dec_inputs, dec_self_attn_mask):
        '''
        dec_inputs: [batch_size, max_len, emb_dim]
        dec_self_attn_mask: [batch_size, max_len, max_len]
        '''
        # dec_outputs: [batch_size, max_len, emb_dim]
        dec_outputs = self.self_multihead(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs = self.layernorm(dec_outputs + dec_inputs)
        output = self.layernorm(self.feed_forward(dec_outputs) + dec_outputs)

        return output


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, max_pos, n_heads, d_k, d_v, n_layers):
        super(Decoder, self).__init__()
        self.embed = Embeddings(vocab_size, d_model, max_pos)
        self.layers = nn.ModuleList([DecoderLayer(n_heads, d_model, d_k, d_v) for _ in range(n_layers)])

    def forward(self, dec_inputs):
        '''
        dec_inputs: [batch_size, max_len]
        '''
        dec_outputs = self.embed(dec_inputs) # [batch_size, max_len, d_model]

        dec_self_attn_mask = create_masks(dec_inputs) # [batch_size, max_len, max_len] 生成mask矩阵

        for layer in self.layers:
            # dec_outputs: [batch_size, max_len, d_model]
            dec_outputs = layer(dec_outputs, dec_self_attn_mask)

        return dec_outputs

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, max_pos, n_heads, d_k, d_v, n_layers):
        super(GPT, self).__init__()
        self.decoder = Decoder(vocab_size, d_model, max_pos, n_heads, d_k, d_v, n_layers)
        self.projection = nn.Linear(d_model,vocab_size)
    def forward(self,dec_inputs):
        """
        dec_inputs: [batch_size, max_len]
        """

        # dec_outpus: [batch_size, max_len, d_model]
        dec_outputs = self.decoder(dec_inputs)
        # dec_logits: [batch_size, max_len, vocab_size]
        dec_logits = self.projection(dec_outputs)
        return dec_logits

