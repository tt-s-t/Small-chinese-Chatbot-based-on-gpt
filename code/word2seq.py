#生成词表
#构造文本序列化和反序列化方法（文本转数字）
#import matplotlib.pyplot as plt
import pickle
import config
from tqdm import tqdm

class Word2Sequence():
    PAD_TAG = "<PAD>" #填充编码
    UNK_TAG = "<UNK>" #未知编码
    EOS_TAG = "<EOS>" #句子结尾

    #上面四种情况的对应编号
    PAD = 0
    UNK = 1
    EOS = 2

    def __init__(self):

        #文字——标号字典
        self.dict = {
            self.PAD_TAG :self.PAD,
            self.UNK_TAG :self.UNK,
            self.EOS_TAG :self.EOS
        }
        #词频统计
        self.count = {}
        self.fited = False #是否统计过词典了

    #以下两个转换都不包括'\t'
    #文字转标号（针对单个词）
    def to_index(self,word):
        """word -> index"""
        assert self.fited == True,"必须先进行fit操作"
        return self.dict.get(word,self.UNK) #无这个词则用未知代替

    #标号转文字（针对单个词）
    def to_word(self,index):
        """index -> word"""
        assert self.fited == True, "必须先进行fit操作"
        if index in self.inversed_dict:
            return self.inversed_dict[index]
        return self.UNK_TAG

    # 获取词典长度
    def __len__(self):
        return len(self.dict)

    #统计词频生成词典
    def fit(self, sentence):
        """
        :param sentence:[word1,word2,word3]
        """
        for a in sentence:
            if (a != '<EOS>'):
                if (a not in self.count):
                    self.count[a] = 0
                self.count[a] += 1

        self.fited = True

    def build_vocab(self, min_count=config.min_count, max_count=None, max_feature=None):

        # 限定统计词频范围
        if min_count is not None:
            self.count = {k: v for k, v in self.count.items() if v >= min_count}
        if max_count is not None:
            self.count = {k: v for k, v in self.count.items() if v <= max_count}

        # 给对应词进行编号
        if isinstance(max_feature, int): #是否限制词典的词数
            #词频从大到小排序
            count = sorted(list(self.count.items()), key=lambda x: x[1])
            if max_feature is not None and len(count) > max_feature:
                count = count[-int(max_feature):]
            for w, _ in count:
                self.dict[w] = len(self.dict)
        else: #按字典序(方便debug查看)
            for w in sorted(self.count.keys()):
                self.dict[w] = len(self.dict)

        # 准备一个index->word的字典
        self.inversed_dict = dict(zip(self.dict.values(), self.dict.keys()))

        #debug专用
        f_debug_word = open("data/debug_word.txt","w",encoding='utf-8')
        t = 0
        for key,_ in self.dict.items():
            t = t + 1
            if t > 3:
                f_debug_word.write(key+"★ "+str(self.count[key]) + "\n") #使用★ 区分是为了防止其中的词语包含分隔符，对我们后续的操作不利

        f_debug_word.close()

    def transform(self, sentence,max_len=None,add_eos=True):
        """
        实现把句子转化为向量
        :param sentence:
        :param max_len:
        :return:
        """
        assert self.fited == True, "必须先进行fit操作"

        r = [self.to_index(i) for i in sentence]
        if max_len is not None: #限定长度
            if max_len>len(sentence):
                if add_eos:
                    #添加结束符与填充符达到一定长度
                    r+=[self.EOS]+[self.PAD for _ in range(max_len-len(sentence)-2)]
                else: #添加填充符达到一定长度
                    r += [self.PAD for _ in range(max_len - len(sentence)-1)]
            else:
                if add_eos:
                    r = r[:max_len-2]
                    r += [self.EOS]
                else:
                    r = r[:max_len-1]
        else:
            if add_eos:
                r += [self.EOS]

        return r

    def inverse_transform(self,indices):
        """
        实现从句子向量 转化为 词（文字）
        :param indices: [1,2,3....]
        :return:[word1,word2.....]
        """
        sentence = []
        for i in indices:
            word = self.to_word(i)
            sentence.append(word)
        return sentence


'''''
#初始
word_sequence = Word2Sequence()
#词语导入
data_path = config.data_path_txt
for line in tqdm(open(data_path,encoding='utf-8').readlines()):
    word_sequence.fit(line.strip().split())


print("生成词典...")
word_sequence.build_vocab(min_count=None,max_count=None,max_feature=None)#这里不限制词典词语数目
print("词典大小：",len(word_sequence.dict))
pickle.dump(word_sequence,open(config.word_sequence_dict,"wb"))
'''''