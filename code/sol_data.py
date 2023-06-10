#首先进行数据预处理（其中没有数字、英文和颜文字，这份数据很nice），并且将其调整为合适的格式保存起来

import re
from tqdm import tqdm
import zhconv
import config

#处理重复符号的表达，如替换多个重复符号
def delete_repeat(s):
    #注释掉的是英文的表达
    #s = re.sub('[!]+','!', s)
    #s = re.sub('[?]+','?', s)
    #s = re.sub('[,]+',',', s)
    #s = re.sub('[:]+',':', s)
    #s = re.sub('[;]+',';', s)
    s = re.sub('[，]+','，', s)
    s = re.sub('[！]+','！', s)
    s = re.sub('[？]+','？', s)
    s = re.sub('[：]+','：', s)
    s = re.sub('[；]+','；', s)
    s = re.sub('[。]+','。', s)
    s = re.sub('[、]+','、', s)
    return s

with open('data/origin_train.txt','r',encoding='utf-8') as f: #打开原始数据集
    lines = f.readlines()

train_datas = []
temp_data = ''
#max_len = 0
#每个多轮对话中使用'<EOS>'将其划分
for line in tqdm(lines):

    if line!='\n':
        line = line.strip()#去除前导后方空格
        #英文标点符号置换为中文标点符号
        line = line.replace('!','！')
        line = line.replace('?','？')
        line = line.replace(',','，')
        line = line.replace('.','。')
        line = line.replace(':','：')
        line = line.replace(';','；')
        line = zhconv.convert(line, 'zh-cn') #转为简体字
        line = " ".join(line)
        temp_data+=(line+' <EOS> ')
    else:
        if len(temp_data.split()) <= config.max_len: #限制长度
            train_datas.append(temp_data)
        temp_data=''

#print(max_len)

with open(config.data_path_txt,'w',encoding='utf-8') as f: #将处理后的数据保存在另一个文件中
    for train_data in train_datas:
        f.write(train_data+'\n')
