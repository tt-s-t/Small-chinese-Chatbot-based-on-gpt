import torch
import pickle
import config
from word2seq import Word2Sequence

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    word_map = pickle.load(open(config.word_sequence_dict,"rb")) #词典
    checkpoint = torch.load('model.pth.rar')
    model = checkpoint['gpt']

    model.eval()
    #初始输入是空，每次加上后面的对话信息
    sentence = []
    while True:
        temp_sentence = input("我:")
        if temp_sentence == "再见，我不想聊了":
            print("拜拜，结束聊天")
            break
        sentence += list(temp_sentence.strip()) + ['<EOS>']

        if len(sentence) > 100:
            #由于该模型输入最大长度为100，避免长度超出限制长度过长需要进行裁剪
            t_index = sentence.index('<EOS>')
            sentence = sentence[t_index + 1:]

        sentence_vec = word_map.transform(sentence, max_len=None, add_eos=False) #词转为标号
        dec_input = torch.LongTensor(sentence_vec).to(device).unsqueeze(0)

        terminal = False
        start_dec_len=len(dec_input[0])
        #一直预测下一个单词，直到预测到"<EOS>"结束，如果一直不到"<EOS>"，则根据长度退出循环，并在最后加上”<EOS>“字符
        while not terminal :
            if len(dec_input[0])-start_dec_len>100:
                next_symbol=word_map.dict['<EOS>']
                dec_input = torch.cat(
                    [dec_input.detach(), torch.tensor([[next_symbol]], dtype=dec_input.dtype, device=device)], -1)
                break
            
            projected = model(dec_input)
            prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
            next_word = prob.data[-1]
            next_symbol = next_word
            if next_symbol == word_map.dict['<EOS>']:
                terminal = True

            dec_input = torch.cat(
                [dec_input.detach(), torch.tensor([[next_symbol]], dtype=dec_input.dtype, device=device)], -1)
            
        out = dec_input.squeeze(0)
        out = word_map.inverse_transform(out.tolist())

        #统计"<EOS>"字符在结果中的索引
        eos_indexs =[]
        for i in range(len(out)):
            if out[i] =="<EOS>":
                eos_indexs.append(i)
        
        #取最后两个<EOS>中间的内容作为回答
        answer = out[eos_indexs[-2]+1:-1]
       
        answer = "".join(answer)

        print("TT:", answer)