# Small-chinese-Chatbot-based-on-gpt
使用GPT搭建一个GPT小模型实现中文聊天（仅供娱乐）<br />
现在是刚试训练完的第一版，可能后续还会有调整，整体代码思路以及网络结构讲解可以看我的博客https://blog.csdn.net/weixin_55073640/article/details/131135003?spm=1001.2014.3001.5501<br />
现在处于缓慢更新状态，我会抽空完善它，谢谢大家的支持，欢迎批评指正以及交流~<br />
## 使用方法
（注：origin_train.txt是原始数据集，dataset.txt是预处理后的数据文件，ws.pkl是生成的词表，你想直接用或者自己重新生成都ok）。<br />
在config.py里把参数设置好，然后先运行"sol_data.py"文件生成预处理后的数据文件。<br />
然后将word2vec.py文件中最后注释掉的部分取消注释，运行一遍得到词表，再将这部分内容进行注释。<br />
接着运行train.py文件进行模型训练，ok之后运行chat.py文件测试你的模型结果啦~
