# nlp技术点汇总
## 词向量
### one-hot
### tf-idf
### word2vector
cbow&skip-gram计算流程
![图片](images/word2vector.png)
### fasttext
属于文本分类任务，包含两类gram:  
- word-n-gram：词序关系  
- char-n-grams：对低频和oov词友好
---
## BERT类语言模型
### elmo
每一个词会给出3个embedding，这3个embedding可以看作是一个词的3个embedding特征，对3个embedding特征会添加三个位置，对于不同的任务会赋予不同的权重，最后根据权重把这三个embedding结合起来做向量平均，把合并后的embedding作为最后词的embedding  
- 优点  
1.使得word embeding根据输入句子不同而动态变化，解决一词多义问题；
2.采用双向LSTM语言模型来捕获句子更长的依赖关系，提高了模型最终效果；
3.简单易上手，ELMo官方allenNLP发布了相关基于PyTorch实现的版本；
- 缺点  
1.双向LSTM对语言模型建模不如注意力模型，训练速度较慢；
2.双向语言模型是采用拼接的方式得到的，特征选择和融合较弱；
3.当数据数量较大和质量较高时，该模型的良好效果不显著;


### Transformer
### bert
### ernie
### ALBert
### RoBert
### TinyBert
### xlnet
### T5
### ELECTRA
---
## GPT类语言模型
### gpt
### chatgpt
---
## 经典NN模型
### TextCNN
### RNN/LSTM
### seq2seq
### attention
### ABCNN
---
## 多模融合模型
### 多域DNN
### 多流交互模型
#### LXMERT
#### VILBERT
### 单流交互模型
#### Unicoder-VL
#### VIsualBERT
### 多模态GPT
#### DaLL E2
#### Stable Diffusion
---
## 多任务模型
### mttbert
### prompt
### CoT,Chain of Thought
---
## 参考资料
