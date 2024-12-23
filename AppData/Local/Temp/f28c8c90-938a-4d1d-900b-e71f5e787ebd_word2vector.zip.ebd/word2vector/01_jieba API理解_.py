import jieba


def t0():
    jieba.load_userdict("C:/Users/17320/Desktop/word2vector/jieba.dict")
    jieba.suggest_freq(('中', '将'), tune=True)  # 自动调整词频，减少将中将分开情况
    word_list = jieba.cut('外卖送餐公司中饿了么是你值得信赖的选择，如果放到post中将出错', HMM=True)
    print(list(word_list))

    import jieba.posseg as posseg

    words = posseg.cut("外卖送餐公司中饿了么是你值得信赖的选择，如果放到post中将出错")  # 词性标注 words转为生成器
    print(type(words))
    print(list(words))


if __name__ == '__main__':
    t0()
