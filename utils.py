import os
import numpy as np

def importDS(filename):
    with open("./datasets/" + filename) as f:
        data = f.readlines()
        data = [line.replace(".", " . ").replace("&#", "").replace("&quot;", " \" ").replace(",", " , ").replace(" ##AT##-##AT## ", " ").replace(" &apos;", " ' ").strip() for line in data]
    
    return data

def buildVocab(dataList):
    data = []

    for ds in dataList:
        data = data + ds

    vocab = {}
    
    splitWords = []

    for line in data:
        splitWords = splitWords + line.split(" ")
    
    splitWords = set(splitWords)
    for idx, word in enumerate(splitWords):
        vocab[word] = idx

    revVocab = dict([(value, key) for key, value in vocab.items()])
    print("Number of entries in vocab: " + str(len(vocab.keys())))

    return vocab, revVocab

def word2Vec(line, vocab, output="int"):
    if output == "onehot":
        pass
    elif output == "int":
        line_vec = []
        line = line.split(" ")
        for word in line:
            line_vec.append(vocab[word])

        return np.array(line_vec)

def vec2Word(line, revVocab, input="int"):
    if input == "onehot":
        pass
    elif input == "int":
        words = []
        for num in line:
            words.append(revVocab[num])

        return convertToSentence(words)

def convertToSentence(words):
    line = ""

    for word in words:
        if (line != "" and word == "\'") or line == "" or line == ".":
            line = line + word
        else:
            line = line + " " + word

    return line

if __name__ == "__main__":

    eng_ds = importDS("ENG-DEU/test2015.en")
    deu_ds = importDS("ENG-DEU/test2015.de")
    
    eng_vocab, eng_revVocab = buildVocab([eng_ds])
    deu_vocab, deu_revVocab = buildVocab([deu_ds])
    print(word2Vec(deu_ds[1222], deu_vocab,output="int"))
    print(vec2Word(word2Vec(deu_ds[1222], deu_vocab,output="int"), deu_revVocab, "int"))
    