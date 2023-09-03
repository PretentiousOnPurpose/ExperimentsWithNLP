import os
import numpy as np

def importDS(filename, forVocab=False):
    with open("./datasets/" + filename) as f:
        data = f.readlines()

        if forVocab == True:
            data = [line.replace(".", " . ").replace("&quot;", " \" ").replace(",", " , ").replace(" ##AT##-##AT## ", " ").replace(" &apos;", " ' ").strip() for line in data]
        else:
            data = [line.replace("&quot;", "\"").replace(" ##AT##-##AT## ", " ").replace(" &apos;", "'").strip() for line in data]
    
    return data

def buildVocab(dataList):
    data = []

    for ds in dataList:
        data = data + ds

    vocab = {}
    
    splitWords = []

    for line in data:
        splitWords.append(line.split(" "))
    
    splitWords = set(splitWords)
    for idx, word in enumerate(splitWords):
        vocab[word] = idx

    revVocab = dict([(value, key) for key, value in vocab.items()])

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

if __name__ == "__main__":
    print(importDS("ENG-DEU/test2015.en")[0])