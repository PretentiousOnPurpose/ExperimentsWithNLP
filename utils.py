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

    vocab["<START>"] = len(vocab.keys())
    vocab["<END>"] = len(vocab.keys())

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

def genTrainingDataset(eng_ds, deu_ds, eng_vocab, deu_vocab, feature_len):
    ENC_INPUT_MAXLEN = np.max(np.array([len(line.split()) for line in eng_ds]))
    DEC_INPUT_MAXLEN = np.max(np.array([len(line.split()) for line in deu_ds])) + 1

    enc_input_data = np.zeros((len(eng_ds), ENC_INPUT_MAXLEN, feature_len))
    dec_input_data = np.zeros((len(deu_ds), DEC_INPUT_MAXLEN, feature_len))
    dec_target_data = np.zeros((len(deu_ds), DEC_INPUT_MAXLEN, feature_len))

    for idx, (input_data, target_data) in enumerate(zip(eng_ds, deu_ds)):
        enc_input_data[idx, :, :] = word2Vec(input_data, eng_vocab, output="int")

        dec_input_data[idx, 1:, :] = word2Vec(target_data, deu_vocab, output="int")
        dec_target_data[idx, :-1, :] = word2Vec(target_data, deu_vocab, output="int")

        dec_input_data[idx, 0, :] = eng_vocab["<START>"]
        dec_target_data[idx, -1, :] = eng_vocab["<END>"]

    return [enc_input_data, dec_input_data, dec_target_data]


# ONLY FOR TESTING------------

# if __name__ == "__main__":
#     eng_ds = importDS("ENG-DEU/test2015.en")
#     deu_ds = importDS("ENG-DEU/test2015.de")
    
#     eng_vocab, eng_revVocab = buildVocab([eng_ds])
#     deu_vocab, deu_revVocab = buildVocab([deu_ds])

#     [enc_input_data, dec_input_data, dec_target_data] = genTrainingDataset(eng_ds, deu_ds, eng_vocab, deu_vocab, 1)

    