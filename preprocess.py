import pandas as pd
import numpy as np


#Disclaimer:
#Credit and inspiration for this function from http://pi19404.github.io/pyVision/2018/01/31/lstm4/
def processIOB(filename):
    out=[]
    labels=[]
    with open(filename, 'r') as f:
        for row in f:
            atis={}
            words = row.split()

            intent=words[len(words)-1]
            words=words[0:len(words)-1]

            text="";
            tagging="";

            flag=-1

            for w in words:
                if w=="EOS":
                    flag=1;
                if flag==0:
                    text=text+" "+w;
                if flag==1:
                    tagging=tagging+" "+w
                if w==words[0]:
                    flag=0

            atis['text']=text
            atis['tags']=tagging
            atis['intent']=intent


            labels.append(intent)
            out.append(atis)

    labels=np.unique(labels)


    return out,labels


def preprocess(filename):

    out,labels=processIOB(filename)

    return out,labels
