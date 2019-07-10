import pandas as pd
import numpy as np
import os

#Source: http://pi19404.github.io/pyVision/2018/01/31/lstm4/
#Yes, this code snippet was taken from an online source and repurposed for this exercise.
#My intent to go ahead and use it is to be honest of how realistic development scenarios take place.
#In true developer fashion, if we find a code snippet that achieves our goal that does not require a license, why not use it and save time?
#For further demonstration, I have gone ahead and added comments to show my understanding of what the code does.

#Create function to process the raw IOB data files, pass in a filename
def processIOBData(filename):
    #Create empty lists
    output=[]
    labels=[]
    #Open and read the file
    with open(filename, 'r') as f:
        #Read each row in the file
        for rowx in f:
            #Create an empty dictionary
            r={}
            #In the row, split the string into a list of words
            words = rowx.split()
            #Save the last word in the list to a variable called intent_name
            intent_name=words[len( words)-1]
            #Save the rest of words in the list, except the last word to a variable called words
            words=words[0:len( words)-1]
            #Set aa and cc to empty strings
            aa="";
            cc="";
            #Set state variable to -1 to track the value in the next for loop
            state=-1
            #Create a for loop to iterate through each word in the previously saved word list
            for w in words:
                #If the word in the list is EOS (end of sentence) set the state flag to positive 1.
                if w=="EOS":
                    state=1;


                #Check the value of the state flag
                #If the state flag is zero, we can see below we have not encountered EOS yet and we are still at 
                #the beginnning of the row, since we saw that every row starts with the BOS airport code.
                #Then we add the word to the empty string with a space between each word. This continues until we loop
                #through and find EOS where the state gets set to 1.
                if state==0:
                    aa=aa+" "+w;
                #When the state flag is 1, we start to create a new string that creates a string with spaces for the characters in the tagging notation.
                if state==1:
                    cc=cc+" "+w
                #As menioned above, find BOS in the list indicates we are the beginning of the row and a flag is raised to indicate this
                if w=="BOS":
                    state=0

            #Add the newly contructed string, aa, as the value created in the previous loop to the dictionary with text as the key
            r['text']=aa
            #Add the newly contructed string, cc, as the value created in the previous loop to the dictionary with tags as the key
            r['tags'] =cc
            #Add the intent_name saved earlier to the dictionary with tags as the key
            r['intent']=intent_name

            #Continulously add the intent_name to the empty list previously created
            labels.append(intent_name)
            #Continulously add the dictionaries to the empty list previously created
            output.append(r)
    #Use the unique method from numpy on the list of labels to get a list of unique values
    labels=np.unique(labels)

    #Function returns the data as a list of dictionaries and also returns the labels as a numpy array
    return output,labels


def preprocess(filename):

    output,labels=processIOBData(filename)

    print("Number of Samples",len(output))
    print("Number of Intent Labels",len(labels))

    return output,labels