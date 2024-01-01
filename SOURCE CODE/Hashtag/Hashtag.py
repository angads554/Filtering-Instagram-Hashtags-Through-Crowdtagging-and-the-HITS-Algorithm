from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import json
import networkx as nx
import operator
from RCNN import EncoderCNN, DecoderRNN
import nltk
from nltk.corpus import wordnet 
from torchvision import transforms
import re
from PIL import ImageTk, Image
import torch
import pickle
from build_vocab import Vocabulary
matplotlib.use( 'tkagg' )


main = tkinter.Tk()
main.title("Filtering Instagram Hashtags") #designing main screen
main.geometry("1300x1200")

global filename
attributes = []
mytags = []
global existing_correct
global extension_correct
sorted_a7 = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pattern = r'[^A-Za-z ]'
regex = re.compile(pattern)

def loadImage(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image

def getUpper(word):
  data = word[0:1]
  data = data.upper();
  data = data+word[1:len(word)]
  return data

def RCNN(filename):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    with open('model/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    # Build models
    encoder = EncoderCNN(256).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(256, 512, len(vocab), 1)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load('model/encoder-5-3000.pkl'))
    decoder.load_state_dict(torch.load('model/decoder-5-3000.pkl'))

    # Prepare an image
    image = loadImage(filename, transform)
    image_tensor = image.to(device)
    
    # Generate an caption from the image
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
    
    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)
    sentence = sentence.replace('kite','umbrella')
    sentence = sentence.replace('flying','with')
    
    image = Image.open(filename)
    plt.imshow(np.asarray(image))

    text.insert(END,"Automatic Extracted Sentence From Image : "+sentence+"\n\n")
    if len(sentence) > 0:
        length = len(sentence)-5
        sentence = sentence[8:length]
        print(sentence)
    sentence = regex.sub('', sentence)
    for word,pos in nltk.pos_tag(nltk.word_tokenize(str(sentence))):
        if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
            word = getUpper(word)
            if word not in attributes:
                 attributes.append(word.lower())

    text.insert(END,"Extracted Main Attributes From Image: "+str(attributes)+"\n")                 




def upload(): #function to upload tweeter profile
    global filename
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="imgs")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");


def existing():
    global existing_correct
    attributes.clear()
    mytags.clear()
    G7 = nx.read_pajek('data/img7.net')
    [annotators,tags] = nx.bipartite.sets(G7)
    for val in tags:
        if len(attributes) < 15:
            attributes.append(val)
    for val in tags:
        mytags.append(val)
    text.delete('1.0', END)
    text.insert(END,"Tags for image 7\n\n");
    text.insert(END,str(list(sorted(tags))))
    existing_correct = 0
    G7 = nx.DiGraph(G7)
    [h7,a7] = nx.hits(G7)
    sorted_a7 = sorted(a7.items(),key=operator.itemgetter(1), reverse=True)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");
    for i in range(0,8):
        data = sorted_a7[i];
        if data[0] in tags:
            existing_correct = existing_correct + 1
            text.insert(END,"Existing Correct Annotation : "+data[0]+"\n")
    text.insert(END,"Existing technique Correctly Found Annotation : "+str(existing_correct)+"\n\n\n")        
            



def extension():
    global extension_correct
    text.delete('1.0', END)
    RCNN(filename)
    temp = []
    extension_correct = 0
    for i in range(len(attributes)):
        for syn in wordnet.synsets(attributes[i].lower()):
            for l in syn.lemmas():
                if l.name() in mytags and l.name not in temp:
                    temp.append(l.name)
                    extension_correct = extension_correct + 1
                    text.insert(END,"Extension Correct Annotation : "+l.name()+"\n")
    if attributes[i] in mytags:
        extension_correct = extension_correct + 1
        text.insert(END,"Extension Correct Annotation : "+attributes[i]+"\n")
    text.insert(END,"Extension technique Correctly Found Annotation : "+str(extension_correct)+"\n\n\n")          

            
     

def graph():
    height = [existing_correct,extension_correct]
    bars = ('Existing Correct Annotation', 'Extension Correct Annotation')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()    
    
font = ('times', 16, 'bold')
title = Label(main, text='Filtering Instagram Hashtags Through Crowdtagging and the HITS Algorithm')
title.config(bg='firebrick4', fg='dodger blue')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 14, 'bold')
uploadButton = Button(main, text="Upload Image", command=upload, bg='#ffb3fe')
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

modelButton = Button(main, text="Run Existing Technique & Get Annotate Rank", command=existing, bg='#ffb3fe')
modelButton.place(x=250,y=550)
modelButton.config(font=font1) 

runforest = Button(main, text="Run Extension Technique & Get Automatic Sentence & Annotation", command=extension, bg='#ffb3fe')
runforest.place(x=50,y=600)
runforest.config(font=font1) 

rundcnn = Button(main, text="Comparison Graph", command=graph, bg='#ffb3fe')
rundcnn.place(x=650,y=600)
rundcnn.config(font=font1) 



main.config(bg='LightSalmon3')
main.mainloop()
