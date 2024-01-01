import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np 
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
attributes = []
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
    attributes.clear()
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

    print('Extracted Sentence From Image : '+sentence)
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

    print("Extracted Attributes : "+str(attributes))                 


with open('data/F8_data.json') as fp:
        data = json.load(fp)
users = list(data.keys())
#print(data[users[0]].keys())

no_split = ['eyes', 'glasses', 'instagram', 'airplane', 'aircraft', 'cheers', 'icecream', 'frenchfries','nuts', 'potatoes','octopus','profiterole','shoes','sneakers','sports']

#exec(open('csv2imageGraphs.py').read())
#G = FullGraph(data,50,no_split,'tt.txt')
#print(G)


#r,p = computeROC('img','data/gold.json',50,0.25,1)
#print(str(r)+" "+str(p))

#[h,a] = nx.hits(G)

G7 = nx.read_pajek('data/img7.net')
[annotators,tags] = nx.bipartite.sets(G7)
print(list(sorted(tags)))

count = 0
##print(list(sorted(annotators)))
#G7 = nx.DiGraph(G7)
#[h7,a7] = nx.hits(G7)
#sorted_a7 = sorted(a7.items(),key=operator.itemgetter(1), reverse=True)
#for i in range(0,4):
#    data = sorted_a7[i];
#    if data[0] in tags:
#        count = count + 1
#    print((data[0])+" "+str(data))
#print(count)    

RCNN('Untitled.png')
for i in range(len(attributes)):
    for syn in wordnet.synsets(attributes[i].lower()):
        for l in syn.lemmas():
            print(attributes[i].lower()+"====="+l.name())
            if l.name() in tags:
                print(l.name())
                count = count + 1
    if attributes[i] in tags:
        print(attributes)
        count = count + 1
print(count)        


                
