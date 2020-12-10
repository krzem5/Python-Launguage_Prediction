from nn.nn import NeuralNetwork
from unidecode import unidecode
import glob
import json
import os
import random
import time



LANG_LIST="pl,en".split(",")
ALPHABET="abcdefghijklmnopqrstuvwxyz"
MAX_LETTERS=12
WORDS={}
MODE=1



def encode_word(w):
	e=[0]*len(ALPHABET)*MAX_LETTERS
	i=0
	for c in w[:MAX_LETTERS]:
		e[i*len(ALPHABET)+ALPHABET.index(c)]=1
		i+=1
	return e
def encode_lang(l):
	e=[0]*len(LANG_LIST)
	e[LANG_LIST.index(l)]=1
	return e



for l in LANG_LIST:
	with open(f"./data/{l}.txt","r") as f:
		WORDS[l]=f.read().lower().split("\n")



def batch(N):
	B=[]
	for l in LANG_LIST:
		wl=WORDS[l][:]
		random.shuffle(wl)
		for w in wl[:N]:
			B+=[[encode_word(w),encode_lang(l)]]
	random.shuffle(B)
	return B



def train(NN,t,BS,log=True,S=0):
	l=-1
	st=time.time()
	for i in range(S*int(t/10000),t):
		d=batch(BS)
		if (log==True and int(i/t*10000)>l):
			l=int(i/t*10000)
			print(f"{l/100}% complete... ({int((time.time()-st)*100)/100}s) Acc={NN.test(batch(1000),log=False)}%")
			st=time.time()
			# if (l>=100):
			# 	if (os.path.isfile(f"./json/NN-data-{l-100}.json")):
			# 		os.remove(f"./json/NN-data-{l-100}.json")
			open(f"./json/NN-data-{l}.json","w").write(json.dumps(NN.toJSON(),indent=4,sort_keys=True))
		for k in d:
			NN.train(k[0],k[1])



def transform_s(S):
	S=unidecode(str(S)).lower()
	N=""
	for k in S:
		if (k==" " or k in ALPHABET):
			N+=k
	return N



def predict(W):
	o=NN.predict(encode_word(W))
	s=sum(o)
	i=0
	for k in o:
		o[i]=k/s
		i+=1
	i=0
	for k in LANG_LIST:
		print(f"{k.title()} \u2012 {int(o[i]*10000)/100}")
		i+=1



def predict_sentence(S):
	S=transform_s(S)
	a=[0]*len(LANG_LIST)
	for W in S.split(" "):
		o=NN.predict(encode_word(W))
		s=sum(o)
		i=0
		for k in o:
			a[i]+=k/s
			i+=1
	t=sum(a)
	for i in range(0,len(a)):
		a[i]/=t
	i=0
	for k in LANG_LIST:
		print(f"{k.title()} \u2012 {int(a[i]*10000)/100}")
		i+=1



def grammar_correction(SW):
	S=transform_s(SW)
	a=[0]*len(LANG_LIST)
	d={}
	j=0
	for W in S.split(" "):
		o=NN.predict(encode_word(W))
		s=sum(o)
		i=0
		d[j]=[]
		for k in o:
			a[i]+=k/s
			d[j]+=[k/s]
			i+=1
		j+=1
	t=sum(a)
	for i in range(0,len(a)):
		a[i]/=t
	L=a.index(max(a))
	ERR=[]
	for k in d.keys():
		w=d[k]
		if (w[L]<max(w)):
			ERR+=[[k,w.index(max(w))]]
	s=SW+"\n"
	i=0
	for k in ERR:
		off=len(" ".join(SW.split(" ")[:k[0]]))+1-(0 if i==0 else len(" ".join(SW.split(" ")[:ERR[i-1][0]+1])))
		l=len(SW.split(" ")[k[0]])
		s+=" "*off+"^"*l
		i+=1
	print(s)



def closest_word(W):
	print("CLOSEST",WORDS,W)



if (MODE==0):
	f=glob.glob("./json/*.json")[-1]
	if (os.path.isfile(f)):
		NN=NeuralNetwork(json.loads(open(f,"r").read()))
	else:
		NN=NeuralNetwork(len(ALPHABET)*MAX_LETTERS,[MAX_LETTERS+2],len(LANG_LIST),lr=0.0075)
	NN.lr=0.0055
	train(NN,100_000,10,S=int(f.rsplit("-",1)[1].split(".")[0]))
	open("./json/NN-data-FULL.json","w").write(json.dumps(NN.toJSON(),indent=4,sort_keys=True))
else:
	f=glob.glob("./json/*.json")[-1]
	NN=NeuralNetwork(json.loads(open(f,"r").read()))
	S="Hej lubię you!"
	grammar_correction(S)