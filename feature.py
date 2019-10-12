datalist=[]
w = open("./dataset/N_train_ready","w")
with open("./dataset/NewsTrain_after","r") as f:
    for l in f.readlines():
        tmp = l[:-2].split("|")
        assert len(tmp)==4
        datalist.append(tmp)
import json
import torch
import torch.nn.functional as F
from fairseq.models.roberta import RobertaModel
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
roberta.eval()  # disable dropout
roberta.cuda()  # use the GPU (optional)

examples=[]
count=0
previous=""
for data in datalist:
    count+=1
#    if count>=102 and count<=121:
#        data[0]=data[0][:158]+data[0][161:-3]
#        print(data[0])
    if data[0]!=previous:
        doc = roberta.extract_features_aligned_to_words(data[0])
        previous=data[0]
    for token in doc:
        if str(token)==data[1]:
#            print(token.vector[:5])
#            examples.append((token.vector,data[2]))
            w.write(str(token.vector.tolist())+"|"+str(data[2]))
            w.write("\n")
            
#            print(count)

            break
w.close()
#print(examples)

#oc = roberta.extract_features_aligned_to_words("In 1964 , to support children who were not eligible for adoption , Buck established the Pearl S. Buck Foundation ( now called Pearl S. Buck International ) to address poverty and discrimination faced by children in Asian countries .")
#for tok in doc:
#    print(str(tok),end="  ")
#    print(tok.vector[:5])
