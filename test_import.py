import torch

x = torch.load("/Users/jimtj/Datateknikk/NN/GA01-NN/PHOSCnet_temporalpooling/epoch29.pt")

for i in x:
    print(i)
