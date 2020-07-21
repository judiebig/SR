import collections
import pickle
from tqdm import tqdm

session = [2,5,6,4,1,2,1]
s = collections.Counter(session)
print(dict(s))





data = pickle.load(open("datasets/yoochoose1_64/all_train_seq.txt","rb"))
train = pickle.load(open("datasets/yoochoose1_64/train.txt","rb"))
print(len(data))
print(len(train[0]))




for epoch in tqdm(range(20)):
    pass