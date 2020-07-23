import collections
import pickle
from tqdm import tqdm
import numpy as np


# session = [2,5,6,4,1,2,1]
# s = collections.Counter(session)
# print(dict(s))
#
# data = pickle.load(open("datasets/yoochoose1_64/all_train_seq.txt","rb"))
# train = pickle.load(open("datasets/yoochoose1_64/train.txt","rb"))
# print(len(data))
# print(len(train[0]))
#
for epoch in tqdm(range(20)):
    pass

u_A = [1, 2, 5, 2, 4, 0, 0, 0]
u = np.unique(u_A)
print(u) # [0 1 2 3 4]
node = 4
print(np.where(u==node)[0][0])  # 3 返回下标


