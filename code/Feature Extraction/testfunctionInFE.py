from utils import *
import pandas as pd

# 给文件添加表头
# data = pd.read_csv('data/Benign/top-1m.csv',header=None)
# data.columns = ['rank', 'domain']
# data.to_csv('data/Benign/top-1m-addHeader.csv',index= None)
# print(data)
# data = pd.read_csv('../data/Benign/top-1m.csv')
data = pd.read_csv('data/Benign/top-1m-addHeader.csv')
domains = data['domain'].unique()

count = calculateLen(domains)
print(count)