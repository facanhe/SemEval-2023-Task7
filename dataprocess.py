import csv
import os
import re
import sys
import logging
import datasets
import json

import numpy
import pandas as pd
import numpy as np
import jsonpath
import glob
def strprocess(str):
    pass

with open("./Training_data/train.json")as f:
    jsonp=json.load(f)
labellist=jsonpath.jsonpath(jsonp,'$..Label')
labellist1=[]
for i in range(len(labellist)):
    if labellist[i]=='Entailment':
        labellist1.append(1)
    else:
        labellist1.append(0)
statementlist=jsonpath.jsonpath(jsonp,'$..Statement')


primaryindexlist=jsonpath.jsonpath(jsonp,'$..Primary_id')
# print(type(labellist))
# print(labellist)
# print(primaryindexlist)
# print(statementlist)
interventionlist=[]
eligibilitylist=[]
resultlist=[]
adverselist=[]
#print(primaryindexlist[0])
for i in range(len(primaryindexlist)):
    #print(secondindexlist[i]+'.json')
    jp=os.path.join("./Training_data/Clinical trial json/",primaryindexlist[i]+'.json')
    # print(jp)
    with open(jp)as f:
        js1=json.load(f)
    intervention=jsonpath.jsonpath(js1,'$..Intervention')
    intervention=','.join(str(i) for i in intervention)
    intervention=re.sub("[^a-zA-Z0-9.+()^*/~!@%&_=:]", " ",intervention)
    intervention=re.sub('\s+'," ",intervention)
    eligibility = jsonpath.jsonpath(js1, '$..Eligibility')
    eligibility = ','.join(str(i) for i in eligibility)
    eligibility = re.sub("[^a-zA-Z0-9.+()^*/~!@%&_=:]", " ", eligibility)
    eligibility = re.sub('\s+', " ", eligibility)
    result = jsonpath.jsonpath(js1, '$..Results')
    result = ','.join(str(i) for i in result)
    result = re.sub("[^a-zA-Z0-9.+()^*/~!@%&_=:]", " ", result)
    result = re.sub('\s+', " ", result)
    adverseevent = jsonpath.jsonpath(js1, '$..Adverse Events')
    adverseevent = ','.join(str(i) for i in adverseevent)
    adverseevent = re.sub("[^a-zA-Z0-9.+()^*/~!@%&_=:]", " ", adverseevent)
    adverseevent = re.sub('\s+', " ", adverseevent)
    interventionlist.append(intervention)
    eligibilitylist.append(eligibility)
    resultlist.append(result)
    adverselist.append(adverseevent)
list = [primaryindexlist, statementlist, labellist1, interventionlist, eligibilitylist, resultlist, adverselist]
list_change=numpy.transpose(list)
head=['Primary_id','Statement','Label','Intervention','Eligibility','Results','Adverse Events']
test=pd.DataFrame(columns=head,data=list_change)
# print(len(primaryindexlist),len(labellist),len(statementlist),len(interventionlist))
# head=['Primary_id','Statement','Label','Intervention','Eligibility','Results','Adverse Events']


# test=pd.DataFrame(columns=head,data=list_change)
# test.to_csv('./Training_data/traindata.csv',index=False,encoding='utf-8')
# test=pd.DataFrame({'Primary_id':primaryindexlist,'Statement':statementlist,'Label':labellist,'Intervention':interventionlist,'Eligibility':eligibilitylist,'Results':resultlist,'Adverse Events':adverselist})
test.to_csv('./Training_data/traindata.csv',index=None,encoding='utf-8')