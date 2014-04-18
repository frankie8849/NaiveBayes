# coding=gbk
import re
import numpy as np
from numpy import *

#################################文本处理############################################ 

def testTextParse(filename):
    text = open(filename).read()
    pattern = '<text>(.*?)</text>'
    str_list = re.findall(pattern, text, re.S)
    doc_list = []
    ptn = re.compile('\\s*')
    for doc in str_list:
        doc = ptn.split(doc)
        doc_list.append([term for term in doc if len(term)>=1 and term != ','and term != '.'and term != '!'and term != '?'and term != '('and term != ')'
                         and term != '\"'and term != '\''
                         and term != '\xa1\xa3' and term != '\xa3\xac' and term != '\xa3\xbf'and term != '\xa3\xa1'and term != '\xa3\xbb'
                         and term != '\xa3\xba'and term != '\xa1\xb0'and term != '\xa1\xb1'and term != '\xa1\xae'and term != '\xa1\xaf'
                         and term != '\xa3\xa8'and term != '\xa3\xa9'and term != '\xa1\xa2'
                         ]) 
    return doc_list

def cvTextParse(filename,start,end):    #用于交叉验证的文档解析
    text = open(filename).read()
    pattern = '<text>(.*?)</text>'
    str_list = re.findall(pattern, text, re.S)
    doc_list = []
    start_index = 0
    end_index = start
    ptn = re.compile('\\s*')
    for doc in str_list:
        start_index +=1
        
        if start_index >= start:
            end_index +=1
            if end_index <= end:
                doc = ptn.split(doc)
                doc_list.append([term for term in doc if len(term)>=1 and term != ','and term != '.'and term != '!'and term != '?'and term != '('and term != ')'
                                 and term != '\"'and term != '\''
                                 and term != '\xa1\xa3' and term != '\xa3\xac' and term != '\xa3\xbf'and term != '\xa3\xa1'and term != '\xa3\xbb'
                                 and term != '\xa3\xba'and term != '\xa1\xb0'and term != '\xa1\xb1'and term != '\xa1\xae'and term != '\xa1\xaf'
                                 and term != '\xa3\xa8'and term != '\xa3\xa9'and term != '\xa1\xa2'
                                 ])
                
    return doc_list

def outputTextParse(filename):
    text = open(filename).read()
    ptn = re.compile('<text>|</text>|\\s*')
    outputText = ptn.sub('',text)
    return outputText

##############################类别向量生成#############################################
def gen_class_list_n(k):
    class_list = []
    for i in range(k):
	class_list.append(0)  #生成否定性评论类别列表
    return class_list

def gen_class_list_p(k):
    class_list = []
    for i in range(k):
	class_list.append(1)  #生成肯定性评论类别列表
    return class_list

##############################词条向量生成#############################################
def createTermSet(doc_list):        #返回文档中出现的所有词组成的词条集合
    termSet = set([])
    for doc in doc_list:
        termSet = termSet | set(doc)
    
    return list(termSet)

def saveTermSet(termSet, termSetfname):
    open(termSetfname, 'w').writelines([term + '\n' for term in termSet])

def bagOfTerms2Vec(termSet,inputSet):   #TF模型
    returnVec = [0]*len(termSet)
    for term in inputSet:
        if term in termSet:
            returnVec[termSet.index(term)] += 1
        
    return returnVec

def setOfTerms2Vec(termSet,inputSet):   #BOOL模型
    returnVec = [0]*len(termSet)
    for term in inputSet:
        if term in termSet:
            returnVec[termSet.index(term)] = 1
        
    return returnVec

def createTrainMat(termSet,doc_list):
    trainMat = []
    for doc in doc_list:
       trainMat.append(setOfTerms2Vec(termSet,doc)) 
    return trainMat

def saveTrainMat(trainMat, trainMatfname):
    open(trainMatfname, 'w').writelines([str(mat) + '\n' for mat in trainMat])


##############################################特征选择###########################################################################
def get_term_dict(doc_list):
    term_dict = {}
    for doc in doc_list:  
        for term in doc:
            term_dict[term] = 1    
    term_list = sorted(term_dict.keys())       
    term_dict = dict(zip(term_list, range(len(term_list))))
    return term_dict      
def get_class_dict(class_list): 
    class_set = sorted(list(set(class_list)))
    class_dict = dict(zip(class_set, range(len(class_set))))
    return  class_dict

def stats_term_df(doc_list, term_dict, term_set):   #计算含有某词条的文档数目
    term_df_dict = {}.fromkeys(term_dict.keys(), 0)
    for term in term_set:
        for doc_terms in doc_list:
            if term in doc_terms:
                term_df_dict[term] +=1                
    return term_df_dict

def stats_class_df(class_list, class_dict):
    class_df_list = [0] * len(class_dict)
    for doc_class in class_list:
        class_df_list[class_dict[doc_class]] += 1
    return class_df_list

def stats_term_class_df(doc_list, class_list, term_dict, class_dict):
    term_class_df_mat = np.zeros((len(term_dict), len(class_dict)), np.float32)  #初始化0矩阵，行数等于词条数，列数等于类别数
    for k in range(len(class_list)):
        class_index = class_dict[class_list[k]]
        doc_terms = doc_list[k]
        for term in set(doc_terms):
            term_index = term_dict[term]
            term_class_df_mat[term_index][class_index] +=1
    return  term_class_df_mat

def feature_selection_ig(class_df_list, term_set, term_class_df_mat, thr):
    A = term_class_df_mat   #包含词条ti，并且类别属于Cj的文档数量
    B = np.array([(sum(x) - x).tolist() for x in A])    #包含词条ti，并且类别属于不Cj的文档数量
    C = np.tile(class_df_list, (A.shape[0], 1)) - A  #不包含词条ti，并且类别属于Cj的文档数量
    N = sum(class_df_list)   #总的文档数量
    D = N - A - B - C        #不包含词条ti，并且类别属于不Cj的文档数量
    term_df_array = np.sum(A, axis = 1)
    class_set_size = len(class_df_list)
    
    p_t = term_df_array / N
    p_not_t = 1 - p_t
    p_c_t_mat =  (A + 1) / (A + B + class_set_size)
    p_c_not_t_mat = (C+1) / (C + D + class_set_size)
    p_c_t = np.sum(p_c_t_mat  *  np.log(p_c_t_mat), axis =1)
    p_c_not_t = np.sum(p_c_not_t_mat *  np.log(p_c_not_t_mat), axis =1)
    
    term_score_array = p_t * p_c_t + p_not_t * p_c_not_t
    sorted_term_score_index = term_score_array.argsort()[: : -1]    #descending sort
    term_set_fs = [term_set[index] for index in sorted_term_score_index][0:thr] 
    return term_set_fs

def feature_selection_mi(class_df_list, term_set, term_class_df_mat, thr):
    A = term_class_df_mat
    B = np.array([(sum(x) - x).tolist() for x in A])
    C = np.tile(class_df_list, (A.shape[0], 1)) - A
    N = sum(class_df_list)
    class_set_size = len(class_df_list)
    
    term_score_mat = np.log(((A+1.0)*N) / ((A+C) * (A+B+class_set_size)))
    term_score_max_list = [max(x) for x in term_score_mat]
    term_score_array = np.array(term_score_max_list)
    sorted_term_score_index = term_score_array.argsort()[: : -1]
    term_set_fs = [term_set[index] for index in sorted_term_score_index][0:thr]
    
    return term_set_fs

def feature_selection_wllr(class_df_list, term_set, term_class_df_mat, thr):
    A = term_class_df_mat
    B = np.array([(sum(x) - x).tolist() for x in A])
    C_Total = np.tile(class_df_list, (A.shape[0], 1))
    N = sum(class_df_list)
    C_Total_Not = N - C_Total
    term_set_size = len(term_set)
    
    p_t_c = (A + 1E-6) / (C_Total + 1E-6 * term_set_size)
    p_t_not_c = (B +  1E-6) / (C_Total_Not + 1E-6 * term_set_size)
    term_score_mat = p_t_c  * np.log(p_t_c / p_t_not_c)
    
    term_score_max_list = [max(x) for x in term_score_mat]
    term_score_array = np.array(term_score_max_list)
    sorted_term_score_index = term_score_array.argsort()[: : -1]
    term_set_fs = [term_set[index] for index in sorted_term_score_index][0:thr]
    
    return term_set_fs

def feature_selection(doc_terms_list, doc_class_list, fs_method, thr):
    class_dict = get_class_dict(doc_class_list)
    term_dict = get_term_dict(doc_terms_list)
    class_df_list = stats_class_df(doc_class_list, class_dict)
    term_class_df_mat = stats_term_class_df(doc_terms_list, doc_class_list, term_dict, class_dict)
    term_set = [term[0] for term in sorted(term_dict.items(), key = lambda x : x[1])]
    term_set_fs = []
    
    if fs_method == 'MI':
        term_set_fs = feature_selection_mi(class_df_list, term_set, term_class_df_mat, thr)
    elif fs_method == 'IG':
        term_set_fs = feature_selection_ig(class_df_list, term_set, term_class_df_mat, thr)
    elif fs_method == 'WLLR':
        term_set_fs = feature_selection_wllr(class_df_list, term_set, term_class_df_mat, thr)
        
    return term_set_fs

################################################对分类器进行训练，使用其分类分类和测试#################################
def trainNB0(trainMat,trainClassList):
    numTrainDocs = len(trainMat)
    numWords = len(trainMat[0])
    p_positive = sum(trainClassList)/float(numTrainDocs)
    p0Num = ones(numWords); p1Num = ones(numWords)       
    p0Denom = 2.0; p1Denom = 2.0                        
    for i in range(numTrainDocs):
        if trainClassList[i] == 1:
            p1Num += trainMat[i]
            p1Denom += sum(trainMat[i])
        else:
            p0Num += trainMat[i]
            p0Denom += sum(trainMat[i])
    p1Vect = log(p1Num/p1Denom)         #取log，避免下溢出 
    p0Vect = log(p0Num/p0Denom)          
    return p0Vect,p1Vect,p_positive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0

def testingNB():
    trainfname_n = '/home/admin/odps_clt_release_64/bin/output/dataset/negative.txt'
    trainfname_p = '/home/admin/odps_clt_release_64/bin/output/dataset/positive.txt'
    doc_list_n = testTextParse(trainfname_n)
    k_n = len(doc_list_n)
    doc_list_p = testTextParse(trainfname_p)
    k_p = len(doc_list_p)
    doc_list_n.extend(doc_list_p)
    doc_list = doc_list_n
    class_list_n = gen_class_list_n(k_n) 
    class_list_p = gen_class_list_p(k_p)
    class_list_n.extend(class_list_p)
    class_list = class_list_n
    termSet = createTermSet(doc_list)
    trainMat=createTrainMat(termSet,doc_list)
    p0V,p1V,p_p = trainNB0(array(trainMat),array(class_list))
    testfname1 = '/home/admin/odps_clt_release_64/bin/output/dataset/testDataset1.txt'
    test_list1 = testTextParse(testfname1)
    testEntry1 = createTermSet(test_list1)
    outputText1 = outputTextParse(testfname1)
    thisDoc = array(bagOfTerms2Vec(termSet,testEntry1))
    print 'The comment:',outputText1,'is classified as: ',classifyNB(thisDoc,p0V,p1V,p_p)
    
    testfname0 = '/home/admin/odps_clt_release_64/bin/output/dataset/testDataset0.txt'
    test_list0 = testTextParse(testfname0)
    testEntry0 = createTermSet(test_list0)
    outputText0 = outputTextParse(testfname0)
    thisDoc = array(bagOfTerms2Vec(termSet,testEntry0))
    print 'The comment:',outputText0,'is classified as: ',classifyNB(thisDoc,p0V,p1V,p_p)
    print '肯定性文档概率为：',p_p
    
##################分别计算不同测试集和训练集下的错误率#############################    
def get_error_rate(test_doc_list, train_doc_list):  #分别取前400个negative、positive评论作为测试集，其它数据作为训练集计算错误率
    train_class_list_n = gen_class_list_n(1600)
    train_class_list_p = gen_class_list_p(1600)
    train_class_list_n.extend(train_class_list_p)
    train_class_list = train_class_list_n
    
    test_class_list_n = gen_class_list_n(400)
    test_class_list_p = gen_class_list_p(400)
    test_class_list_n.extend(test_class_list_p)
    test_class_list = test_class_list_n
    
   
    #fs_method = 'IG'
    fs_method = 'MI'
    #fs_method = 'WLLR'
    #thr = 500
    #thr = 1000
    #thr = 2000
    thr = 5000
    
    term_set_fs = feature_selection(train_doc_list, train_class_list, fs_method, thr)  
   
    trainMat = createTrainMat(term_set_fs, train_doc_list)
    
    test_size = range(800)
    
    p0V,p1V,p_p = trainNB0(array(trainMat),array(train_class_list))
    errorCount = 0
    for docIndex in test_size:       
        #wordVector = setOfTerms2Vec(term_set_fs, test_doc_list[docIndex])   #BOOL模型
        wordVector = bagOfTerms2Vec(term_set_fs, test_doc_list[docIndex])  #TF模型
        if classifyNB(array(wordVector),p0V,p1V,p_p) != test_class_list[docIndex]:
            errorCount += 1
            #print "classification error",test_doc_list[docIndex]
    error_rate = float(errorCount)/len(test_doc_list)
    print error_rate, errorCount
    return error_rate
   
def cross_validate():
    nfilename = '/home/admin/odps_clt_release_64/bin/output/dataset/negative.txt'
    pfilename = '/home/admin/odps_clt_release_64/bin/output/dataset/positive.txt'
    test_doc_list1n = cvTextParse(nfilename,1,401)
    test_doc_list1p = cvTextParse(pfilename,1,401)
    test_doc_list1n.extend(test_doc_list1p)
    test_doc_list1 = test_doc_list1n
    train_doc_list1n = cvTextParse(nfilename,401,2001)
    train_doc_list1p = cvTextParse(pfilename,401,2001)
    train_doc_list1n.extend(train_doc_list1p)
    train_doc_list1 = train_doc_list1n

    error_rate1 = get_error_rate(test_doc_list1, train_doc_list1)


    test_doc_list2n = cvTextParse(nfilename,401,801)
    test_doc_list2p = cvTextParse(pfilename,401,801)
    test_doc_list2n.extend(test_doc_list2p)
    test_doc_list2 = test_doc_list2n
    
    train_doc_list2n = cvTextParse(nfilename,1,401)
    train_doc_list2n.extend(cvTextParse(nfilename,801,2001))
    train_doc_list2p = cvTextParse(pfilename,1,401)
    train_doc_list2p.extend(cvTextParse(pfilename,801,2001))
    train_doc_list2n.extend(train_doc_list2p)
    train_doc_list2 = train_doc_list2n

    error_rate2 = get_error_rate(test_doc_list2, train_doc_list2)

    test_doc_list3n = cvTextParse(nfilename,801,1201)
    test_doc_list3p = cvTextParse(pfilename,801,1201)
    test_doc_list3n.extend(test_doc_list3p)
    test_doc_list3 = test_doc_list3n
    
    train_doc_list3n = cvTextParse(nfilename,1,801)
    train_doc_list3n.extend(cvTextParse(nfilename,1201,2001))
    train_doc_list3p = cvTextParse(pfilename,1,801)
    train_doc_list3p.extend(cvTextParse(pfilename,1201,2001))
    train_doc_list3n.extend(train_doc_list3p)
    train_doc_list3 = train_doc_list3n


    error_rate3 = get_error_rate(test_doc_list3, train_doc_list3)

    test_doc_list4n = cvTextParse(nfilename,1201,1601)
    test_doc_list4p = cvTextParse(pfilename,1201,1601)
    test_doc_list4n.extend(test_doc_list4p)
    test_doc_list4 = test_doc_list4n
    
    train_doc_list4n = cvTextParse(nfilename,1,1201)
    train_doc_list4n.extend(cvTextParse(nfilename,1601,2001))
    train_doc_list4p = cvTextParse(pfilename,1,1201)
    train_doc_list4p.extend(cvTextParse(pfilename,1601,2001))
    train_doc_list4n.extend(train_doc_list4p)
    train_doc_list4 = train_doc_list4n

    error_rate4 = get_error_rate(test_doc_list4, train_doc_list4)

    test_doc_list5n = cvTextParse(nfilename,1601,2001)
    test_doc_list5p = cvTextParse(pfilename,1601,2001)
    test_doc_list5n.extend(test_doc_list4p)
    test_doc_list5 = test_doc_list4n
    
    train_doc_list5n = cvTextParse(nfilename,1,1601)
    train_doc_list5p = cvTextParse(pfilename,1,1601)
    train_doc_list5n.extend(train_doc_list5p)
    train_doc_list5 = train_doc_list5n

    error_rate5 = get_error_rate(test_doc_list5, train_doc_list5)
    
    print error_rate1, error_rate2, error_rate3, error_rate4, error_rate5
    print 'the error rate is: ',float(error_rate1 + error_rate2 + error_rate3 + error_rate4 + error_rate5)/float(5)

