# -*- coding: utf-8 -*-
"""
@author: yijin
"""

import os
import pandas as pd
import re
from konlpy.tag import Okt
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

def set_root_dir(root_directory):
    if root_directory == 'colab':
        root_dir = os.getcwd() + '/drive/My Drive/product_category_classification/'   
        data_dir = root_dir + 'data/'  
    else: ##root_directory == 'local'
        root_dir = os.getcwd()
        data_dir = root_dir + '\\data\\'
    
    print('YOUR WORKING DIR: ', root_dir)
    print('YOUR DATA DIR: ', data_dir)
    return root_dir, data_dir
          
def read_csv(filename):
    return pd.read_csv(data_dir + filename, encoding='utf-8')

def save_csv(df, filename):
    return df.to_csv(data_dir + filename, index=False)
     
def shape_type_of_data(filename):
    data = read_csv('bunjang_product.csv')
    # category = pd.read_csv(DATA_DIR + 'src_data/bunjang_category.csv', encoding='utf-8')

    print('Shape of all data: ', data.shape)
    print('DTypes of product data')
    print(data.dtypes)

class StepOne:
     
    def __init__(self, filename, column_list):
        self.data = read_csv(filename)
        self.column_list = column_list
        print('shape:', self.data.shape)
          
    def stepone(self):
        self.extract_two_weeks()
        self.organize_rows_cols()
        print(self.data.shape)
     
    def extract_two_weeks(self):
        temp1 = self.data['create_date'].str[0:10]>= '2019-12-03'
        temp2 =  self.data['create_date'].str[0:10] <= '2019-12-16'
        self.data =  self.data[temp1 &temp2]
          
        print('Data is for {} days and shape is {}'.format(len(pd.unique( self.data['create_date'].str[0:10])), self.data.shape))  #Check if the data is for 14 days
        #print from the start to the end for 14days data for checking
        print( self.data['create_date'].head(1))
        print("-----")
        print( self.data['create_date'].tail(1))

        save_csv( self.data, 'extract_two_weeks.csv')
          
    def organize_rows_cols(self):
        self.data = self.data[column_list]
        print(self.data.shape)
        self.data = self.data.dropna(axis=0)
        print(self.data.shape)
        self.data= self.data.drop_duplicates()
        print(self.data.shape)
        self.split_category()
        print(self.data.shape)
        self.del_unnecessary_cate()
        print(self.data.shape)
        
        save_csv(self.data, 'step1.csv')
                   
          
    def split_category(self):
        self.data['category_id'] = self.data['category_id'].map(lambda x:int(x))
        self.data['cate_one']   = self.data['category_id'].map(lambda x: str(x)[0:3])
        self.data['cate_two']   = self.data['category_id'].map(lambda x: str(x)[0:6])
        # self.data['cate_three'] = self.data['category_id'].map(lambda x: str(x)[0:9])
        self.data = self.data.drop(['category_id'], axis=1)
          
    def del_unnecessary_cate(self):
        unnecessary_cate = [100, 200, 210, 220, 230, 240, 300, 999]
        for cate in unnecessary_cate:
            temp = self.data[self.data.cate_one.astype(int) == cate]
            self.data = self.data.drop(temp.index)
               
        temp = self.data[self.data.cate_two.astype(int)<1000]
        self.data = self.data.drop(temp.index)        
          
class StepTwo:
    def __init__(self, filename, column_list):
        self.data = read_csv(filename)
        self.column_list = column_list
        print('shape:', self.data.shape)
          
         
    def steptwo(self):
        for colname in self.column_list:
              self.data[colname] = self.data[colname].apply(self.remove_new_line_character)
              self.data[colname] = self.data[colname].apply(self.re_sub_part)
              self.data[colname] = self.data[colname].apply(self.remove_spaces)               
        save_csv(self.data, 'step2.csv')
          
    def cleaning(self, str):
        self.remove_new_line_character()
        self.re_sub_part()
        self.remove_spaces()
          
    def remove_new_line_character(self, str):
        return re.sub(r'\\n', ' ', str)
    def re_sub_part(self, str):
        return re.sub('[^0-9a-zA-Z가-힗]', ' ', str)
    def remove_spaces(self, str):
        return re.sub(' +', ' ', str)
          
class StepThree:
    def __init__(self, filename, column_list):
        self.data = read_csv(filename)
        self.column_list = column_list
        print('shape:', self.data.shape)
          
          
    def stepthree(self):
        for colname in self.column_list:
            self.data[colname] = self.data[colname].map(self.okt_token)
        save_csv(self.data, 'step3.csv')
     
     
    def okt_token(self, text):
        okt = Okt()
        s = ''
        pos = okt.pos(text, norm = True, stem = True)
        for keyword, type in pos:
            if type == 'Noun' or type == 'Alpha' or type == 'Number':
                s = s + ' ' + keyword
        return s
    
class StepFour:
    def __init__(self, filename, column_list):
        self.data = read_csv(filename)
        self.column_list = column_list
          
          
    def stepfour(self):
        for colname in self.column_list:
            self.data[colname] = self.data[colname].map(self.remove_stopword)
               
        save_csv(self.data, 'step4.csv')
        
    def remove_stopword(self, word):
        stopword_str = "제품 사이즈 상품 구매 배송 판매 문의 시 택배 상태 사진 교환 환불 가격 만 후 정품 연락 일 사용 직거래 가능 거래 및 색상 수 분 개 이상 cm 톡 하자 무료 비 퀄리티 정도 배송비 상점 주문 확인 번 카톡 더 반품 착용 중고 블랙 번톡 입금 번개 가슴 박스 할인 한번 추가 실 경우 S 등 시간 X 특성 제 모든 기간 부분 눌 구입 용 품 불가 용감 포함 해외 x L 주시 천 옷 때 안 년 발송 참고 M 전 바로 점 단 어깨 택포 케이스 국내 꼭 거의 가죽 총장 오염 길이 "
        stop_word = stopword_str.split(' ')
        s = ''
        word_tokens = word_tokenize(str(word))
        for token in word_tokens:
            if token not in stop_word:
                s = s + token + ' '
        return s                     
          
class ExploreData:
    def __init__(self,filename, column_list):
        self.data = read_csv(filename)
        self.column_list = column_list

    def explore_data(self):
        print('length of string')
        for colname in self.column_list:
            print('Column: ', colname)
            self.string_length(colname)

        print('================')
        for colname in self.column_list:
            print('word count: ', colname)
            self.count_words(colname)

        print('================')
        for colname in self.column_list:
            print('word frequency: ', colname)
            self.most_common_words(colname)


    def string_length(self,colname):
        self.data[colname+'_strlen'] = self.data[colname].str.len()
        print('└Max: ', max(self.data[colname +'_strlen']))
        print('└average: ', self.data[colname +'_strlen'].mean())
         
    def count_words(self,colname):
        self.data[colname +'_wordlen'] = self.data[colname].str.split().str.len()
        print('└Max: ', max(self.data[colname +'_wordlen']))
        print('└average: ', self.data[colname +'_wordlen'].mean())

    def most_common_words(self, colname):
      with open(data_dir + '{}_word_freq.txt'.format(colname), 'w', encoding='utf-8')as f:
        temp = self.data[colname].values
        temp = ' '.join(' '.join(temp.astype(str)).split())
        temp = temp.split(' ')
        globals()['c_{}'.format(colname)] = Counter(temp)
        count = 0
        for word, freq in globals()['c_{}'.format(colname)].most_common():
            if freq >= 10:
                f.write("{} {}\n".format(word, freq))
                count = count + 1
                if count == 1000:
                    f.write('======================')
            else: #freq under 10
                break      
        f.write('Number of most common in {} : {} {}'.format(colname, count, len(globals()['c_{}'.format(colname)])))    
      print('Number of most common in {} : {} {}'.format(colname, count, len(globals()['c_{}'.format(colname)])))      



if __name__ == "__main__":
     # where you are working 'colab' or 'local'
    root_dir, data_dir = set_root_dir('local')

#     shape_type_of_data('bunjang_product.csv')

    column_list = ['name', 'keyword', 'category_id', 'description']
    column_str_list = ['name', 'keyword', 'description']

    step1 = StepOne('bunjang_product.csv', column_list)
    step1.stepone()
    
    step2 = StepTwo('step1.csv', column_str_list)
    step2.steptwo()

    step3 = StepThree('step2.csv', column_str_list)
    step3.stepthree()

    step4 = StepFour('step3.csv', column_str_list)
    step4.stepfour()

    exploredata = ExploreData('step4.csv', column_str_list)
    exploredata.explore_data()  