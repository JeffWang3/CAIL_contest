import sys
import re
from tcnn import predict_law
import fasttext
import jieba
import json
import math



class Predictor:

    def __init__(self):
        self.batch_size = 2048
        self.model_task1 = "predictor/model/accusation_model.bin"
        self.model_task2 = "predictor/model/articles_model.bin"
        self.model_task3 = "predictor/model/imprisonment_model.bin"
        #self.law_dict = self.load_law()
        self.dropdict = self.load_dropdict()
    '''
    def load_law(self):
        law_dict = {}
        flaw = open("predictor/dict/law.txt","r",encoding="utf8")
        inlines = flaw.readlines()
        for i in range(0, len(inlines)):
            law_dict[int(inlines[i].replace("\n",""))] = i+1
        #print(law_dict)
        return law_dict
    '''

    def check_result(self, text, label_list1, label_list2, label_list3):
        '''
        卖淫
        '''
        if "卖淫" in text:
            final_maiyin = 16
            maiyin = [6,16,82,122,183]
            if "组织" in text:
                if "协助" in text:
                    final_maiyin = 82
                elif "强迫" in text:
                    final_maiyin = 6
                else:
                    final_maiyin = 16
            elif "强迫" in text:
                if "引诱" in text or "容留" in text or "介绍" in text:
                    final_maiyin = 6
                else:
                    final_maiyin = 122
            else:
                final_maiyin = 183
        if final_maiyin not in label_list1:
            label_list1.append(final_maiyin)
        for acc_ in label_list1:
            if acc_ in maiyin and acc_ != final_maiyin:
                label_list1.remove(acc_)

        return label_list1, label_list2, label_list3




    def load_dropdict(self):
        dropdict = []
        
        stopwords = "predictor/dict/drop.dict"
        fstop = open(stopwords,"r",encoding="utf8")
        slines = fstop.readlines()
        for i in range(0,len(slines)):
            dropdict.append(slines[i].replace("\n",""))
        '''
        ner = "predictor/dict/ner"
        fner = open(ner,"r",encoding="utf8")
        nlines = fner.readlines()
        for i in range(0,len(nlines)):
            dropdict.append(nlines[i].replace("\n",""))
        '''
        return dropdict

    def get_tokens(self, text):
        tokens_list = jieba.cut(text, cut_all=False)
        new_list = []
        for word in tokens_list:
            if word not in self.dropdict:
                new_list.append(word)
        return new_list

    def split(self, text):
        remove_list = u'[0-9a-zA-Z，。？：“”"＝/#＠@§；,.:-?;\'&()<>＋（）_《》\／^L·-、、．%\%％－[\]［ ］\x7f\n\t]+'
        ori_text = re.sub(remove_list,"",text)
        split_text = self.get_tokens(ori_text)
        return " ".join(split_text)

    def get_label(self, labels):
        #print(labels)
        label_list = []
        prob_current = 0
        for label, prob in labels:
            if prob_current == 0:
                prob_current = prob
                label_list.append(int(label.replace("__label__","")))
            elif prob > 0.1:
                label_list.append(int(label.replace("__label__","")))
        return label_list

    def get_label_single(self, labels):
        for label, prob in labels:
            return int(label.replace("__label__",""))

    def predict(self, content):
        result = []
        classifier_task1 = fasttext.load_model(self.model_task1)
        classifier_task2 = fasttext.load_model(self.model_task2)
        classifier_task3 = fasttext.load_model(self.model_task3)

        law_dict = {}
        law_file = open("predictor/model/spe_split", "r", encoding="utf-8")
        for line in law_file:
            law_item = json.loads(line)
            law_id = law_item["No."]
            law_fact = law_item["Content"]
            law_dict[law_id] = law_fact
    

        f = open('predictor/model/law.txt', 'r', encoding = 'utf8')
        law = {}
        lawname = {}
        line = f.readline()
        while line:
            lawname[len(law)+1] = line.strip()
            law[line.strip()] = len(law)
            line = f.readline()
        f.close()


        for text in content: 

            #print(text)
            text_content = self.split(text)
            #text_content = text
            #print(text_content)

            # accusation
            # labels_task1 = classifier_task1.predict_proba([text_content],k=3)
            # label_list1 = self.get_label(labels_task1[0])
            label_list1 = predict_law(text_content)


            # articles
            labels_task2 = classifier_task2.predict_proba([text_content],k=3)
            label_list2 = self.get_label(labels_task2[0])
            
            # imprisonment
            target_accu = int(lawname[int(label_list2[0])])
            if target_accu in law_dict.keys():
                combined_content = text_content + " "+ law_dict[target_accu]
            else:
                combined_content = text_content
            labels_task3 = classifier_task3.predict_proba([combined_content],k=1)
            label_list3 = self.get_label_single(labels_task3[0])
            #label_list3 = -1

           # r_1, r_2, r_3 = self.check_result(text, label_list1, label_list2, label_list3)

            result.append({
                "accusation": label_list1,
                "articles": label_list2,
                "imprisonment": label_list3
            })
        #print(result)
        return result
