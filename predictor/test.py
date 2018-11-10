import sys
import json
from predictor import Predictor

def get_score(pre_label, ori_label):
    TP = 0
    FP = 0
    FN = 0
    for l in label_list:
        if l in result_dict[key]:
            TP += 1
        else:
            FP += 1
    FN = len(ori_label) - TP
    if TP == 0:
        if FN == 0 and FP == 0:
            precision = 1.0
            recall = 1.0
            f1 = 1.0
        else:
            precision = 0
            recall = 0
            f1 = 0
    else:
        precision = round(TP/(TP+FP), 5)
        recall = round(TP/(TP+FN), 5)
        f1 = round(2*precision*recall/(precision+recall))
    print(TP, FP, FN, precision, recall, f1)
    return TP, FP, FN, precision, recall, f1


if __name__=="__main__":
    if len(sys.argv) != 3:
        print("Please enter the Path of test json!")
        print("Please enter the Path of output file!")
        sys.exit()

    testfile = sys.argv[1]
    outfile = sys.argv[2]  
    pre = Predictor()

    ftest = open(testfile, "r")
    fout = open(outfile, "w")
    tlines = ftest.readlines()
    result_dict = {}

    for i in range(0,len(tlines)):
        content = json.loads(tlines[i].replace("\n",""))
        result_dict[content["fact"]] = content["meta"]["relevant_articles"]

    TP_list = []
    FP_list = []
    FN_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    pre_classifier = Predictor()
    for key in result_dict.keys():
        print("-------------------")
        #print(result_dict[key])
        labels = pre_classifier.predict([key])
        label_list = labels[0]['articles']
        print(label_list)
        
        tp,fp,fn, precision, recall, f1 = get_score(label_list, result_dict[key])
        TP_list.append(tp)
        FP_list.append(fp)
        FN_list.append(fn)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    # total score
    TP_micro = 0
    for tpi in TP_list:
        TP_micro += tpi
    FP_micro = 0
    for fpi in FP_list:
        FP_micro += fpi
    FN_micro = 0
    for fni in FN_list:
        FN_micro += fni
    #print(TP_micro, FP_micro, FN_micro)
    precision_micro = TP_micro/(TP_micro+FP_micro)
    recall_micro = TP_micro/(TP_micro+FN_micro)
    F1_micor = 2*precision_micro*recall_micro/(precision_micro+recall_micro)
    total_f1 = 0
    for f1i in f1_list:
        total_f1 += f1i
    F1_macro = total_f1/len(f1_list)
    total_score = (F1_macro+F1_micor)*100/2
    print(round(total_score,5))