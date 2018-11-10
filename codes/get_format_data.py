import jieba
import re
import sys

def get_tokens(text):
    '''
    split sentence into tokens by jieba
    '''
#    tokens_list = jieba.cut(text, cut_all=False)
    text = text.replace(" ","")
    tokens_list = []
    for i in range(0,len(text)):
        tokens_list.append(text[i])
    return tokens_list

def rmitems(text):
    '''
    remove useless items
    '''
    remove_items = u'[a-zA-Z0-9０-９Ａ-Ｙａ-ｙ，。？：“”"＝/#＠@§；,.:-?;\'&()<>＋（）_《》\uf8e7\u3000\x0c\／^L·-、、．%\%％－[\]［ ］\x7f-]+'
    text_rmitems = re.sub(remove_items, '', text)
    text_rmblank = re.sub("\t", '',text_rmitems)
    return text_rmblank

if __name__=="__main__":

    if(len(sys.argv)!= 4):
        print("Numbers of parameters is wrong !")
        print("Please enter the Path of the posfile!")
        print("Please enter the Path of the negfile!")
        print("Please enter the Path of the output file!")
        sys.exit()

    posfile = sys.argv[1]
    negfile = sys.argv[2]
    outfile = sys.argv[3]

    fpos = open(posfile,"r")
    fneg = open(negfile,"r")
    fout = open(outfile,"w")

    plines = fpos.readlines()
    for i in range(0,len(plines)):
        data = rmitems(plines[i].replace("\n",""))
        fout.write(" ".join(get_tokens(data))+"\t__label__POS\n")

    nlines = fneg.readlines()
    for i in range(0,len(nlines)):
        data = rmitems(nlines[i].replace("\n",""))
        fout.write(" ".join(get_tokens(data))+"\t__label__NEG\n")

    fpos.close()
    fneg.close()
    fout.close()
