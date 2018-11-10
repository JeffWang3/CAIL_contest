import fasttext
import sys

if __name__=="__main__":
    if len(sys.argv) != 5:
        print("Please enter the input file!")
        print("Please enter the Path and the Name of model!")
        print("Please enter the epoch!")
        print("w2v file u/bt/n ")
        sys.exit()

    infile = sys.argv[1]
    model = sys.argv[2]
    epoch_num = int(sys.argv[3])
    w2v_flag = sys.argv[4]
    w2v_u = "/Users/kate/Desktop/word2vector.txt"
    w2v_t = "/Users/kate/Desktop/dissertation/word2vec/embedding/traindata_w2v.txt" 
    model_name = model+"_"+sys.argv[3]+"_"+sys.argv[4]
    if w2v_flag == "u":
        classifier = fasttext.supervised(infile, model_name, label_prefix='__label__', dim=64, pretrained_vectors=w2v_u, epoch=epoch_num)
    elif w2v_flag == "t":
        classifier = fasttext.supervised(infile, model_name, label_prefix='__label__', dim=64, pretrained_vectors=w2v_t, epoch=epoch_num)
    else:
        classifier = fasttext.supervised(infile, model_name, label_prefix='__label__')
    
