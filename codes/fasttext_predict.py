import sys
import fasttext

if __name__=="__main__":
    if len(sys.argv) != 3:
        print("Please enter the Path of model!")
        print("Please enter the Path of test file!")
        sys.exit()

    model = sys.argv[1]
    testfile = sys.argv[2]
    print(model)    
    classifier = fasttext.load_model(model)

    result = classifier.test(testfile)
    print ('precision:', round(result.precision, 5))
    print ('recall:', round(result.recall, 5))
    print ('Number of examples:', result.nexamples)
