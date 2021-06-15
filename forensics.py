


import hashlib
import pyewf
import pytsk3
import random
import pandas as pd
from random_words import RandomWords
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

#this is the function used to securely hash each file as it is loaded, add to a dictionary and display as dataframe
activeimage = []
hashactive = []
worklist =[]
def hashPasser(worklist):
    for i in worklist:
        encoded = i.encode('utf-8')
        workhash = hashlib.md5()
        workhash.update(encoded)
        result = workhash.hexdigest()
        activeimage.append(i)
        hashactive.append(result)
        activeoutput = dict(zip(activeimage,hashactive))
        testoutput = pd.DataFrame.from_dict(activeoutput, orient='index',columns=["MD5 Hash"])
    return testoutput
#creates a sequence of dummyfiles to test
def RandomFile():
            rw = RandomWords()
            fileformat = [".txt",".doc",".html",".xls",]
            filename = rw.random_word()
            fileoutput = filename + random.choice(fileformat)
            worklist.append(fileoutput)
for i in range(50):
    RandomFile()
hashPasser(worklist)


# In[5]:


def convert_time(ts):
    if str(ts) == 0:
        return ""
    return datetime.fromtimestamp(ts)
    
def hashPasser(i):
        filestring = str(i)
        encoded = filestring.encode('utf-8')
        workhash = hashlib.md5()
        workhash.update(encoded)
        result = workhash.hexdigest()
        return result
def getFiles(path):
    directoryObject = filesystemObject.open_dir(path)
    recordFiles(directoryObject)
def setFiles(dirs):
    for directoryObject in dirs:
        print("processing")
        getInfo(directoryObject)
    return data
def getInfo(directoryObject):
        global dirs
        dirs= []
        for entryObject in directoryObject:
            file_name = entryObject.info.name.name
            file_ext = file_name.rsplit(b'.')[-1].lower()
            file_path = "{}{}".format(
            "/".join("/"), entryObject.info.name.name)
            f_type = entryObject.info.meta.type
            size = entryObject.info.meta.size
            create = convert_time(entryObject.info.meta.crtime)
            change = convert_time(entryObject.info.meta.ctime)
            modify = convert_time(entryObject.info.meta.mtime)
            file_hash = hashPasser(file_name)
            if entryObject.info.meta.type == pytsk3.TSK_FS_META_TYPE_DIR:
                dirs.append(directoryObject)
            data[file_name] =[file_ext,file_path,f_type,create,change,modify,size,file_hash]
        return data
def recordFiles(directoryObject):
        getInfo(directoryObject)
        print("complete")                
folders = []
class ewf_Img_Info(pytsk3.Img_Info):
      def __init__(self, ewf_handle):
        self._ewf_handle = ewf_handle
        super(ewf_Img_Info, self).__init__(
            url="", type=pytsk3.TSK_IMG_TYPE_EXTERNAL)
        def close(self):
            self._ewf_handle.close()
            
      def read(self, offset, size):
        self._ewf_handle.seek(offset)
        return self._ewf_handle.read(size)

      def get_size(self):
        return self._ewf_handle.get_media_size()
#define image as filename of local variable
image = ("CraigTuckerDesktop.E01")
#open file as filelike object and allow writing
file_object = open(image, "rb")
#assign pyewf values to overwrite pytsk values 
ewf_handle = pyewf.handle()
e01_metadata = ewf_handle
ewf_handle.open_file_objects([file_object])
filenames = pyewf.glob(image)
imagehandle = ewf_Img_Info(ewf_handle)
img_info =ewf_Img_Info(ewf_handle)
ewf_handle.open(filenames)
#open handle and print partiton table
partitionTable = pytsk3.Volume_Info(img_info)
for partition in partitionTable:
    print (partition.addr, partition.desc,partition.start, partition.start * 512, partition.len)
    if(b'NTFS') in partition.desc:
        filesystemObject = pytsk3.FS_Info(imagehandle, offset=(partition.start*512))
        #open master directory
        data = {}
        getFiles('/')
        setFiles(dirs)
        print("By default, this function returns all files in a hard drive, If you wish to search for specific directory,repeat the function for passing file names")
        testoutput = pd.DataFrame.from_dict(data, orient='index',columns=["Name","Path","PYTSK Type","Create","Change","Modify","Size","MD5 Hash"])
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        display(testoutput)


# In[2]:


def sortFiles(filelist,):
    extensions = {}
    for i in filelist:
        if i in dirs:
            print("this is a folder and has an existing list")
        elif i not in extensions:
            extensions.append(i)
        else:
            extensions[i]=[file_name]
    return extensions


# In[23]:


pip install pycrypto


# In[34]:


# this is a function to provide a facility to securely write and encrypt file hashes to an external file, preserving integrity
from Crypto.Cipher import AES
from Crypto import Random
key = b'Sixteen byte key'
randomblock = Random.new().read(AES.block_size)
cipher = AES.new(key, AES.MODE_CFB, randomblock)
def hashWrite(filename,filehash):
    f = open("Hash.txt", "a")
    filedictionary = zip(filename,filehash)
    for i in filedictionary:
        stringfile = str(i)
        message = randomblock+cipher.encrypt(stringfile)
        strhash = str(message)
        print(message)
        f.write(strhash+'\n')
    f.close
    return "complete"
filename = ["john","joe","jamie"]
filehash = ["1","2","3"]
hashWrite(filename,filehash)


# In[58]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Load scikit's random forest classifier library
import numpy as np
import string
import gensim.downloader as api
from scipy.spatial.distance import cdist
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# get synonyms of most common words
def getCosine(list1,list2):
    worddict = {}
    word_vectors = api.load("glove-wiki-gigaword-100")
    for (a,b) in zip(list1,list2):
        compare1 = a
        compare2 = b
        similarity = word_vectors.wmdistance(compare1, compare2)
        worddict[compare1]=[compare2,similarity]
    global dictframe        
    dictframe = pd.DataFrame.from_dict(worddict, orient='index',columns=["Comparison Word","Word Movers Distancce"])
    display(dictframe)
    return dictframe
x = ["king","queen","castle"]
y = ["chess","balloon","man"]
def expandList(listinput):
    synonymlist = []
    global completelist
    completelist = []
    for i in listinput:
        for syn in wordnet.synsets(i):
            for l in syn.lemmas():
                synonymlist.append(l.name())
    completelist = synonymlist + listinput
    return completelist  
#tokenise and remove stopwords and digits
def removeWord(wordlist):
    stop_words = set(stopwords.words('english')) 
    stoplist = stopwords.words('english') + list(string.punctuation)+ list(string.digits)
    stoplist = set(stoplist)
    global cleanlist
    cleanlist = [word for word in word_tokenize(wordlist) if word.lower() not in stoplist and not word.isdigit()]
    return cleanlist
#import attack dataset
file = r'dataset.xls'
pd.set_option("display.max_rows", None, "display.max_columns", None)
df = pd.read_excel(file, encoding = 'iso8859-1')
dfpart= df.head(10000)
#get all descriptions
#transfer all items in the df column to a list. This is a written summary of the attacks 
dfpart['summary'].dropna(inplace=True)
textframe = dfpart['summary'].to_string()
removeWord(textframe)
#split incidents into seperate list
count = CountVectorizer()
x = count.fit_transform(cleanlist)
print(x.shape)
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(x)
features = count.get_feature_names()
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=features,columns=["Importance Weight"],)
df_idf.sort_values(by=['Importance Weight'], ascending =False,inplace=True)
display(df_idf)
#next, get the head of the dataframe to get the 100 most common words
dfwords =df_idf.head(100)
commonwords = dfwords.index.tolist()
expandList(commonwords)
# Now there is a knowledge base of words of  interest, compute similarity of the wordlist and files
getCosine(commonwords,completelist)
# now perform LDA to guess topics
wordsarray = x
LDA = LatentDirichletAllocation(n_components=10, random_state=100)
LDA.fit(wordsarray)
first_topic = LDA.components_[0]
top_topic_words = first_topic.argsort()[-10:]
for i,topic in enumerate(LDA.components_):
    print(f'Top 10 words for topic #{i}:')
    print([count.get_feature_names()[i] for i in topic.argsort()[-10:]])
    print('\n')
topic_values = LDA.transform(x)
topic_values.shape
topic_frame = pd.DataFrame(topic_values,index =cleanlist,columns=[topic.argsort()[-10:]])
display(topic_frame)


# In[ ]:



