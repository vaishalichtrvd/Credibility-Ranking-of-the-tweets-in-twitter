import pandas as pd
import re
import collections
from string import whitespace
import os
os.chdir("C:/Users/Ani/Documents")
pd.set_option('display.max_colwidth', 30000)
My_col=['Date','Tweet_Text','Tweet_Id','User_Id','User_Name','User_Screen_Name','Retweets','Favorites']
try:
    Raw_file=pd.read_csv('log - Copy.csv', sep=',', usecols=My_col, index_col=None)
except (FileNotFoundError, FileExistsError, MemoryError) as e:
    print("file is not in correct format")

Processed_file = pd.DataFrame(columns=My_col)
Processed_file=Raw_file.dropna(subset=My_col)
Processed_file['Date']=Processed_file['Date'].astype(str)

try:
    Processed_file['Date']=Processed_file['Date'].map(lambda x: re.match(r"[^[]*\[([^]]*)\]", x).groups()[0])
except(TypeError,SyntaxError, SystemExit,SyntaxWarning) as e:
    print("Syntax error")

try:
    Processed_file=Processed_file.sort_values(by='Date')
except ValueError:
    print("Values are not in correct format")

Processed_file=Processed_file[(Processed_file['Date'] > '2011-08-04') & (Processed_file['Date'] < '2011-08-12')]
Processed_file['Tweet_Text']=Processed_file['Tweet_Text'].astype(str)
Processed_file['Tweet_length']=Processed_file['Tweet_Text'].map(lambda x: len(x.translate(str.maketrans('','', whitespace))))
Processed_file['Number_of_URL']=Processed_file['Tweet_Text'].map(lambda x: x.count('http'))
Processed_file['No_of_@_word']=Processed_file['Tweet_Text'].map(lambda x: x.count('@'))
Processed_file['No_of_hash_word']=Processed_file['Tweet_Text'].map(lambda x: x.count('#'))
Processed_file['User_Screen_Name']=Processed_file['User_Screen_Name'].astype(int)
print(Processed_file['User_Screen_Name'])
Processed_file['Length_of_User_Name']=Processed_file['User_Screen_Name'].map(lambda x: len(str(x)))

Spam_count={}
def Spam_word_count(Word):
    print(Word)
    Spam_list = ['London','Riots']
    for i in Spam_list:
        try:
            Spam_count[i]=Word.count(i)
        except ValueError:
            print("Cant find the word list as a parameter")
    return sum(Spam_count.values())
Processed_file['Number_of_Spam_Word']=Processed_file['Tweet_Text'].map(lambda x: Spam_word_count(x.split(' ')))

Swear_count={}
def Swear_word_count(Word):
    print(Word)
    Swear_list = ['SwearGod','GodPromise']
    for i in Swear_list:
        try:
            Swear_count[i]=Word.count(i)
        except ValueError:
            print("Cant find the word list as a parameter")
    return sum(Swear_count.values())
Processed_file['Number_of_Swear_Word']=Processed_file['Tweet_Text'].map(lambda x: Swear_word_count(x.split(' ')))

print(Processed_file)

try:
    Processed_file.to_csv('processed.csv', header=None, sep=',')
except PermissionError:
    print("file is opened by someone, please rerun after closing the file")