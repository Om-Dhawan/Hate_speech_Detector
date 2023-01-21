import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import re
import nltk
from nltk.util import pr
from tkinter import *
stemmer = nltk. SnowballStemmer ("english")
from nltk.corpus import stopwords                      
import string
stopword = set(stopwords.words ("english"))

df = pd.read_csv("AI&ML/data.csv")
# print(df.head())

df['labels'] = df['class'].map({0: "Hate Speech Detected", 1:"Offensive language detected", 2:"No hate and offensive speech"})
# print(df.head())

df=df[['tweet','labels']]

def clean (text):
    text= str(text).lower()
    text = re.sub('\[.*?\]', '',text) 
    text=re.sub('https?://\S+|www\.\S+', '', text) 
    text= re.sub('<.*?>+', '',text)
    text= re.sub('[%s]' % re.escape(string.punctuation),'',text)
    text = re.sub('\n', '', text)
    text= re.sub('\w*\d\w*', '', text)
    text= [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text= [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text

df["tweet"]= df["tweet"].apply(clean)
# print(df.head())

# df_new = df[np.isfinite(df).all(1)]
# df.dropna(inplace=True)
df_new=df.dropna()

x = np.array(df_new["tweet"])
y = np.array(df_new["labels"])
# print(x)
# print(y)
cv= CountVectorizer()
x = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 0.33, random_state= 42)    
# print(X_train)         
# print("\n\n",y_train)
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)


a=0
root = Tk()
root.geometry("500x240")
root.title("Hate and offensive speech detector") 

frame = Frame(root)
frame.pack(side= TOP)
def fun():
    # test_data=input("Enter text: ")
    global a
    for widgets in frame.winfo_children():
        c=widgets
    if(a==1):
        c.destroy()
    df=cv.transform([text.get()]).toarray()
    # print(clf.predict(df))
    mylable3=Label(frame,text=clf.predict(df),font=("Arial", 20))
    mylable3.pack()
    a=1

mylable1=Label(frame,text="\nEnter 'Text' to be detected:\n",font=("Arial", 15))
mylable1.pack(anchor="w",ipadx=2)
text=Entry(frame,width=70)
text.pack()
mylable2=Label(frame,text='\n')
mylable2.pack()
mybutton=Button(frame,text="Submit",padx=22,pady=5,command=fun)
mybutton.pack(anchor="center")

root.mainloop()