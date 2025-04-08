

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns          #Visulization Library
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,recall_score,precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import streamlit as st
#import warnings
#warnings.filterwarnings('ignore')




df=pd.read_csv("PCOS_data_without_infertility.csv")





st.header("pcos detection ")







html_code = '''
<h1 style="color: blue;">PCOS Detection</h1>
<p>Polycystic Ovary Syndrome (PCOS) is a complex hormonal disorder affecting individuals assigned female at birth, with manifestations that extend beyond reproductive health. It is characterized by hormonal imbalances, leading to irregular menstrual cycles and the presence of small cysts on the ovaries. Alongside irregular periods, it often presents with symptoms such as excess androgen levels, causing acne, hirsutism, and hair thinning.</p>


'''

st.markdown(html_code, unsafe_allow_html=True)





df.head(10)




df.isnull().sum()




df['BMI'].value_counts() #Here We Can see That an null value is defined as #NAME? so we need to replace it with np.nan




df["BMI"].replace("#NAME?", np.NaN, inplace=True)





df.info()




df['BMI']=pd.to_numeric(df["BMI"])


# In[70]:


df['BMI'] = df['BMI'].fillna(df['BMI'].mean())


# In[33]:


#df.drop("Unnamed: 39",axis=1,inplace=True)


# In[71]:


df.dropna(axis=0,inplace=True)


# In[72]:


#fig, ax = plt.subplots(figsize = (42, 32))
#sns.heatmap(df.corr(), annot=True, fmt='1.2f', annot_kws={'size' : 10}, linewidth=1, cmap="coolwarm")
#plt.show()


# In[73]:


#sns.displot(df,x=df['Endometrium (mm)'],kind="kde",hue="PCOS (Y/N)")
#plt.plot()


# In[74]:


#sns.displot(df,x=df['RBS(mg/dl)'],kind="kde",hue="PCOS (Y/N)")
#plt.plot()        


# In[75]:


#sns.displot(df,x=df['PRL(ng/mL)'],kind="kde",hue="PCOS (Y/N)")
#plt.plot()          


# In[76]:


#sns.displot(df,x=df['Follicle No. (L)'],kind="kde",hue="PCOS (Y/N)")
#plt.plot() 


# In[77]:


train, test = train_test_split(df, test_size = 0.3, random_state =1)


# In[78]:


train.shape, test.shape


# In[79]:


x_train = train.drop(['PCOS (Y/N)'], axis=1)


# In[80]:


x_train.shape


# In[81]:


y_train=train['PCOS (Y/N)']


# In[82]:


y_train.shape


# In[83]:


x_test = test.drop(['PCOS (Y/N)'], axis=1)
y_test = test['PCOS (Y/N)']


# In[84]:


ss = StandardScaler()    
x_scaled = ss.fit_transform(x_train)  
x_train_scaled = pd.DataFrame(x_scaled, columns = x_train.columns)   
x_test_scaled = ss.transform(x_test)
x_test_scaled = pd.DataFrame(x_test_scaled, columns = x_test.columns)


# In[93]:


dt = DecisionTreeClassifier()
dt.fit(x_train_scaled, y_train)
DT_pred = dt.predict(x_test_scaled)
print(DT_pred)
DT_acc = accuracy_score(y_test, DT_pred)
print(DT_acc)


# In[88]:



    
    
RF = RandomForestClassifier(max_depth=5, random_state=0,bootstrap=True)
RF.fit(x_train_scaled, y_train)
RF_pred = RF.predict(x_test_scaled)
#print(pred)
RF_acc = accuracy_score(y_test, RF_pred)
RF_acc=(RF_acc*100)
print(RF_acc)
   


# In[89]:



    
    
nb = GaussianNB()
nb.fit(x_train_scaled, y_train)
nb_pred = nb.predict(x_test_scaled)
#print(pred)
nb_acc = accuracy_score(y_test, nb_pred)
nb_acc=(nb_acc*100)
print(nb_acc)


# In[90]:


LR = LogisticRegression(random_state=0)
LR.fit(x_train, y_train)
LR_pred = LR.predict(x_test)
#print(LR_pred)
LR_acc = accuracy_score(y_test, LR_pred)
print(LR_acc)


# In[95]:


confusion_matrix(y_test,DT_pred)  #ratio of tp's, tn's ,fn's,fp's


# In[96]:


linscores = cross_validate(RF, x_train, y_train, scoring="accuracy", cv =5, return_estimator=True)
print(linscores['test_score'])   
#No Overfitting observed in the model as the crossvalidation scores are mostly equivalent to the accuracy score






a=st.number_input("Enter Age")
b=st.number_input("Enter Weight in Kgs")
c=st.number_input("Enter No of years of marriage (if not married Enter 0)")
d=st.number_input("If Weight Gain Enter 1 or 0")
e=st.number_input("If pimples present enter 1 or 0")
f=st.number_input("Enter Level of Folicile Stimulating Hormone(FSH)")
g=st.number_input("Specify Cycle if regular enter:2 or irregular:4")

filled_inputs = sum(1 for i in [a,b,c,d,e,f,g] if i)
threshold=7
if filled_inputs >= threshold:

    ip4=np.array([[a,b,c,d,e,f,g,19.24399261,11.16003697,15,5,73.24768946,0,0,664.5492348,238.2329926,8.475914972,6.469918669,37.99260628,33.84103512,2.981280961,5.513957447,24.32149723,49.91587431,0.610944547,99.83585952,23.99829787,0,0,0,156.4848355,1,0,114.6617375,76.92791128,6,5,15,18]])
    IP_pred = RF.predict(ip4)
    if IP_pred==1:
       if 15<=a<=30 and 45<=b<=70:
        st.write("Excessive Weight")
        st.write("Diet Chart")
        st.write("PCOS Detected")
        #st.write("PCOS DETECTED")
    elif IP_pred==0:
       st.write("Chances of PCOS Pls control ur weight")
    

    
    
    





#print(IP_pred)
#if IP_pred==1:
    #print("PCOS Detected")
    #st.write("PCOS DETECTED")


#else:
    #print("PCOS Not Detected")
    #st.write("PCOS NOT DETECTED")







    
    
    







