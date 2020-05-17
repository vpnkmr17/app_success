#Importing the libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
#from app.models import AppSuccess
import pickle
import os


def feature_engineering(googlePS):
    googlePS.Category.replace(['1.9'],'noCatagory',inplace=True) #replace '1.9' in 'Category' column by 'noCatagory'
    googlePS.Reviews.replace(['3.0M'],3000000.0,inplace=True)    #replace '3.0M' in 'reviews' column by '3000000.0'
    googlePS.Reviews=googlePS.Reviews.astype(float)
    category_list=list(googlePS.Category.unique())
    category_list
    review_list=[]

    for i in category_list:
        filter_for_catagory=googlePS.Category==i
        filtered=googlePS[filter_for_catagory]
        sum_reviewsfor_catagory=sum(filtered.Reviews)/1000000   #million
        review_list.append(sum_reviewsfor_catagory)
    
    
    #Data Visualisation
    # dataFrame=pd.DataFrame({'review_list':review_list,'category_list':category_list})


    # index=(dataFrame['review_list'].sort_values(ascending=False)).index.values
    # index
    # sorted_data=dataFrame.reindex(index)


    # plt.figure(figsize=(80,25))
    # sns.barplot(x=sorted_data['category_list'][:5],y=sorted_data['review_list'][:5])  #first 5 
    # plt.rc('font', size=15)  

    # plt.xlabel('Categories',fontsize=80)
    # plt.ylabel('Reviews / Million',fontsize=80)

    thaLastIndex=[]
    for i in googlePS['Size']:
        if i[-1] !='M' and i[-1] !='k' :
            thaLastIndex.append(i)
            
    Counter(thaLastIndex)

    cleared_data_for_size=[]

    for i in googlePS['Size']:
        if i[-1]=='k':
            cleared_data_for_size.append(float(i[:-1])*1000)
        elif i[-1]=='M':
            cleared_data_for_size.append(float(i[:-1])*1000000)
        else:
            cleared_data_for_size.append(0.0)   # the values will be replaced by the avarage value of the Size 

    #Counter(cleared_data_for_size)
    googlePS.Size=pd.Series(cleared_data_for_size)
    googlePS.Size.replace(0.0,googlePS.Size.mean(),inplace=True)
    googlePS.Size.head()

    thaLastIndex_for_Installs=[]
    for i in googlePS['Installs']:
        if i[-1] !='+':
            thaLastIndex_for_Installs.append(i)
            
    Counter(thaLastIndex_for_Installs)

    googlePS.Installs.replace('0','0+',inplace=True)
    googlePS.Installs.replace('Free','0+',inplace=True)

    #the Install values has ',' which prevent us to convert to float, so we need to remove these
    replaced_for_Installs=[]
    [replaced_for_Installs.append(float(x[:-1].replace(',' , '')))for x in googlePS.Installs]

    googlePS.Installs=pd.Series(replaced_for_Installs)
    googlePS.Installs.head()

    data_for_Installs=[]
    data_for_Sizes=[]
    for i in googlePS.Category.unique():
        filter_for_cat=googlePS.Category==i
        filtered=googlePS[filter_for_cat]
        data_for_Installs.append(sum(filtered.Installs))
        data_for_Sizes.append(sum(filtered.Size))
        
        
    data_for_Installs=pd.Series(data_for_Installs)
    data_for_Sizes=pd.Series(data_for_Sizes)
    categories=googlePS.Category.unique()
    googlePS=googlePS[googlePS['Content Rating'].isnull()!=True]

    return googlePS

#Preprocessing the data
def preprocessing(googlePS,flag):
    top_10_category=['ART_AND_DESIGN','AUTO_AND_VEHICLES','BEAUTY','COMICS','COMMUNICATION','EDUCATION','ENTERTAINMENT','FOOD_AND_DRINK','HOUSE_AND_HOME','LIBRARIES_AND_DEMO']
    def one_hot_top(df, variable,top_X_labels):
        for label in top_X_labels:
            df[variable+'_'+label]=np.where(df[variable]==label,1,0)
    one_hot_top(googlePS,'Category',top_10_category)
    googlePS = googlePS.drop(googlePS[googlePS['Rating'].isnull()].index, axis=0)
    googlePS["Type"] = (googlePS["Type"] == "Paid").astype(int)
    popApps = googlePS.copy()
    popApps = popApps.drop_duplicates()
    popApps["Price"] = popApps["Price"].str.replace("$","")
    popApps[popApps['Price']=='Everyone']=0
    popApps["Price"] = popApps["Price"].astype("float64")
    popApps["Size"] = popApps["Size"].astype("int64")
    popApps["Reviews"] = popApps["Reviews"].astype("int64")

    if flag==True:
        popApps = popApps.sort_values(by="Installs",ascending=False)
        popApps.reset_index(inplace=True)
        popApps.drop(["index"],axis=1,inplace=True)
        popApps["Installs"] = popApps["Installs"].astype("int64")
    popAppsCopy = popApps.copy()
    #category=pd.get_dummies(popAppsCopy['Category'])
    content=pd.get_dummies(popAppsCopy['Content Rating'])
    #genre=pd.get_dummies(popAppsCopy['Genres'])
    
    popAppsCopy=pd.concat([popApps,content],axis=1)
    print("hiii................")
    popAppsCopy = popAppsCopy.drop(["App","Last Updated","Current Ver","Android Ver"],axis=1)
    print("Byeeee.........")
    popAppsCopy = popAppsCopy.drop(["Category","Content Rating","Genres"],axis=1)
    
    return popAppsCopy

def training(googlePS):
    popAppsCopy=preprocessing(googlePS,True)
    popAppsCopy["Installs"] = (popAppsCopy["Installs"] > 100000)*1 #Covert the values into binary format
    popAppsCopy=popAppsCopy.rename(columns={"Installs": "success"})
    train = popAppsCopy.sample(frac=1).reset_index(drop=True)
    y = train["success"]
    X = train.drop(["success"],axis=1)
    X = X.loc[:,~X.columns.duplicated()]

    col=X.columns
    print(len(col))

    #dummyRow=pd.DataFrame(np.zeros(len(X.columns)).reshape(1,len(X.columns)),columns=X.columns)
    #dummyRow.to_csv('dummyRow.csv')
    # Fitting Decision Tree Classification to the Training set
    #from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    #from xgboost import XGBClassifier
    classifier = RandomForestClassifier(n_estimators=100,max_depth=10)
    #classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0,max_depth=5)
    #classifier = XGBClassifier()
    classifier.fit(X, y)
    
    pkl_filename="model.pkl"
    with open(pkl_filename,"wb") as file:
        pickle.dump(classifier,file)
    #Applying k-Fold Cross Validation
    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator = classifier, X = X, y = y, cv = 10)
    print("Accuracy score is ",accuracies.mean())
    # accuracies.std()

def predict(ob):
    d1=ob.to_dict()
    df=pd.DataFrame(d1,index=[0])
    print('Dataframe is  ',df)
    df=preprocessing(df,False)
    print("DF columns after preprocessing ",df.columns)
    dummyrow_filename="dummyRow.csv"
    dummyrow_filename=os.path.join(os.path.abspath(os.path.dirname(__file__)),dummyrow_filename)
    df2=pd.read_csv(dummyrow_filename)
    print(" precious DF2 values is ",df2)
    for c in df.columns:
        df2[c]=df[c]
    print("#############")
    print("#############")
    print("DF2 value is    ",df2)
    print("DF2 columns is ",df2.columns)
    pkl_filename="model.pkl"
    pkl_filename=os.path.join(os.path.abspath(os.path.dirname(__file__)),pkl_filename)
    with open(pkl_filename,'rb') as file:
        model=pickle.load(file)
    pred=model.predict(df2)
    probability=model.predict_proba(df2)
    print("predicted values is ",pred)
    return pred,probability


#Reading csv file
# google_data=pd.read_csv("googleplaystore.csv")
# googlePS=feature_engineering(google_data)
# training(googlePS)
