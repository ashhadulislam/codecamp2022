# https://analyticsindiamag.com/how-to-implement-convolutional-autoencoder-in-pytorch-with-cuda/
# https://stackoverflow.com/questions/59924310/load-custom-data-from-folder-in-dir-pytorch

import numpy as np
import os


import time
import copy

import pandas as pd

from collections import Counter
import pickle

import pandas as pd
import numpy as np



from PIL import Image


def convert_age(df,age_col):
    age_col="AGE"
    ages=[]
    for ag in df[age_col]:

        if " Years" in ag:
    #         print(ag)
            age_num=int((ag.split(" Years")[0]))
    #         print(age_num)
        elif " Weeks" in ag:
    #         print(ag)
            age_num=int((ag.split(" Weeks")[0]))
            age_num=(age_num*7)/365
    #         print(age_num)
        elif " Months" in ag:
    #         print(ag)        
            age_num=int((ag.split(" Months")[0]))
            age_num=age_num/12
    #         print(age_num)
        elif " Days" in ag:
    #         print(ag)        
            age_num=int((ag.split(" Days")[0]))
            age_num=age_num/365
    #         print(age_num)

        elif " Hours" in ag:
            age_num=0
    #         print(ag,age_num)
        else:
            print("Mismatch for ",ag)
        ages.append(age_num)
    df["AGE"]=ages
    return df


def removeBadGenders(df,gender_col):
    '''
    we assume that there are very very few rows
    for genders like Indeterminate and null
    '''
    df2 = df[df['GENDER'].notna()]
    df2 = df2[df2['GENDER']!="Indeterminate"]
    gender_mapper={gender_col:{"Male":0,"Female":1}}
    df2.update(df2[list(gender_mapper)].apply(lambda col: col.map(gender_mapper[col.name])))
    return df2


def create_common_mapping_dict(df,cols):
    mapping_dict={}
    for this_col in cols:
        mapping_dict[this_col]={}
        counter_term=0
        for uniq_val in df[this_col].unique():
            mapping_dict[this_col][uniq_val]=counter_term
            counter_term+=1
    return mapping_dict
            

    

def encodeColumnGeneric(df,this_col):
    col_mapper={this_col:{}}
    counter_term=0
    for val in df[this_col].unique():
        col_mapper[this_col][val]=counter_term
        counter_term+=1
    df.update(df[list(col_mapper)].apply(lambda col: col.map(col_mapper[col.name])))
    return df
    
    
def countDaysBetween(df,start_date_col,end_date_col,durationCol):

    df[start_date_col] = df[start_date_col].map(lambda x: str(x)[:-10])
    df[end_date_col] = df[end_date_col].map(lambda x: str(x)[:-10])

    df[start_date_col] = pd.to_datetime(df[start_date_col], format='%d/%m/%Y')
    df[end_date_col] = pd.to_datetime(df[end_date_col], format='%d/%m/%Y')
    df[durationCol] = (df[end_date_col] - df[start_date_col]).dt.days
    return df

def drg_encoder(list_drg):
    list_drg = list_drg.tolist()
    list_drg = [str(x) for x in list_drg if str(x)!="nan"]
#     print(list_drg)
    if len(list_drg)>1:
#         print("yes")
        pr_drg = list_drg[0]
        add_drg = list_drg[1:]
        add_drg.sort()
        new_drg = "__".join([x for x in add_drg])
        new_drg = str(pr_drg)+"_"+str(new_drg)   
        return new_drg
    else:
        return np.nan

def drg_encoder_counter(list_drg):
    list_drg = list_drg.tolist()
    list_drg = [str(x) for x in list_drg if str(x)!="nan"]
    return len(list_drg)    




def load_image(image_file):
    img = Image.open(image_file)
    return img

def pre_process(data_file):

    print("File is ",data_file)
    if ".xlsx" in data_file.name:
        df=pd.read_excel(data_file)
        print(df.shape)

    df=convert_age(df,"AGE")

    
    # to combine the Diagnosys and processes
    cols_to_combine=['PRINCIPAL_DIAG', 'ADDITIONAL_DIAG1', 'ADDITIONAL_DIAG2',
           'ADDITIONAL_DIAG3', 'ADDITIONAL_DIAG4', 'ADDITIONAL_DIAG5',
           'ADDITIONAL_DIAG6', 'ADDITIONAL_DIAG7']
    df["drg_combine"]=df[cols_to_combine].apply(drg_encoder, axis=1) 
    df["CountDiagnoses"] = df[cols_to_combine].apply(drg_encoder_counter, axis=1)

    cols_to_combine=['PRINCIPAL_PROC', 'ADDITIONAL_PROC1', 'ADDITIONAL_PROC2',]
    df["prc_combine"]=df[cols_to_combine].apply(drg_encoder, axis=1) 
    df["CountProceddures"] = df[cols_to_combine].apply(drg_encoder_counter, axis=1)


    # general mapping to categorical
    cols_to_map=["GENDER","DISCHARGE_FACILITY","MEDICAL_SERVICE","ADMIT_SOURCE","ADMIT_TYPE","ENCOUNTER_TYPE","SAME_DAY_DISCHARGE", 
                 "DISCH_DISPOSITION", "ACCOMMODATION", "LAST_IP_DX_ICD", "SAME_DRG", 
                 "SAME_PRINCIPAL_DIAGNOSIS", "SAME_ATTENDING_PHYSICIAN", "PRINCIPAL_DIAG", "FINAL_DRG_CODE", "PRINCIPAL_PROC",
                "drg_combine","prc_combine"]
    mapp_dic=pickle.load(open("data/saves/mapp_dic.p","rb"))

    df.update(df[list(mapp_dic)].apply(lambda col: col.map(mapp_dic[col.name])))

    

    df=countDaysBetween(df,"ADMIT_DATE","DISCH_DT_TM","DurationStay")


    keep_cols=['MRN', 'ENCOUNTER_NUMBER', 'AGE', 'GENDER', 'DISCHARGE_FACILITY',
       'MEDICAL_SERVICE', 'ADMIT_SOURCE', 'ADMIT_TYPE', 'ENCOUNTER_TYPE',
       'SAME_DAY_DISCHARGE', 'DISCH_DISPOSITION', 'ACCOMMODATION',
       'LAST_IP_DX_ICD', 'SAME_DRG', 'SAME_PRINCIPAL_DIAGNOSIS',
       'SAME_ATTENDING_PHYSICIAN', 'PRINCIPAL_DIAG', 
       'CountDiagnoses', 'PRINCIPAL_PROC', 
         'CountProceddures', 'FINAL_DRG_CODE',
        'DurationStay', 'drg_combine', 'prc_combine',"ADMIT_DATE","DISCH_DT_TM"]

    df=df[keep_cols]        
    df=df.drop(columns=["MRN","ENCOUNTER_NUMBER","ADMIT_DATE","DISCH_DT_TM"])

        




    return df


def test_model(df):

    clf=pickle.load(open("models/RFC.p","rb")) 
    # considering last column is labe;
    X=df.iloc[:,:].values
    # y=df.iloc[:,-1].values

    # change lAter
    y_hat=clf.predict(X)
    # con=confusion_matrix(y,y_hat)
    # print(con)

    # acc=(con[0][0]+con[1][1])/np.sum(con)
    # print(acc)
    print(y_hat)

    df["predictions"]=y_hat

    return df







label_map={
    0:"Chickenpox",
    1:"Measles",
    2:"Monkeypox",
    3:"Normal"
}
classes = ('Chickenpox', 'Measles', 'Monkeypox', 'Normal')
PATH = 'models/resnet18_net.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                     transforms.Resize((64,64)),
                                     transforms.ToTensor()])





def load_model():
    '''

    load a model 
    by default it is resnet 18 for now
    '''

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes))
    model.to(device)

    model.load_state_dict(torch.load(PATH,map_location=device))
    model.eval()
    return model





def image_loader(image_name):
    """load image, returns cuda tensor"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    picture = Image.open(image_name)
    image = data_transform(picture)
    images=image.reshape(1,1,64,64)
    new_images = images.repeat(1, 3, 1, 1)
    return new_images
    
def predict(model, image_name):
    '''

    pass the model and image url to the function
    Returns: a list of pox types with decreasing probability
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    picture = Image.open(image_name)
    image = data_transform(picture)
    images=image.reshape(1,1,64,64)
    new_images = images.repeat(1, 3, 1, 1)

    outputs=model(new_images)

    _, predicted = torch.max(outputs, 1)
    ranked_labels=torch.argsort(outputs,1)[0]
    probable_classes=[]
    for label in ranked_labels:
        probable_classes.append(classes[label.numpy()])
    probable_classes.reverse()
    return probable_classes



