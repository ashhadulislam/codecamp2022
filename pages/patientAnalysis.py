import streamlit as st
from lib import commons

@st.cache
def convert_df(df):
   return df.to_csv().encode('utf-8')






def app():
    header=st.container()
    result_all = st.container()
    # model=commons.load_model()
    with header:
        st.subheader("Test whether Patients will come back in a month")
        data_file = st.file_uploader("Upload File", type=["csv","xls","xlsx"])



        if data_file is not None:
            # To See details
            file_details = {"filename":data_file.name, "filetype":data_file.type,
                          "filesize":data_file.size}
            st.write(file_details)

            df=commons.pre_process(data_file)
            # apply ML
            df=commons.test_model(df)

            csv = convert_df(df)

            st.download_button(
               "Press to Download",
               csv,
               "file.csv",
               "text/csv",
               key='download-csv'
            )


            # # To View Uploaded Image
            # st.image(commons.load_image(image_file)
            #     ,width=250
            #     )
            # print("Image file is it showing location?",image_file)            
            # predictions=commons.predict(model,image_file)
            print("Loaded image for model")
        else:
            # proxy_img_file="data/chicken00.jpg"
            # st.image(commons.load_image(proxy_img_file),width=250)        
            # predictions=commons.predict(model,proxy_img_file)
            print("Loaded proxy image for model")

    with result_all:                        
        i=1
        st.subheader("Result here:")
        # for pred in predictions:
        #     st.text(str(i)+". "+pred)    
        #     i+=1