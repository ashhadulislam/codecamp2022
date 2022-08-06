import streamlit as st
from lib import commons
import plotly.graph_objects as go

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

            df_bkup,df,MRN_list=commons.pre_process(data_file)
            # apply ML
            df=commons.test_model(df)

            df["MRN"]=MRN_list

            df2 = df.drop_duplicates(subset=["MRN"], keep=False)

            percents=[(1,0.9),(0.9,0.8),(0.8,0.7),(0.7,0.6),(0.6,0.5),(0.5,0.4),(0.4,0.3)]
            counts_list=[]
            print("calc count")
            for percent_range in percents:
                counts=commons.count_more_than(df2,percent_range)
                counts_list.append(counts)
            print(counts_list)

            perc_range=['>90%', '80-90%', '70-80%','60-70%','50-60%','40-50%','30-40%']

            fig = go.Figure([go.Bar(x=perc_range, y=counts_list)])
            st.plotly_chart(fig, use_container_width=True)


            # give details about patients with high percentage
            percentage_val=(1,0.6)
            dic_nry=commons.get_details(df2,df_bkup,percentage_val)

            print(dic_nry)

            num_els=len(dic_nry["MRN"])
            dic_count=0
            st.subheader("Patients with high chances of returning")

            for i in range(num_els):
                for k,v in dic_nry.items():
                    print(k,v[i])
                    st.text(k+":"+str(v[i]))
                    # st.text(v[i])
                st.markdown("""---""")







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