import streamlit as st 
import pickle

from utils import db_connect
engine = db_connect()

# your code here
cv = pickle.load(open("/workspaces/machine-learning-python-template-ds-Julio/models/cv.pkl", 'rb'))
clf = pickle.load(open("/workspaces/machine-learning-python-template-ds-Julio/models/clf.pkl", 'rb'))

def main():
    st.title("Email Spam Class")

    st.markdown(""" 
        <style>
            .column{
                padding:10px;
                display: inline;
                text-align: center;
                float: left;
                height: auto;
            }
            .center{
                text_align: center;
            }
            .border{
                border_right: solid green;
            }
        </style>
        """, unsafe_allow_html=True)
    
    col1,col2=st.columns([2,3])

    with col1:
        st.markdown("### Sample Emails")
        st.markdown("<h4 style = 'color: blue;'>SPAM</h4>", unsafe_allow_html=True)
        st.markdown("Your spam sample text here..")
        st.markdown("<h4 style = 'color: red;'> NOT SPAM</h4>", unsafe_allow_html=True)
        st.markdown("your not spam sample text here..")
    
    with col2:
        email=st.text_area("type or paste your email here",height=300)
        if st.button("Check"):
            x=cv.transform([email])
            prediction=clf.predict(x)
            prediction=1 if prediction == 1 else -1

            if prediction==1:
                st.error("Spam")
            else:
                st.success("not spam")
if __name__=='__main__':
    main()