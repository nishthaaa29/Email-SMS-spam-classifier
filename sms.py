import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import sklearn
try:
    # Streamlit < 0.65
    from streamlit.ReportThread import get_report_ctx

except ModuleNotFoundError:
    try:
        # Streamlit > 0.65
        from streamlit.report_thread import get_report_ctx

    except ModuleNotFoundError:
        try:
            # Streamlit > ~1.3
            from streamlit.script_run_context import get_script_run_ctx as get_report_ctx

        except ModuleNotFoundError:
            try:
                # Streamlit > ~1.8
                from streamlit.scriptrunner.script_run_context import get_script_run_ctx as get_report_ctx

            except ModuleNotFoundError:
                # Streamlit > ~1.12
                from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx as get_report_ctx

                try:
                    # Streamlit < 0.65
                    from streamlit.ReportThread import get_report_ctx

                except ModuleNotFoundError:
                    try:
                        # Streamlit > 0.65
                        from streamlit.report_thread import get_report_ctx

                    except ModuleNotFoundError:
                        try:
                            # Streamlit > ~1.3
                            from streamlit.script_run_context import get_script_run_ctx as get_report_ctx

                        except ModuleNotFoundError:
                            try:
                                # Streamlit > ~1.8
                                from streamlit.scriptrunner.script_run_context import \
                                    get_script_run_ctx as get_report_ctx

                            except ModuleNotFoundError:
                                # Streamlit > ~1.12
                                from streamlit.runtime.scriptrunner.script_run_context import \
                                    get_script_run_ctx as get_report_ctx
ps=PorterStemmer()


tf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title('Email/SMS Spam Classifier')
input_sms = st.text_input("Enter the Message")
if st.button('Predict'):
    # preprocessing
    def transformed_text(text):
        text = nltk.word_tokenize(text)
        y = []
        for i in text:
            if i.isalnum():
                y.append(i)

        text = y[:]
        y.clear()

        for i in text:
            if i not in nltk.corpus.stopwords.words('english') and i not in string.punctuation:
                y.append(i)

        text = y[:]
        y.clear()

        for i in text:
            y.append(ps.stem(i))

        return " ".join(y)



    transformed_sms = transformed_text(input_sms)
    # vectorization
    vector_input = tf.transform([transformed_sms])

    # predict
    result = model.predict(vector_input)[0]

    # display
    if result == 1:
        st.header("Spam")
    elif result == 0:
        st.header('Not Spam')









