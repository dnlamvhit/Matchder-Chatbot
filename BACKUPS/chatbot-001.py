import streamlit as st
import pandas as pd
import random
import time
import openai
#import textwrap3 as textwrap
import dotenv
from dotenv import load_dotenv
import numpy as np
import os
from pinecone import Pinecone
import csv
# For sending email
import json
# import sendgrid
# from sendgrid import SendGridAPIClient
# from sendgrid.helpers.mail import Mail
# from joblib import load
# For getting google sheet content
# import requests
# from bs4 import BeautifulSoup
# For update knowledge base
# import Update_KB

delimiter = ""
initial_context=[
            {'role': 'system', 'content': f"""\
You will answer questions about California Science and Technology University (CSTU) based on contents provided at system role. At fisrt, welcome to Matchder chatbot.\
If user want to register courses, ask his name and email. If he provide name and email, call offer_courses function. If he select courses, ask for his registration confirmation. If he agree, complete registration.\
If user want to get registration record, ask their email. If they provide email, call get_registration function.\
If user want to get course grades, ask their email. If they provide email, call get_grades function.\
If user want to update course grades, call updade_grades function.\
If you don't have data to answer question related to CSTU, ask user to check the catalog file cstu.edu/pages/tmpl/Catalog.pdf, website www.cstu.edu or contact email admission@cstu.edu and tel 408-400-3948."""} ]

#st.sidebar.image("CSTU.png", use_column_width=True)
#st.sidebar.image("robo.gif", use_column_width=True)
st.sidebar.markdown("<font color='darkblue'><b><p style='text-align: center; line-height: 0;'>MATCHDER CHATBOT</p></b></font>", unsafe_allow_html=True)
st.sidebar.markdown("<font color='darkblue'><b><p style='text-align: center; line-height: 0;'>Version 1.0 - 09/2024</p></b></font>", unsafe_allow_html=True)
st.sidebar.markdown("""
<style>
.stButton > button {
  display: block;
  margin-left: auto;
  margin-right: auto;
  color: darkblue;
  font-weight: bold;
}
</style>
""", unsafe_allow_html=True)
if st.sidebar.button("CLEAR CHAT HISTORY"):
    st.session_state.chat_history = []
    st.session_state.prompt_history = initial_context
# Set the text color and alignment for the selectbox label and options
st.sidebar.markdown("""
<style>
.stSelectbox > label,
.stSelectbox > select {
  color: darkblue;
  font-weight: bold;
}
</style>
""", unsafe_allow_html=True)
st.sidebar.markdown("<font color='darkblue'><b><p style='text-align: center; line-height: 0;'>KNOWLEDGE-BASE SETUP</p></b></font>", unsafe_allow_html=True)
n_KB = int(st.sidebar.selectbox("Number of KB retrieved records/query:", ["1", "2", "3", "4", "5"]))
# Create a menu
if 'index' not in st.session_state:
    st.session_state['index'] = None
st.sidebar.markdown("<font color='darkblue'><b><p style='text-align: center; line-height: 0;'>DATABASE MANAGEMENT</p></b></font>", unsafe_allow_html=True)
options = st.sidebar.selectbox(
    'Choose a task option:',
    ('UPDATE KNOWLEDGE-BASE', 'UPLOAD 2', 'UPLOAD 3'),
    key='menu',
    index = st.session_state['index']
)
if options is not None:
    st.session_state['index'] = ('UPDATE KNOWLEDGE-BASE', 'UPLOAD 2', 'UPLOAD 3').index(options)
    expander=st.sidebar
    if 'login_status' not in st.session_state:
        st.session_state['login_status'] = False
    st.session_state['key'] = 0
    with expander:
        username = st.text_input("Enter your username:", key=st.session_state['key'])
        password = st.text_input("Enter your password:", type="password", key=st.session_state['key']+1)
        #upload_window = st.empty()
        if st.button("LOGIN"):
            for user in st.secrets["admin_accounts"].get("users", []):
                if username == user.get("username") and password == user.get("password"):
                    st.session_state['login_status'] = True
                    st.balloons()
                    break
            if st.session_state['login_status'] == False:
                st.warning("Invalid username or password. Please try again!")
        if st.button("LOGOUT"):
             st.session_state['login_status'] = False
             st.session_state['key'] += 2
             st.session_state['index'] = None
             st.rerun()
        if st.session_state['login_status']:
            if options == 'UPDATE KNOWLEDGE-BASE':
               file_name = st.file_uploader("Select your pdf catalog file", type="pdf")
               if file_name is not None:
                   st_button=st.empty()
                   if st_button.button("UPLOAD"):
                      st_button.empty()
                      try:
                            result=update_kb_openai()
                            st.balloons()
                            st.success("Knowledge-base records updated:\n"+result)
                      except Exception as e:
                            st.write(e)

embedding_model = "text-embedding-ada-002"

try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    SENDGRID_API_KEY = st.secrets["SENDGRID_API_KEY"]
except Exception as e:
    # Secrets not found in Streamlit, try loading from local .env file
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
    if not OPENAI_API_KEY or not PINECONE_API_KEY or not SENDGRID_API_KEY:
        st.error("Environment file error or secrets not found!")
        st.error(e)

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

index_name = 'cstugpt-kb'
pc = Pinecone( # initialize connection to pinecone
    api_key=PINECONE_API_KEY,
    environment="us-west1-gcp-free")
try:
    pincone_index = pc.Index(index_name) # connect to pinecone index
except PineconeConnectionError:
    print("Could not connect to Pinecone")
    pincone_index = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "prompt_history" not in st.session_state: # Initialize the chat history with the system message if it doesn't exist
        st.session_state.prompt_history = initial_context
def chat_complete_messages(messages, temperature=0):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
        functions = [
         {
            "name": "offer_courses",
            "description": "Display a list of available courses for registration",
         },
         {
            "name": "registration",
            "description": "complete registration",
            "parameters": {
                "type": "object",
                "properties": {
                    "student_name": {"type":"string","description":"The name of the user",},
                    "student_email": {"type": "string", "description": "The email of user",},
                    "courses":{"type":"string", "description":"The courses the user want to register",},
                    "body": {"type": "string", "description": "Confirmation content of CSTU about courses registered by user",},
                },
                "required": ["student_name", "student_email", "courses","body"],
            }
         },
        {
            "name": "get_registration",
            "description": "get registration record",
            "parameters": {
                "type": "object",
                "properties": {
                    "student_email": {"type": "string", "description": "The email of the student user",}
                },
                "required": ["student_email"],
            }
         },
        {
            "name": "update_grades",
            "description": "update course grades",
            "parameters": {
                "type": "object",
                "properties": {
                    "prof_name": {"type": "string", "description": "Professor name",}
                },
                #"required": ["prof_name"],
            }
         },
        {
            "name": "get_grades",
            "description": "get course grades",
            "parameters": {
                "type": "object",
                "properties": {
                    "student_email": {"type": "string", "description": "The email of the student user",}
                },
                "required": ["student_email"],
            }
         },
        ],
       function_call="auto",
    )
    return response.choices[0]["message"]

def offer_courses():
    url = "https://docs.google.com/spreadsheets/d/1R3vmk-TKWYtTAjEkgsIJRpPdWY01PL9-ibyiHEcourk/edit#gid=1433325642?sheet=2024%20Planning"
    # Get the HTML content
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    # Find the table containing the sheet data (adjust selectors as needed)
    table = soup.find("table", class_="waffle")
    # Extract data from table cells
    courses = "Here are available courses:"
    for row in table.find_all("tr"):
      # Check if there are any cells in the row
      if len(row.find_all("td")) > 0:
        # Get the first cell's text
        cell_text = row.find_all("td")[0].text.strip()
        # Remove leading/trailing spaces and duplicates
        cell_text = " ".join(cell_text.split())  # Replace multiple spaces with single space
        # Check if cell starts with "CSE" or "MB" (modify as needed)
        if cell_text.startswith(("CSE", "MB")) and cell_text not in courses:
          index = courses.count("\n") + 1
          courses = courses + "\n" + str(index) + ". " + cell_text
    ai_message = {"role": "assistant", "content": courses +"\n\n" + "Select courses you wish to enroll in."}
    st.session_state.chat_history.append(ai_message)
    st.session_state.prompt_history.append(ai_message)
def registration(student_name,student_email,courses,body):
    try:
        csv_file = "registration_records.csv"
        data = [time.strftime("%Y-%m-%d %H:%M:%S"), student_name, student_email, courses]
        if not os.path.exists(csv_file):
            with open(csv_file, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["REGISTRATION TIME","STUDENT NAME", "EMAIL ADDRESS", "COURSE NAME"])
        with open(csv_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(data)
        message = Mail(
            from_email='cstu02@gmail.com',
            to_emails=student_email,
            subject='Course registration confirmation from CSTU',
            html_content=body)
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        result = "Thank you for your registration. A confirmation message has been sent to your email " + function_args.get("student_email")
    except Exception as e:
        print(e.message)
        result = "Registration failed due to system error. Please try again letter."
    ai_message = {"role": "assistant", "content": result}
    st.session_state.chat_history.append(ai_message)
    st.session_state.prompt_history.append(ai_message)


# Generate an embedding for a text using CSTU embedding model
def generate_embedding(model, text):
    tokens = nltk.word_tokenize(text)
    word_vectors = [model.wv[token] for token in tokens if token in model.wv]
    if not word_vectors: # If no valid word vectors are found, return a vector of zeros
        return np.zeros(model.vector_size)
    embedding = np.mean(word_vectors, axis=0)
    return embedding

# Accept user input
if user_input := st.chat_input("Enter a prompt here to ask me for information"):
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    if OPENAI_API_KEY:
        # Word2Vector embedding
        # input_emb=generate_embedding(embedding_model, user_input)
        # OpenAI embedding
        res = openai.Embedding.create(input=[user_input],engine=embedding_model)
        input_emb=res['data'][0]['embedding']
        kb_res = pincone_index.query(vector=input_emb, top_k=n_KB, include_metadata=True, namespace='cstu', metric="cosine")
        #If the include_metadata parameter is set to True, the query method will only return the id, score, and metadata for each document. The vector for each document will not be returned
        metadata_text_list = [x['metadata']['text'] for x in kb_res['matches']]
        limit = 3600  #set the limit of knowledge base words
        kb_content = " "
        count = 0
        proceed = True
        while proceed and count < len(metadata_text_list):  # append until hitting limit
            if len(kb_content) + len(metadata_text_list[count]) >= limit:
                proceed = False
            else:
                    kb_content += metadata_text_list[count]
            count += 1

        # Add knowledge base and user message to promt history
        st.session_state.prompt_history.append({"role": "system", "content": f"{delimiter}{kb_content}{delimiter}"})
        st.session_state.prompt_history.append({"role": "user", "content": user_input})

        # Get the model response
        response = chat_complete_messages(st.session_state.prompt_history, temperature=0)


# Display chat messages
for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], list):
                st.write("Here is your results:")
                df = pd.DataFrame(message["content"])
                st.dataframe(df)
            else:
                st.markdown(message["content"])
