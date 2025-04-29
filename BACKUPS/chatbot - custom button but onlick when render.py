import streamlit as st
import pandas as pd
import random
import time
from openai import OpenAI
#import textwrap3 as textwrap
import dotenv
from dotenv import load_dotenv
import numpy as np
import os
from pinecone import Pinecone
import csv
import json
from supabase import create_client
# For sending email
# import sendgrid
# from sendgrid import SendGridAPIClient
# from sendgrid.helpers.mail import Mail
# from joblib import load
# For getting google sheet content
# import requests
# from bs4 import BeautifulSoup
# For update knowledge base
# import Update_KB

embedding_model = "text-embedding-ada-002"

try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    SENDGRID_API_KEY = st.secrets["SENDGRID_API_KEY"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except Exception as e:
    # Secrets not found in Streamlit, try loading from local .env file
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    if not OPENAI_API_KEY or not PINECONE_API_KEY or not SENDGRID_API_KEY or not SUPABASE_KEY:
        st.error("Environment file error or secrets not found!")
        st.error(e)
# Set OpenAI API key
client = OpenAI(api_key = OPENAI_API_KEY)

index_name = 'cstugpt-kb'
pc = Pinecone( # initialize connection to pinecone
    api_key=PINECONE_API_KEY,
    environment="us-west1-gcp-free")
try:
    pincone_index = pc.Index(index_name) # connect to pinecone index
except PineconeConnectionError:
    print("Could not connect to Pinecone")
    pincone_index = None

supabase = create_client("https://ziaisnybjhkjcbprxava.supabase.co", SUPABASE_KEY)
categories_list = []
def fetch_categories():
    """Fetches category names from the public.categories table and stores them in the global categories_list."""
    global categories_list
    try:
        # Query the categories table to fetch category names
        response = supabase.table("categories").select("name").execute()
        if response.data:
            # Extract category names from the query result
            categories_list = ["All"] + [category['name'] for category in response.data]
        else:
            categories_list = ["No categories found"]
    except Exception as e:
        st.error(f"Error fetching categories: {e}")
        categories_list = ["Error fetching categories"]

# Call the function to fetch categories when the app starts
fetch_categories()

# Function to update filters
def update_filters(function_args):
    """Parses user input and creates filters."""
    #allow_inspect_value = function_args.get("allow_inspect")
    #allow_inspect_value = "Yes" if allow_inspect_value is True else "No" if allow_inspect_value is False else None
    st.session_state.deal_type = function_args.get("deal_type", "All")
    st.session_state.product_name = function_args.get("product_name", "")
    st.session_state.address = function_args.get("address", "")
    st.session_state.num_days = function_args.get("num_days", 365)
    st.session_state.condition = function_args.get("condition", "All")
    st.session_state.status = function_args.get("status", "All")
    st.session_state.allow_inspect = function_args.get("allow_inspect", "All")  
    st.session_state.delivery_method = function_args.get("delivery_method", "All")
    st.session_state.min_price = function_args.get("min_price", 0.0)
    st.session_state.max_price = function_args.get("max_price", 2000000000.0)
    st.session_state.category_name = function_args.get("category_name", "All")
    st.session_state.product_description = function_args.get("product_description", "")

    filters = {
        "_category_name": function_args.get("category_name", None),
        "_product_description": function_args.get("product_description", None),
        "_deal_type": function_args.get("deal_type", None),
        "_min_price": function_args.get("min_price", None),
        "_max_price": function_args.get("max_price", None),
        "_condition": function_args.get("condition", None),
        "_status": function_args.get("status", None),
        "_product_name": function_args.get("product_name", None),
        "_allow_inspect": function_args.get("allow_inspect", None),
        "_delivery_method": function_args.get("delivery_method", None),
        "_address": function_args.get("address", None),
        "_num_days": function_args.get("num_days", None),
    }
    return filters

# Function to fetch products
def fetch_product(filters):
    columns_to_display = ["name", "description", "deal_type", "price", "published_at", "condition",
                          "status", "rating", "allow_inspect", "delivery_method", "address", "quantity", "product_image_urls"]
    try:
        product_data, count = supabase.rpc("filter_products", filters).execute()
        if product_data[1]:
            # Construct the product display table
            for item in product_data[1]:
                if item.get("type") == "Buyer":
                    min_budget = item.get("min_budget")
                    max_budget = item.get("max_budget")
                    item["price"] = f"${min_budget} - ${max_budget}" if min_budget and max_budget else "Not available"
                elif item.get("type") == "Seller":
                    item["price"] = f"${price}" if price is not None else "Not available"

                # Format dates and links
                if "published_at" in item and item["published_at"]:
                    try:
                        published_at_date = datetime.fromisoformat(item["published_at"])
                        now_utc = datetime.now(timezone.utc)
                        days_since_posting = (now_utc - published_at_date).days
                        item["published_at"] = f"{days_since_posting} days ago"
                    except ValueError:
                        item["published_at"] = "Invalid date"

                # Create a link to Google Maps for the address
                latitude = item.get("latitude")
                longitude = item.get("longitude")
                address = item.get("address")
                if latitude and longitude:
                    map_url = f"https://www.google.com/maps?q={latitude},{longitude}"
                    item["address"] = f"[{address}]({map_url})"

                # Display product information
                st.write(f"**Product Name**: {item.get('name', '')}")
                st.write(f"**Description**: {item.get('description', '')}")
                st.write(f"**Deal Type**: {item.get('deal_type', '')}")
                st.write(f"**Price**: {item.get('price', '')}")
                st.write(f"**Condition**: {item.get('condition', '')}")
                st.write(f"**Status**: {item.get('status', '')}")
                st.write(f"**Posted**: {item.get('published_at', '')}")
                st.write(f"**Allow Inspection**: {item.get('allow_inspect', '')}")
                st.write(f"**Delivery Method**: {item.get('delivery_method', '')}")
                st.write(f"**Address**: {item.get('address', '')}")
                st.write("---")

            return "Here are products matching your query"
        else:
            return "No products match your query."
    except (ValueError, TypeError, Exception) as e:
        return f"Error: {str(e)}"

# Initialize history and state for chatbot
delimiter = ""
system_guide = [("system", "You are Matchder marketing assistant...")]
# Initialize default values in session state (this only runs the first time)
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.chat_history = []
    st.session_state.prompt_history = system_guide
    st.session_state.deal_type = "All"
    st.session_state.product_name = ""
    st.session_state.address = ""
    st.session_state.num_days = 365
    st.session_state.condition = "All"
    st.session_state.status = "All"
    st.session_state.allow_inspect = "All"
    st.session_state.delivery_method = "All"
    st.session_state.min_price = 0.0
    st.session_state.max_price = 2000000000.0
    st.session_state.category_name = "All"
    st.session_state.product_description = ""

# STREAMLIT INTERFACE
#st.sidebar.image("CSTU.png", use_column_width=True)
#st.sidebar.image("robo.gif", use_column_width=True)
st.sidebar.markdown("<font color='darkblue'><b><p style='text-align: center; line-height: 0;'>MATCHDER CHATBOT</p></b></font>", unsafe_allow_html=True)

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

st.markdown("""
    <style>
    .resizable-textarea textarea {
        resize: both !important;
        overflow: auto;
    }
    .stButton > button {
        width: 50px;
        height: 50px;
        font-size: 50px;
    }
    </style>
""", unsafe_allow_html=True)
button_style = '''
    <style>
    .mybutton {
        font-size: 25px; /* Adjust font size */
        height: 45px;    /* Adjust button height */
        width: 45px;    /* Adjust button width */
        justify-content: center; /* Center horizontally */
        align-items: center; /* Center vertically */
    }
    </style>'''
 
# Input form
#with st.form(key='input_form'):

with st.container():
  col1, col2 = st.columns([30, 1])

# Filters section inside expander
  with col1:
    with st.expander("SEARCH FOR PRODUCT BASED ON USER-SELECTED FILTERS"):
      st.session_state.deal_type = st.radio("Are you looking for", ["Buyer", "Seller", "All"], 
                                            index=["Buyer", "Seller", "All"].index(st.session_state.deal_type))    
      st.session_state.product_name = st.text_input("Product Name", value=st.session_state.product_name)
      st.session_state.address = st.text_input("Address", value=st.session_state.address)
      st.session_state.num_days = st.number_input("Within number of days since advertised", min_value=0, value=st.session_state.num_days) 
      st.session_state.condition = st.radio("Condition", ["All", "New", "Like New", "Excellent", "Good", "Fair", "Salvage"], 
                                            index=["All", "New", "Like New", "Excellent", "Good", "Fair", "Salvage"].index(st.session_state.condition))
      st.session_state.status = st.radio("Status", ["All", "Available", "Sold out", "Still searching", "Bought"], 
                                        index=["All", "Available", "Sold out", "Still searching", "Bought"].index(st.session_state.status))
      allow_inspect_options = {"All": None, "Yes": True, "No": False}
      allow_inspect_selection = st.radio("Pre-purchase Inspection Allowed", ["All", "Yes", "No"],
                                        index=["All", "Yes", "No"].index("Yes" if st.session_state.allow_inspect is True else "No" if st.session_state.allow_inspect is False else "All"))
      st.session_state.allow_inspect = allow_inspect_options[allow_inspect_selection]  # Store the boolean value
      st.session_state.delivery_method = st.radio("Delivery Method", ["All", "Pickup", "Delivery Driver", "Shipping"], 
                                                  index=["All", "Pickup", "Delivery Driver", "Shipping"].index(st.session_state.delivery_method))
      st.session_state.min_price = st.number_input("Min Price", min_value=0.0, value=st.session_state.min_price, format="%.2f") 
      st.session_state.max_price = st.number_input("Max Price", min_value=0.0, value=st.session_state.max_price, format="%.2f")
      st.session_state.category_name = st.selectbox("Category Name", categories_list, 
                                                    index=categories_list.index(st.session_state.category_name))
      st.session_state.product_description = st.text_input("Product Description", value=st.session_state.product_description)
    user_input = st.text_area("ENTER YOUR QUESTION", placeholder="Type here ...", key='user_input', height=1, help='Resize by dragging the bottom-right corner', max_chars=None,)
  # Product search button
  with col2: 
    button_search = f'''{button_style} <button id="searchButton" class="mybutton" title="Search products">🔍</button>'''   
    if st.markdown(button_search, unsafe_allow_html=True):
    #if st.markdown(button_search, unsafe_allow_html=True):
    #if st.markdown('<button title="Search products">🔍</button>', unsafe_allow_html=True): #Search button
    #if st.markdown("""<button class="mybutton" onclick="window.location.href='#';">🔍</button>""", unsafe_allow_html=True):
          def reverse_map(value):
              """Map 'All' and '' to None, 'Yes' to True, 'No' to False."""
              if value in ["All", "", 0]:
                  return None
              elif value == "Yes":
                  return True
              elif value == "No":
                  return False
              return value
          filters = {
              "_category_name": reverse_map(st.session_state.category_name),
              "_product_description": reverse_map(st.session_state.product_description),
              "_deal_type": reverse_map(st.session_state.deal_type),
              "_min_price": reverse_map(st.session_state.min_price),
              "_max_price": reverse_map(st.session_state.max_price),
              "_condition": reverse_map(st.session_state.condition),
              "_status": reverse_map(st.session_state.status),
              "_product_name": reverse_map(st.session_state.product_name),
              "_allow_inspect": reverse_map(st.session_state.allow_inspect),
              "_delivery_method": reverse_map(st.session_state.delivery_method),
              "_address": reverse_map(st.session_state.address),
              "_num_days": reverse_map(st.session_state.num_days),
          }
          chat_response = ("assistant", "Here are products matching filters content:\n" + fetch_product(filters))
          st.session_state.chat_history.append(chat_response) # Combine the existing content with the new ones (both are lists of tuples)

    # Enter button  
    #st.write("")
    st.markdown("<br><br>", unsafe_allow_html=True)
    button_submit = f'''{button_style} <button class="mybutton" title="Submit your question">⏎</button>'''   
    submit_button = st.markdown(button_submit, unsafe_allow_html=True)
    #submit_button = st.markdown('<button title="Submit your question">⏎</button>', unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    button_delete = f'''{button_style} <button class="mybutton" title="Delete chat history">🗑️</button>'''   
    #if st.markdown(button_delete, unsafe_allow_html=True):
        #st.session_state.chat_history = []
        #st.session_state.prompt_history = system_guide

n_KB = int(st.sidebar.selectbox("Number of KB records to find/query:", ["1", "2", "3", "4", "5"]))
# Create a menu
if 'index' not in st.session_state:
    st.session_state['index'] = None
st.sidebar.markdown("<font color='darkblue'><b><p style='text-align: center; line-height: 0;'>DATABASE MANAGEMENT</p></b></font>", unsafe_allow_html=True)
options = st.sidebar.selectbox(
    'Choose a task option:',
    ('UPDATE KNOWLEDGE-BASE', '', ''),
    key='menu',
    index = st.session_state['index']
)
if options is not None:
    st.session_state['index'] = ('UPDATE KNOWLEDGE-BASE', '', '').index(options)
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

# CHAT PROCESS
# Chatbot function
def chat(user_input):
    st.session_state.prompt_history.append(("user", user_input))
    st.session_state.chat_history.append(("user", user_input))
    try:
        # Assuming GPT-4 API interaction
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": role, "content": content} for role, content in st.session_state.prompt_history],
            functions=[{
                "name": "search_product",
                "description": "Search products from the database based on user input",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category_name": {"type": "string", "description": "Product category explicitly entered by the user with the word 'category'"},
                        "product_name": {"type": "string", "description": "Product name"},
                        "product_description": {"type": "string", "description": "Product description"},
                        "deal_type": {"type": "string", "description": "Deal type value is 'Seller' if user asks to buy/find items posted for sale or 'Buyer' if user asks to sell/find items that others are looking to buy"},
                        "min_price": {"type": "number", "description": "Product min price"},
                        "max_price": {"type": "number", "description": "Product max price"},
                        "condition": {"type": "string", "description": "Product condition: New, Like New, Excellent, Good, Fair, Salvage"},
                        "status": {"type": "string", "description": "Product status: Available, Sold out, Still searching, Bought"},
                        "allow_inspect": {"type": "boolean", "description": "Whether allow pre-purchase inspection"},
                        "delivery_method": {"type": "string", "description": "Delivery method: Pickup, Delivery Driver, Shipping"},
                        "address": {"type": "string", "description": "Address"},
                        "num_days": {"type": "number", "description": "The maximum number of days since the product was posted, indicating that the user wants to search for products posted within this number of days from today"},
                    }
                }
            }]
        )
        #st.write(response)
        content = response.choices[0].message.content
        if content:
            st.session_state.prompt_history.append(("assistant", content))
            st.session_state.chat_history.append(("assistant", content))

        if response.choices[0].finish_reason == "function_call":
            function_name = response.choices[0].message.function_call.name
            function_args = json.loads(response.choices[0].message.function_call.arguments)

            if function_name == "search_product":
                filters = update_filters(function_args)
                st.session_state.chat_history.append(("assistant", fetch_product(filters)))
                return st.session_state.chat_history, filters
            else:
                st.session_state.chat_history.append(("system", f"Unhandled function call: {function_name}"))

    except Exception as e:
        st.session_state.chat_history.append(("system", f"An error occurred: {str(e)}"))

    #return st.session_state.chat_history, None
# Handle chat submission
if submit_button and user_input.strip() != "":
    chat(user_input)
    st.session_state.prompt_history
    st.session_state.chat_history
    user_input = ""
    #st.rerun()

# Display chat history
for chat in st.session_state.chat_history:
    if chat[0] == "user":
        st.write(f"**You**: {chat[1]}")
    elif chat[0] == "assistant":
        st.write(f"**Bot**: {chat[1]}")
