import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger()
# Set the watchdog logger to suppress debug messages
watchdog_logger = logging.getLogger("watchdog")
watchdog_logger.setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("numexpr").setLevel(logging.WARNING)
logging.getLogger("pinecone_plugin_interface").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("hpack").setLevel(logging.WARNING)
logging.getLogger("pinecone.core.openapi.shared.rest").setLevel(logging.WARNING)
logging.getLogger("geopy").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
print("Starting Streamlit App") # file=sys.stdout

import streamlit as st
import pandas as pd
import random
import time
# from openai import OpenAI
import google.generativeai as Ggenai
from google.protobuf.struct_pb2 import Struct # Import Struct for safer dict conversion
#import textwrap3 as textwrap
import dotenv
from dotenv import load_dotenv
import numpy as np
import os
from pinecone import Pinecone
import csv
import json
from supabase import create_client
from datetime import datetime, timezone
from streamlit_folium import st_folium
import folium
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderUnavailable, GeocoderTimedOut
from PIL import Image
import base64
import io
import traceback

embedding_model = "text-embedding-ada-002"

try:
    #OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"] 
    #PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    #SENDGRID_API_KEY = st.secrets["SENDGRID_API_KEY"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except Exception as e:
    # Secrets not found in Streamlit, try loading from local .env file
    load_dotenv()
    #OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")    
    #PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    #SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    if not SUPABASE_KEY or not GOOGLE_API_KEY: #not OPENAI_API_KEY or not PINECONE_API_KEY or not SENDGRID_API_KEY or 
        st.error("Environment file error or secrets not found!")
        st.error(e)
# Set OpenAI API key
#gpt = OpenAI(api_key = OPENAI_API_KEY)
Ggenai.configure(api_key = GOOGLE_API_KEY)
#gemini = Ggenai.GenerativeModel(model_name="gemini-2.0-pro-exp") 
gemini = Ggenai.GenerativeModel(
    model_name="gemini-1.5-flash-latest",
)

#index_name = 'matchder-kb'
#pc = Pinecone( # initialize connection to pinecone
#    api_key=PINECONE_API_KEY,
#    environment="us-west1-gcp-free")
#try:
#    pincone_index = pc.Index(index_name) # connect to pinecone index
#except:
#    print("Could not connect to Pinecone")
#    pincone_index = None

supabase = create_client("https://ziaisnybjhkjcbprxava.supabase.co", SUPABASE_KEY)
product_category_list = ["All"]
#def fetch_categories():
# Fetches category names from the public.categories table and stores them in the global categories_list."""
  # global categories_list
try:
        # Query the categories table to fetch category names
        response = supabase.table("categories").select("name").execute()
        if response.data:
            # Extract category names from the query result
            product_category_list += [category['name'] for category in response.data]
        else:
            product_category_list = ["No categories found"]
except Exception as e:
        st.error(f"Error fetching categories: {e}")

# Call the function to fetch categories when the app starts
# fetch_categories()

# --- Define Gemini Tool Schema ---
# valid_categories = [cat for cat in product_category_list if cat != "All"] # Exclude "All" for LLM choices
extract_product_features_tool = Ggenai.types.FunctionDeclaration(
    name="extract_product_features",
    description="Extract product features based on content attached",
    parameters={
        "type": "object",
        "properties": {
            "category_name": {"type": "string", "description": f"Product category, choose from: {', '.join(product_category_list)}"},
            "product_name": {"type": "string", "description": "Product name or keywords"},
            "product_description": {"type": "string", "description": "Follow specific instructions provided in the prompt for how to generate Product description"}, # "Key words of prod description"
            "deal_type": {"type": "string", "enum": ["Buyer", "Seller"], "description": "Use 'Seller' if the user wants to buy/find items for sale. Use 'Buyer' if the user wants to sell/find purchase requests."},
            "min_price": {"type": "number", "description": "Minimum product price in USD"},
            "max_price": {"type": "number", "description": "Maximum product price in USD"},
            "condition": {"type": "string", "enum": ["New", "Like New", "Excellent", "Good", "Fair", "Salvage"], "description": "Product condition"},
            "status": {"type": "string", "enum": ["Available", "Sold out", "Still searching", "Bought"], "description": "Product listing status"},
            "allow_inspect": {"type": "boolean", "description": "Whether pre-purchase inspection is allowed (true/false)"},
            "delivery_method": {"type": "string", "enum": ["Pickup", "Delivery Driver", "Shipping"], "description": "Available delivery method"},
            "location": {"type": "string", "description": "Address or general area of the product location"},
            "search_radius": {"type": "number", "description": "Maximum distance in miles to search around the location"},
            "num_days": {"type": "integer", "description": "Search for products posted within this many days from today"},
        }
        # "required": ["deal_type"] # Example: require at least deal_type
    }
)

def coordinate2location(lat, lon):
    geolocator = Nominatim(user_agent="myGeocoder")
    try: 
        location = geolocator.reverse((lat, lon), exactly_one=True, timeout = 10)
        return location.address if location else "" # "Location not found"
    except (GeocoderUnavailable, GeocoderTimedOut) as e:
        print(f"Error in coordinate2location: {e}")
        return ""
def location2coordinate(addr):
    geolocator = Nominatim(user_agent="myGeocoder")
    try:
        location = geolocator.geocode(addr, timeout =10)
        if location is not None: return [location.latitude, location.longitude]
        else: return None
    except (GeocoderUnavailable, GeocoderTimedOut) as e:
        print(f"Error in location2coordinate: {e}")
        return None

def update_filters(function_args):
    """Parses LLM function arguments and updates session state filters"""
    #allow_inspect_value = function_args.get("allow_inspect")
    #allow_inspect_value = "Yes" if allow_inspect_value is True else "No" if allow_inspect_value is False else None
    st.session_state.deal_type = function_args.get("deal_type", "All")
    st.session_state.product_name = function_args.get("product_name", "")
    st.session_state.location = function_args.get("location", "")
    st.session_state.search_radius = function_args.get("search_radius", 0.0)
    st.session_state.num_days = str(function_args.get("num_days", "")) #Text field
    st.session_state.condition = function_args.get("condition", "All")
    st.session_state.status = function_args.get("status", "All")
    st.session_state.allow_inspect = function_args.get("allow_inspect", "All")
    st.session_state.delivery_method = function_args.get("delivery_method", "All")
    st.session_state.min_price = function_args.get("min_price", 0.0)
    st.session_state.max_price = str(function_args.get("max_price", ""))  #Text field 2000000000.0
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
        "_location": function_args.get("location", None),
        "_coordinates": location2coordinate(function_args.get("location", None)),
        "_search_radius": function_args.get("search_radius", None),
        "_num_days": function_args.get("num_days", None),
    }
    return filters

# Function to fetch products
def fetch_product(filters):
    columns_to_display = ["name", "description", "deal_type", "price", "published_at", "condition",
                          "status", "rating", "allow_inspect", "delivery_method", "location", "quantity", "product_image_urls"]
    try:
        product_data, count = supabase.rpc("filter_products", filters).execute()
        if product_data[1]:
            products_list = []
            # Construct the product display table
            for item in product_data[1]:
                if item.get("type") == "Buyer":
                    min_budget = item.get("min_budget")
                    max_budget = item.get("max_budget")
                    item["price"] = f"${min_budget} - ${max_budget}" if min_budget and max_budget else ""
                elif item.get("type") == "Seller":
                    price = item.get("price")
                    item["price"] = f"${price}" if price is not None else ""

                # Format dates and links
                if "published_at" in item and item["published_at"]:
                    try:
                        published_at_date = datetime.fromisoformat(item["published_at"])
                        now_utc = datetime.now(timezone.utc)
                        days_since_posting = (now_utc - published_at_date).days
                        item["published_at"] = f"{days_since_posting} days ago"
                    except ValueError:
                        item["published_at"] = "Invalid date"

                # Create a link to Google Maps for the location
                latitude = item.get("latitude")
                longitude = item.get("longitude")
                location = item.get("location")
                map_url = "N/A"
                if latitude and longitude:
                    map_url = f"https://google.com/maps?q={latitude},{longitude}"
                    #item["address"] = f"<a href='{map_url}' target='_blank'>{address}</a>"

                # Add item to the product list
                products_list.append({
                    "PRODUTCT NAME": item.get('name', ''),
                    "DESCRIPTION": item.get('description', ''),
                    "DATA PROVIDER": item.get('deal_type', ''),
                    "PRICE": item.get('price', ''),
                    "CONDITION": item.get('condition', ''),
                    "STATUS": item.get('status', ''),
                    "LISTING DURATION": item.get('published_at', ''),
                    "INSPECTABLE": item.get('allow_inspect', ''),
                    "DELIVERY METHOD": item.get('delivery_method', ''),
                    "LOCATION": item.get('location', ''),
                    "MAP LINK": map_url
                })
            df = pd.DataFrame(products_list)
            # Add a sequence number column starting from 1
            df.index = df.index + 1
            df.index.name = 'No.'
            return df
        else:
            return "No products match your query."
    except (ValueError, TypeError, Exception) as e:
        return f"Error: {str(e)}"

# Initialize history and state for chatbot
MAX_OUTPUT_TOKEN = 300
MAX_PROMPT_HISTORY_LENGTH = 30
# Initialize default values in session state (this only runs the first time)
def initialize():
  if 'initialized' in st.session_state:
    if len(st.session_state.prompt_history) > MAX_PROMPT_HISTORY_LENGTH:
      st.session_state.prompt_history = st.session_state.prompt_history[-MAX_PROMPT_HISTORY_LENGTH:]
  else: # session_state not initialized
    st.session_state.initialized = True
    st.session_state.chat_history = []
    st.session_state.prompt_history = []
    st.session_state.deal_type = "All"
    st.session_state.product_name = ""
    st.session_state.location = "Fremont, CA"
    st.session_state.search_radius = 0.0
    st.session_state.num_days = ""   # Empty mean infinity 365
    st.session_state.condition = "All"
    st.session_state.status = "All"
    st.session_state.allow_inspect = "All"
    st.session_state.delivery_method = "All"
    st.session_state.min_price = 0.0
    st.session_state.max_price = ""  # Empty mean infinity 2000000000.0
    st.session_state.category_name = "All"
    st.session_state.product_description = ""
    default_location = location2coordinate("Fremont, CA") or [37.5483, -122.2711]
    st.session_state['map'] = folium.Map(location=default_location, zoom_start=10) #creates a map centered at a specific location
    st.session_state['click_data'] = None

# Initialize the status dictionary if it doesn't exist
if 'search_expander_status' not in st.session_state:
    st.session_state.search_expander_status = None

initialize()
# STREAMLIT INTERFACE
with st.sidebar:
  #st.sidebar.image("robo.gif", use_container_width=True)
  # Set the text color and alignment for the selectbox label and options
  st.markdown("""
    <style>
    .stSelectbox > label,
    .stSelectbox > select {
      color: darkblue;
      font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
  st.markdown("<font color='darkblue'><b><p style='font-size: 21px; text-align: center; line-height: 0;'>MATCHDER CHATBOT</p></b></font>", unsafe_allow_html=True)
  #col1, col2 = st.columns([4.5, 1])
  #with col1: 
  #  st.write("<font color='darkblue'><b><p style='text-align: center; line-height: 0;'>Reset chat session</p></b></font>", unsafe_allow_html=True) 
  #with col2: 🗑️
  if st.button("**Reset chat session** 🗑️"):
      initialize()
      st.rerun()



  n_KB = 3 #int(st.selectbox("Number of KB records to find/query:", ["1", "2", "3", "4", "5"]))
  # Create a menu
  if 'index' not in st.session_state:
      st.session_state['index'] = None
  st.markdown("<font color='darkblue'><b><p style='text-align: center; line-align: 0;'>DATABASE MANAGEMENT</p></b></font>", unsafe_allow_html=True)
  options = st.selectbox(
      'Choose a task option:',
      ('UPDATE KNOWLEDGE-BASE', '', ''),
      key='menu',
      index = st.session_state['index']
  )
  if options is not None:
      st.session_state['index'] = ('UPDATE KNOWLEDGE-BASE', '', '').index(options)
      #expander=st.sidebar
      if 'login_status' not in st.session_state:
          st.session_state['login_status'] = False
      st.session_state['key'] = 0
      with st.expander:
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
                file_name = st.file_uploader("Select your files", type="pdf")

# Main interface
st.markdown("""
    <style>
    .stChatInput > div {
        border: 1px solid black !important;
        border-radius: 10px !important;
        resize: both !important;
        overflow: auto !important;
    }
    .stButton > button {
        background-color: #F0F0F0; /* very light gray */
        color: black; /* text */
        border: 2px solid black; /* Black border */
        border-radius: 5px; /* Rounded corners */
        padding: 5px 20px; /* Adjust padding as needed */
        min-width: 50px; /* Minimum width to avoid collapsing */
        box-sizing: border-box; /* Include padding in width calculation */
    }
    .stTextArea textarea {
        height: auto !important;
        min-height: 150px !important;
        resize: vertical !important;
    }
    </style>
""", unsafe_allow_html=True)
#    .stButton > button {
#        width: 45px;
#        height: 45px;
#        font-size: 25px;}

#with st.container():
with st.expander("**SEARCH FOR PRODUCTS BASED ON USER-SELECTED FILTERS AND IMAGES**", expanded=True):
  col1, col2 = st.columns([1, 1]) #([50, 1])
  with col1:
      st.session_state.deal_type = st.radio("Are you looking for", ["Buyer", "Seller", "All"],
         index=["Buyer", "Seller", "All"].index(st.session_state.deal_type) if st.session_state.deal_type in ["Buyer", "Seller", "All"] else 2)
      st.session_state.product_name = st.text_input("Product Name", value=st.session_state.product_name)
      st.session_state.location = st.text_input("Address of Product Location", value=st.session_state.location)
      coords = location2coordinate(st.session_state.location)
      if coords: # Update the map with the new location and the latest marker
        st.session_state['map'] = folium.Map(location=coords, zoom_start=10)
        folium.Marker(location=coords, popup=st.session_state.location).add_to(st.session_state['map'])
      else: st.warning("Invalid location. Please enter a valid address.")
      st.session_state.search_radius = st.number_input("Location Search Radius in Miles", min_value=0.0, value=st.session_state.search_radius)
      st.session_state.num_days = st.text_input("Within number of days since advertised", value=st.session_state.num_days)
      st.session_state.condition = st.radio("Condition", ["All", "New", "Like New", "Excellent", "Good", "Fair", "Salvage"],
         index=["All", "New", "Like New", "Excellent", "Good", "Fair", "Salvage"].index(st.session_state.condition) if st.session_state.condition in ["All", "New", "Like New", "Excellent", "Good", "Fair", "Salvage"] else 0)
      st.session_state.status = st.radio("Status", ["All", "Available", "Sold out", "Still searching", "Bought"],
         index=["All", "Available", "Sold out", "Still searching", "Bought"].index(st.session_state.status) if st.session_state.status in ["All", "Available", "Sold out", "Still searching", "Bought"] else 0)
      allow_inspect_options = {"All": None, "Yes": True, "No": False}
      allow_inspect_selection = st.radio("Pre-purchase Inspection Allowed", ["All", "Yes", "No"],
         index=["All", "Yes", "No"].index("Yes" if st.session_state.allow_inspect is True else "No" if st.session_state.allow_inspect is False else "All"))
      st.session_state.allow_inspect = allow_inspect_options[allow_inspect_selection]  # Store the boolean value
      st.session_state.delivery_method = st.radio("Delivery Method", ["All", "Pickup", "Delivery Driver", "Shipping"],
         index=["All", "Pickup", "Delivery Driver", "Shipping"].index(st.session_state.delivery_method) if st.session_state.delivery_method in ["All", "Pickup", "Delivery Driver", "Shipping"] else 0)
      st.session_state.min_price = st.number_input("Min Price", min_value=0.0, value=float(st.session_state.min_price), format="%.2f")
      st.session_state.max_price = st.text_input("Max Price", value=st.session_state.max_price)
      st.session_state.category_name = st.selectbox("Category Name", product_category_list,
         index=product_category_list.index(st.session_state.category_name) if st.session_state.category_name in product_category_list else 0)
      st.session_state.product_description = st.text_area("Product Description", value=st.session_state.product_description)
      st.session_state.use_description_for_search = st.checkbox("Include Product Description as a search filter", value=False)
      search_result = None
      if st.button("**SEARCH PRODUCT** 🔍", key="search_button"):
          def reverse_map(value):
              """Map 'All' and '' to None, 'Yes' to True, 'No' to False."""
              if value in ["All", "", 0]:
                  return None
              elif value == "Yes":
                  return True
              elif value == "No":
                  return False
              return value
          def str_to_int(str):
            try: return int(str)
            except: return None
          def str_to_float(str):
            try: return float(str)
            except: return None
          filters = {
              "_category_name": reverse_map(st.session_state.category_name),
              "_product_description": reverse_map(st.session_state.product_description) if st.session_state.use_description_for_search else None,
              "_deal_type": reverse_map(st.session_state.deal_type),
              "_min_price": st.session_state.min_price,
              "_max_price": str_to_float(st.session_state.max_price),
              "_condition": reverse_map(st.session_state.condition),
              "_status": reverse_map(st.session_state.status),
              "_product_name": reverse_map(st.session_state.product_name),
              "_allow_inspect": reverse_map(st.session_state.allow_inspect),
              "_delivery_method": reverse_map(st.session_state.delivery_method),
              "_location": reverse_map(st.session_state.location),
              "_coordinates": location2coordinate(st.session_state.location),
              "_search_radius": st.session_state.search_radius,
              "_num_days": str_to_int(st.session_state.num_days),
          }
          search_result = fetch_product(filters)
  with col2:
    # Handle map click events
    # print("Click data:", st.session_state['click_data'])  For debug
    if st.session_state.get('click_data') and st.session_state['click_data'].get('last_clicked'):
        lat = st.session_state.click_data['last_clicked']['lat']
        lon = st.session_state.click_data['last_clicked']['lng']
        if lat is not None and lon is not None:
          st.session_state.location = coordinate2location(lat, lon)
          st.session_state['click_data'] = None # Avoide this handling process when rerun without new click
          # Display the location in a text input
          st.session_state.location = st.text_input("Address of Product Location", value=st.session_state.location)
          st.session_state['map'] = folium.Map(location=[lat, lon], zoom_start=10) # Update the map with the latest marker
          folium.Marker(location=[lat, lon], popup=st.session_state.location).add_to(st.session_state['map'])
    # Display the map and capture click events
    st.session_state['click_data'] = st_folium(st.session_state['map'], width=700, height=1300, key="map_display")
    # returned data of st_folium: { 'last_clicked': { 'lat': 37.7749, 'lng': -122.4194 } }    
    uploaded_file = st.file_uploader("**UPLOAD PRODUCT IMAGE 🖼️**", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:           
            image = Image.open(uploaded_file)           
            st.image(image, caption='Uploaded Image of Product', width = None) # None: Fill width  

            if st.button("**UPDATE FILTERS FROM IMAGES**"):
                if image.mode in ('RGBA', 'LA', 'P'): # Convert to RGB explicitly to remove alpha channel
                    rgb_image = image.convert('RGB') 
                else:
                    rgb_image = image.copy()
                buffered = io.BytesIO()
                rgb_image.save(buffered, format="JPEG")
                #openai_image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
                image_data = buffered.getvalue() # Get the bytes, not base64 string
                product_category_list = ["Unknown" if item == "All" else item for item in product_category_list]
                mime_type = "image/jpeg"  # Default to JPEG
                if uploaded_file.type == "image/png":
                    mime_type = "image/png"
                elif uploaded_file.type == "image/gif":
                    mime_type = "image/gif"
                elif uploaded_file.type == "image/jpeg" or uploaded_file.type == "image/jpg":  # JPG check
                    mime_type = "image/jpeg"

                product_description_guide = "Product description in advertising style. Uncertain features are described within square brackets, using the most likely or popular features of similar products"
                try:
                    response = gemini.generate_content(
                        contents=[("Extract all relevant product features visible in the image using extract_product_features_tool tool."
                         "For features unknown or uncertain, refer to the most likely or popular available attributes of similar products inferred from image."                        
                         "For product_description field in tool, generate product description in advertising style with uncertain phrases enclosed in []."),
                         {"mime_type": mime_type, "data": image_data}],
                        generation_config=Ggenai.types.GenerationConfig(
                            temperature=1, 
                            max_output_tokens=MAX_OUTPUT_TOKEN),
                        tools=[extract_product_features_tool]
                    )
                    if not response.candidates:
                        raise ValueError("No response from Gemini!")
                    candidate = response.candidates[0]
                    if not candidate.content or not candidate.content.parts:
                        if response.prompt_feedback.block_reason:
                            block_reason = response.prompt_feedback.block_reason.name
                            safety_ratings = {rating.category.name: rating.probability.name for rating in response.prompt_feedback.safety_ratings}
                            error_message = f"Response blocked by safety filter: {block_reason}. Ratings: {safety_ratings}"
                            st.warning(error_message)
                    first_part = candidate.content.parts[0]
                    if hasattr(first_part, 'function_call') and first_part.function_call.name == "extract_product_features":
                        function_args = {}      # Convert proto map Struct to Python dict safely
                        try: # Use Struct to dict conversion for safety
                            temp_struct = Struct()
                            temp_struct.update(first_part.function_call.args)
                            function_args = dict(temp_struct)
                            print(f"Image Analysis Function call args received: {function_args}")
                        except Exception as args_e:
                            print(f"Error converting function args from Struct: {args_e}")
                            st.error("Internal error processing function arguments from image analysis.")
                            # Potentially set an error status here too if desired
                            # timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            # st.session_state.search_expander_status = {'message': f'Error processing image arguments at {timestamp}.', 'type': 'error'}
                            uploaded_file = None
                        filters_for_db = update_filters(function_args) # Update UI filters based on LLM args
                        # Store success status in session state instead of showing directly
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state.search_expander_status = {
                            'message': f"Filters updated successfully at {timestamp}!",
                            'type': 'success'
                        }
                        uploaded_file = None
                        st.rerun()  # Force a rerun to update the UI
                        # print(f"LLM response content: {response}") # This line won't be reached due to rerun
                    else: # Model didn't make the function call
                        text_response = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
                        st.warning(f"LLM did not extract features via function call. Model response: {text_response}")
                        # Potentially set a warning status
                        # timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        # st.session_state.search_expander_status = {'message': f'Image analysis incomplete at {timestamp}. LLM response: {text_response[:50]}...', 'type': 'warning'}
                        uploaded_file = None
                except Exception as e:
                    st.error(f"Error during Gemini API call for image analysis: {str(e)}")
                    # Potentially set an error status
                    # timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    # st.session_state.search_expander_status = {'message': f'Error during image analysis API call at {timestamp}.', 'type': 'error'}
                    print(f"Traceback: {traceback.format_exc()}")
                    uploaded_file = None
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            # Potentially set an error status
            # timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # st.session_state.search_expander_status = {'message': f'Error processing uploaded image at {timestamp}.', 'type': 'error'}
            print(f"Traceback: {traceback.format_exc()}")
            uploaded_file = None

  # --- Status Banner Display --- (Place this at the end, but INSIDE the expander) ---
  if st.session_state.search_expander_status:
    status = st.session_state.search_expander_status
    message = status.get('message', 'Status updated.') # Default message
    status_type = status.get('type', 'info') # Default type

    if status_type == 'success':
        st.success(message)
    elif status_type == 'warning':
        st.warning(message)
    elif status_type == 'error':
        st.error(message)
    else: # Default to info
        st.info(message)
    # The status persists until updated again or explicitly cleared elsewhere
    # To clear after showing once, uncomment below:
    # st.session_state.search_expander_status = None

if isinstance(search_result, pd.DataFrame):  # If it's a DataFrame, display it as a table
    st.write("Products match your selected filters are listed as follows:\n")
    st.dataframe(search_result, column_config={'MAP LINK': st.column_config.LinkColumn()})
    st.write("________________________________________________________________________________________________________________")
    #st.markdown(search_result.to_html(escape=False, index=False), unsafe_allow_html=True)
elif isinstance(search_result, str):  # If it's a text message, display the text
    st.write(search_result)

if st.session_state.chat_history:
    prev_role = None  # Track the previous chat role
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            if chat["role"] == "user" and prev_role != "user": st.markdown("**YOU:**")
            elif chat["role"] == "assistant" and prev_role != "assistant": st.markdown("**MATCHDER:**")
            elif chat["role"] == "system" and prev_role != "system": st.markdown("**SYSTEM:**")
            # Check if content is a list (assumed to be list of dicts) or a pandas DataFrame
            if isinstance(chat["content"], (list, pd.DataFrame)):
                st.write("Products match your query are listed as follows:\n")    
                if isinstance(chat["content"], list): # Convert list of dicts to DataFrame if necessary
                    df = pd.DataFrame(chat["content"])
                else: # it's already a DataFrame
                    df = chat["content"]
                st.dataframe(df, column_config={'MAP LINK': st.column_config.LinkColumn()})          
            else: st.write(chat["content"])
            prev_role = chat["role"]  # Update the previous role
#user_input = st.text_area("ENTER YOUR QUESTION", placeholder="Type here ...", key='user_input', height=1, help='Resize by dragging the bottom-right corner', max_chars=None,)
user_input = st.chat_input("Type your question here ...")

# CHAT PROCESS
def chat(user_input_text):
    chat_system_instruction=("You are Matchder, a helpful assistant for searching products."
        "Analyze user requests to understand their needs, ask if they can provide more info for unclear input or that's all."
        "If that's all, confirm final query with them before calling 'extract_product_features_tool' tool."
        "Note that deal_type is 'Seller' for buyers/finding sale items, 'Buyer' for sellers/finding purchase items."
        "If you lack enough info to proceed, refer users to the Matchder website at dev.matchder.com \n")
       
    if not st.session_state.prompt_history:  # Prepend system instructions to the first user message
        combined_first_message = chat_system_instruction + "\n\nUSER: " + user_input_text
        st.session_state.prompt_history.append({"role": "user", "parts": [{"text": combined_first_message}]})
    else:
        st.session_state.prompt_history.append({"role": "user", "parts": [{"text": user_input_text}]})
    st.session_state.chat_history.append({"role": "user", "content": user_input_text})
    # Ensure history doesn't exceed limit before sending
    if len(st.session_state.prompt_history) > MAX_PROMPT_HISTORY_LENGTH:
        keep_len = MAX_PROMPT_HISTORY_LENGTH
        if keep_len % 2 != 0: keep_len -= 1 # Try to keep pairs
        st.session_state.prompt_history = st.session_state.prompt_history[-keep_len:]
        print(f"Prompt history trimmed to {len(st.session_state.prompt_history)} before LLM call.")
    # contents = [{"role": "system", "parts": [{"text": system_prompt}]}] + st.session_state.prompt_history
    try:
        # First call to Gemini, potentially triggering function call
        response1 = gemini.generate_content(
            contents=st.session_state.prompt_history,
            generation_config=Ggenai.types.GenerationConfig(
                temperature=1, 
                max_output_tokens=MAX_OUTPUT_TOKEN
            ),
            tools=[extract_product_features_tool]
        )
        if not response1.candidates:
             raise ValueError("Gemini response did not contain candidates.")
        candidate = response1.candidates[0]
        # Handle safety blocks or empty responses
        if not candidate.content or not candidate.content.parts:
            if response1.prompt_feedback.block_reason:
                block_reason = response1.prompt_feedback.block_reason.name
                safety_ratings = {rating.category.name: rating.probability.name for rating in response1.prompt_feedback.safety_ratings}
                error_message = f"Response blocked by safety filter: {block_reason}. Ratings: {safety_ratings}"
                st.warning(error_message)
                st.session_state.chat_history.append({"role": "system", "parts": [{"text": error_message}]})
                # st.session_state.prompt_history.append({"role": "model", "parts": [{"text": f"[Blocked: {block_reason}]"}]})
                return # Stop processing if blocked
            else: # Normal empty response
                st.session_state.chat_history.append({"role": "assistant", "content": ""})
                st.session_state.prompt_history.append({"role": "model", "parts": [{"text": ""}]})
                return
        first_part = candidate.content.parts[0]
        # Check for Function Call
        if hasattr(first_part, 'function_call') and first_part.function_call.name == "extract_product_features":
            function_args = {}      # Convert proto map Struct to Python dict safely
            try: # Use Struct conversion
                temp_struct = Struct()
                temp_struct.update(first_part.function_call.args)
                function_args = dict(temp_struct)
            except Exception as args_e:
                 print(f"Error converting function args: {args_e}")
                 st.error("Internal error processing function arguments.")
                 st.session_state.chat_history.append({"role": "system", "content": "Error processing function arguments."})
                 return # Stop processing
            print(f"Function call: extract_product_feature", file=sys.stdout)
            print(f"Arguments: {function_args}", file=sys.stdout)

            # Update UI filters based on LLM args before calling fetch
            filters_for_db = update_filters(function_args)
            # st.success("Assistant is searching with updated filters...") # Indicate action
            # st.session_state.chat_history.append({"role": "assistant", "content": f"Okay, I will search for products with these criteria: {function_args}"})
            search_result_data = fetch_product(filters_for_db)
            search_result_json_str = "Error preparing result for LLM." # Default error
            if isinstance(search_result_data, pd.DataFrame):
                # Convert DataFrame to list of dicts for JSON serialization & display
                search_result_list_dict = search_result_data.to_dict(orient='records')
                try:
                    search_result_json_str = json.dumps(search_result_list_dict)
                except (TypeError, OverflowError) as json_err:
                    print(f"Error serializing DataFrame to JSON: {json_err}", file=sys.stdout)
                    search_result_json_str = json.dumps({"error": "Could not serialize product data."})                
                # Store the dataframe itself for display, not the list of dicts
                st.session_state.chat_history.append({"role": "assistant", "content": search_result_data})
            elif isinstance(search_result_data, str): # Error messages or "No products found" string            
                search_result_json_str = json.dumps({"message": search_result_data})
                st.session_state.chat_history.append({"role": "assistant", "content": search_result_data})
            else: # Handle unexpected return types            
                 search_result_json_str = json.dumps({"error": "Unexpected result type from search."})
                 st.session_state.chat_history.append({"role": "assistant", "content": "Error: Unexpected result from product search."})

            # Add the model's function call request and our function response to history for the next turn
            st.session_state.prompt_history.append({"role": "model", "parts": candidate.content.parts}) # Add Function response
            st.session_state.prompt_history.append({
                "role": "function",
                "parts": [{
                    "function_response": {
                        "name": "extract_product_features",
                        "response": {"result": search_result_json_str} # Ensure the response is a dict (JSON object) for Gemini
                    }
                }]
            }) # Add Function response dictionary
            # --- Second Call to Gemini for Summary ---
            response2 = gemini.generate_content(
                contents=st.session_state.prompt_history,
                generation_config=Ggenai.types.GenerationConfig(
                    temperature=0.5, # Less creative summary
                    max_output_tokens=MAX_OUTPUT_TOKEN),
                tools=[extract_product_features_tool] # Include tools again
            )
            if not response2.candidates:
                 raise ValueError("Gemini summary response did not contain candidates.")
            summary_candidate = response2.candidates[0]
            # Check for blocked summary
            if response2.prompt_feedback.block_reason:
                 block_reason = response2.prompt_feedback.block_reason.name
                 summary_content = f"[Summary blocked by safety filter: {block_reason}]"
                 st.warning(summary_content) # Show warning in UI
            elif summary_candidate.content and summary_candidate.content.parts:
                 summary_content = "".join(part.text for part in summary_candidate.content.parts if hasattr(part, 'text'))
            else:
                 summary_content = "[Assistant did not provide a summary.]"
            # Add single summary to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": summary_content})
            # Add summary to prompt history for next turn
            st.session_state.prompt_history.append({"role": "model", "parts": [{"text": summary_content}]}) # Text content for LLM
        else: # --- No Function Call - Regular Text Response ---
            content1 = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
            print(f"Regular response: {content1}", file=sys.stdout)
            if content1 is not None:
                st.session_state.prompt_history.append({"role": "model", "parts": [{"text": content1}]})
                st.session_state.chat_history.append({"role": "assistant", "content": content1})
            else: # Handle case where text response is None unexpectedly
                 st.session_state.chat_history.append({"role": "assistant", "content": "[Assistant produced no text response.]"})
                 st.session_state.prompt_history.append({"role": "model", "parts": [{"text": ""}]})
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(f"Error during chat processing: {error_message}")
        traceback.print_exc() # Print full traceback to console/log
        st.error("Sorry, something went wrong. Please try again.")
        st.session_state.chat_history.append({"role": "system", "content": error_message})
        # Attempt to recover history if possible, remove last user input if error occurred?
        # This part needs careful consideration to avoid corrupting state.
if user_input: # Handle chat submission
  if user_input.strip() != "":
    chat(user_input)
    user_input = ""
    st.rerun()
