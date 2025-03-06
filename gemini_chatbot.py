import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
GEMINI_API_KEY = "AIzaSyDvHWEDEEEoM_M-rgVb8XqTTFB-8YcssW4"
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Load fashion dataset
df = pd.read_csv(
    "hm_products.csv"
)  # Ensure this file exists and is structured correctly

# Initialize session state for shopping cart if not exists
if "cart" not in st.session_state:
    st.session_state.cart = []


# Function to generate AI response
def get_ai_response(prompt):
    model = genai.GenerativeModel("gemini-1.5")
    response = model.generate_content(prompt)
    return response.text


# Function to search for products
def search_products(query):
    results = df[df["Name"].str.contains(query, case=False, na=False)]
    return results


# Function to recommend outfits
def recommend_outfits(style):
    prompt = f"Suggest a stylish outfit based on the style: {style}. Choose from the following products: {df['Name'].tolist()}"
    return get_ai_response(prompt)


# Function to add item to cart
def add_to_cart(item_name):
    item = df[df["Name"] == item_name].to_dict(orient="records")
    if item:
        st.session_state.cart.append(item[0])
        st.success(f"Added {item_name} to your cart!")
    else:
        st.error("Item not found.")


# Function to remove item from cart
def remove_from_cart(item_name):
    st.session_state.cart = [
        item for item in st.session_state.cart if item["Name"] != item_name
    ]
    st.success(f"Removed {item_name} from your cart!")


# Function to display cart and total price
def show_cart():
    if st.session_state.cart:
        cart_df = pd.DataFrame(st.session_state.cart)
        st.table(cart_df[["Name", "Price", "Colors", "Product Link"]])
        total_price = cart_df["Price"].astype(float).sum()
        st.write(f"### Total Price: ${total_price:.2f}")
    else:
        st.write("Your cart is empty.")


# Streamlit UI
st.title("🛍️ Fashion Store Chatbot")
st.write(
    "Welcome! Ask me about our products, outfit recommendations, or manage your cart."
)

user_query = st.text_input("Type your question:")

if user_query:
    if "recommend an outfit" in user_query.lower():
        style = user_query.split("recommend an outfit for")[-1].strip()
        response = recommend_outfits(style)
        st.write(response)
    elif "add" in user_query.lower() and "cart" in user_query.lower():
        item_name = user_query.replace("add", "").replace("to cart", "").strip()
        add_to_cart(item_name)
    elif "remove" in user_query.lower() and "cart" in user_query.lower():
        item_name = user_query.replace("remove", "").replace("from cart", "").strip()
        remove_from_cart(item_name)
    elif "show cart" in user_query.lower():
        show_cart()
    else:
        products = search_products(user_query)
        if not products.empty:
            for _, row in products.iterrows():
                st.write(
                    f"**{row['Name']}** - ${row['Price']} - Colors: {row['Colors']}"
                )
                st.write(f"[View Product]({row['Product Link']})")
        else:
            response = get_ai_response(user_query)
            st.write(response)

# Show cart button
if st.button("View Cart"):
    show_cart()
