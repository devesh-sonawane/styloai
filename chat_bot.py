import os
import streamlit as st
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables (API Key)
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load Gemini 2.0 model
MODEL_NAME = "gemini-2.0-flash"
model = genai.GenerativeModel(model_name=MODEL_NAME)

# Load the fashion dataset
DATA_FILE = "hm_products.csv"
df = pd.read_csv(DATA_FILE)

# Ensure required columns exist
df = df[["Name", "Price", "Colors", "Product Link"]]

# Convert price to float for filtering
df["Price"] = df["Price"].replace("[\$,]", "", regex=True).astype(float)

# Initialize shopping cart
if "cart" not in st.session_state:
    st.session_state.cart = []

# Vectorize product names for RAG search
vectorizer = TfidfVectorizer(stop_words="english")
product_vectors = vectorizer.fit_transform(df["Name"])


# Function to find relevant products based on user query
def retrieve_similar_products(query, budget=None, top_n=5):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, product_vectors).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]

    retrieved_df = df.iloc[top_indices]

    # Filter products under the given budget
    if budget:
        retrieved_df = retrieved_df[retrieved_df["Price"] <= budget]

    return retrieved_df


# Function to extract budget from user query (if specified)
def extract_budget(query):
    words = query.split()
    for word in words:
        if word.startswith("$") and word[1:].replace(".", "", 1).isdigit():
            return float(word[1:])
    return None


# Function to generate a Gemini AI response
def generate_gemini_response(user_query, retrieved_products):
    product_details = "\n".join(
        [
            f"{row['Name']} - ${row['Price']} - {row['Colors']} ({row['Product Link']})"
            for _, row in retrieved_products.iterrows()
        ]
    )

    prompt = f"""
    You are a smart fashion shopping assistant. A user has asked: "{user_query}".
    Here are the best matching products:

    {product_details}

    Please provide a helpful response, suggest the best options, and include links if relevant.
    """

    response = model.generate_content([prompt])  # Gemini 2.0 expects a list input
    return response.text if response else "Sorry, I couldn't find relevant products."


# Streamlit UI
st.title("🛍️ Fashion Chatbot - H&M Products")

# User query input
user_query = st.text_input(
    "Ask me about fashion products, pricing, or recommendations:"
)

if user_query:
    # Extract budget if specified
    budget = extract_budget(user_query)

    # Retrieve similar products under budget (if specified)
    retrieved_products = retrieve_similar_products(user_query, budget)

    # Generate AI response
    response_text = generate_gemini_response(user_query, retrieved_products)

    # Display AI response in UI-friendly format
    st.subheader("🤖 AI Recommendation")
    st.write(response_text)

    # **🌟 Display AI Recommended Products in a structured format**
    st.subheader("🛒 AI Recommended Products")

    # Create product columns (2 per row)
    columns = st.columns(2)

    for i, (_, product) in enumerate(retrieved_products.iterrows()):
        with columns[i % 2]:  # Distribute across 2 columns
            st.markdown(f"### {product['Name']}")
            st.write(f"💲 **Price:** ${product['Price']}")
            st.write(f"🎨 **Colors Available:** {product['Colors']}")
            st.link_button("🔗 View Product", product["Product Link"])

            # Add to cart button with a unique key
            if st.button(
                f"🛒 Add '{product['Name'][:15]}'", key=f"cart_{product['Name']}"
            ):
                st.session_state.cart.append(product)
                st.success(f"Added {product['Name']} to cart!")

# Display shopping cart
st.subheader("🛍️ Shopping Cart")
cart_items = st.session_state.cart
if cart_items:
    total_price = sum(item["Price"] for item in cart_items)
    for item in cart_items:
        st.write(f"✅ {item['Name']} - 💲{item['Price']} - 🎨 {item['Colors']}")
    st.write(f"**🛒 Total Price: 💲{total_price:.2f}**")

    # Option to clear cart
    if st.button("🗑️ Clear Cart"):
        st.session_state.cart = []
        st.rerun()
