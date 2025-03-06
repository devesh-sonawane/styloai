import os
import faiss
import numpy as np
import pandas as pd
import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = "AIzaSyDvHWEDEEEoM_M-rgVb8XqTTFB-8YcssW4"
api_key = GOOGLE_API_KEY
genai.configure(api_key=api_key)

# Load product data
df = pd.read_csv("hm_products.csv")

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Vectorize product descriptions
product_texts = df["Name"] + " " + df["Colors"]
embeddings = embedding_model.encode(product_texts.tolist(), convert_to_numpy=True)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# User Cart
cart = []


# Function to get similar products
def retrieve_similar_products(query, top_k=5):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    return df.iloc[indices[0]]


# Streamlit UI
st.title("Fashion Product Chatbot")
st.write("Ask about fashion products, recommendations, or add to cart!")

user_input = st.text_input("Ask a question:")
if user_input:
    retrieved_products = retrieve_similar_products(user_input)
    context_text = "\n".join(
        [
            f"Product: {row['Name']}, Price: ${row['Price']}, Colors: {row['Colors']}, Link: {row['Product Link']}"
            for _, row in retrieved_products.iterrows()
        ]
    )

    # Query Gemini with retrieved product context
    model = genai.GenerativeModel("gemini-2.0")
    prompt = f"You are a helpful fashion assistant. Use the following retrieved products as context: {context_text}. Answer the user's query accordingly."
    response = model.generate_content(prompt)
    st.write(response.text)

    # Display retrieved products
    for _, row in retrieved_products.iterrows():
        st.write(f"**{row['Name']}** - ${row['Price']}")
        st.write(f"Colors: {row['Colors']}")
        st.write(f"[View Product]({row['Product Link']})")
        if st.button(f"Add {row['Name']} to Cart", key=row["Name"]):
            cart.append(row)

# Show Cart
if st.button("View Cart"):
    st.subheader("Your Cart")
    total_price = 0
    for item in cart:
        st.write(f"**{item['Name']}** - ${item['Price']}")
        total_price += float(item["Price"])
    st.write(f"**Total: ${total_price:.2f}**")
