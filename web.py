import openai
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents import AgentType
import webbrowser
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load fashion product data
df = pd.read_csv("hm_products.csv")
df["Images"] = df["Images"].apply(
    lambda x: x.split(",")[0] if isinstance(x, str) else x
)

# Initialize embedding model & FAISS index
embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
embeddings = embedding_model.encode(df["Name"].tolist(), convert_to_numpy=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Initialize OpenAI GPT model
llm = ChatOpenAI(
    temperature=0, model="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True
)
pandas_df_agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    allow_dangerous_code=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    handle_parsing_errors=True,
)

# Initialize session states
if "cart" not in st.session_state:
    st.session_state.cart = []
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hey, my name is Mina. How can I help you today?",
        }
    ]


# Function to display chat messages
def display_messages():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])


# Show chat history before user input
display_messages()

# Handle user input
if user_input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    if any(keyword in user_input.lower() for keyword in ["show", "display", "view"]):
        display_fashion_products(user_input)
    else:
        ai_msg = llm.invoke(
            [
                {"role": "system", "content": "You are an AI fashion assistant."},
                {"role": "user", "content": user_input},
            ]
        )
        response = ai_msg.content
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Display assistant response
    with st.chat_message("assistant"):
        st.write(response)

    # Refresh chat display
    display_messages()
