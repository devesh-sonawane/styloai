import openai
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents import AgentType
from langchain.callbacks import StreamlitCallbackHandler
import webbrowser
import streamlit.components.v1 as components

# Configure OpenAI API
openai_api_key = "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"  # Replace with your OpenAI API key
# Replace with your OpenAI API key

# Load your fashion product data
df = pd.read_csv(
    "/Users/deveshsonawane/Devesh_Workspace/Spring_2025/GenAI/styloai/hm_products.csv"
)  # Replace with your actual CSV file path
df["Version"] = (
    df.groupby("Name").cumcount() + 1
)  # Adding a version count starting from 1

# Create a new 'Name' column by appending the version to the original 'Name'
df["Name"] = df["Name"] + " V" + df["Version"].astype(str)

# Keep only the first image link from the 'Images' column (assuming it's a string or list separated by commas)
df["Images"] = df["Images"].apply(
    lambda x: x.split(",")[0] if isinstance(x, str) else x
)

# Drop the 'Version' column as it's no longer needed
df = df.drop(columns=["Version"])

# Initialize SentenceTransformer model for embedding
embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Create FAISS index
embeddings = embedding_model.encode(df["Name"].tolist(), convert_to_numpy=True)
index = faiss.IndexFlatL2(
    embeddings.shape[1]
)  # Create FAISS index with appropriate dimensions
index.add(embeddings)

# Initialize the LLM
llm = ChatOpenAI(
    temperature=0, model="gpt-4o", openai_api_key=openai_api_key, streaming=True
)

# Create the pandas dataframe agent
pandas_df_agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    allow_dangerous_code=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    handle_parsing_errors=True,
)

system_message = (
    "You are Mina, an expert fashion shopping assistant for H&M. Your goal is to provide product recommendations, pricing, and cart assistance in a friendly and professional tone. "
    "Ensure responses are concise, engaging, and formatted clearly. If a query is unrelated to fashion or shopping, reply with: "
    "'I couldn’t understand. Please ask a different question related to H&M products or services.'\n\n"
    "**Example Interactions:**\n\n"
    "User: 'What’s a good summer outfit?'\n"
    "Mina: 'A light cotton dress or shorts with a breathable top would be perfect for summer! Would you like some recommendations?'\n\n"
)

# Style
st.markdown(
    """
    <style>
        /* Set white background */
        body {
            background-color: white !important;
        }

        /* Style title */
        .stMarkdown h1 {
            color: #8B0000 !important;  /* Dark Red Color */
            text-align: center;
        }

        /* Style all buttons */
        .custom-button {
            padding: 8px 16px;
            font-size: 16px;
            cursor: pointer;
            border: 2px solid #8B0000;  /* Dark Red Border */
            color: #8B0000;  /* Dark Red Text */
            background-color: white;
            border-radius: 5px;
            transition: 0.3s;
        }

        /* Hover effect */
        .custom-button:hover {
            background-color: #8B0000;  /* Dark Red Background on Hover */
            color: white;
        }

        /* Align all buttons in the center */
        .button-container {
            display: flex;
            justify-content: center;
            gap: 10px;
            flex-wrap: wrap;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session states
if "cart" not in st.session_state:
    st.session_state.cart = []
if "contact_details" not in st.session_state:
    st.session_state.contact_details = {}
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hey, my name is Mina. How can I help you today?",
        }
    ]
if "system" not in st.session_state:
    st.session_state.system_message = system_message


# Function to display cart
def show_cart():
    if st.session_state.cart:
        st.write("### Your Cart")
        total_price = sum(
            float(p["Price"][1:]) * p["Quantity"] for p in st.session_state.cart
        )
        for product in st.session_state.cart:
            st.write(
                f"**{product['Name']}** - {product['Price']} x {product['Quantity']}"
            )
            st.image(product["Image Link"], width=80)
        st.write(f"**Total Price:** ${total_price:.2f}")
    else:
        st.write("🛒 Your cart is empty.")


# Function to handle contact details
def add_edit_contact_details():
    st.session_state.contact_details["Name"] = st.text_input(
        "Full Name", st.session_state.contact_details.get("Name", "")
    )
    st.session_state.contact_details["Phone"] = st.text_input(
        "Phone Number", st.session_state.contact_details.get("Phone", "")
    )
    st.session_state.contact_details["Card"] = st.text_input(
        "Card Number", st.session_state.contact_details.get("Card", "")
    )
    st.session_state.contact_details["Address"] = st.text_area(
        "Address", st.session_state.contact_details.get("Address", "")
    )
    if st.button("Save Details"):
        st.success("Contact details saved!")


# Function to handle checkout
def checkout():
    if not st.session_state.contact_details:
        st.warning("Please add your contact details before checkout.")
        add_edit_contact_details()
    else:
        st.success("✅ Checkout completed!")


# Title with red color
st.markdown("<h1>Shopping Assistant</h1>", unsafe_allow_html=True)

# Create a layout with styled buttons
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1.1, 2])

with col1:
    if st.button("View Cart"):
        show_cart()

with col2:
    if st.button("Clear Cart"):
        st.session_state.cart.clear()
        st.success("🛒 Cart cleared!")
with col3:
    if st.button("Gift Card"):
        webbrowser.open_new("https://www2.hm.com/en_us/customer-service/gift-card.html")


with col4:
    if st.button("Newsletter"):
        webbrowser.open_new(
            "https://www2.hm.com/en_us/customer-service/newsletter.html"
        )

with col5:
    if st.button("Go to Checkout"):
        checkout()


col6, col7, col8, col9 = st.columns(
    [3, 2, 2.5, 2]
)  # Adjusted widths for the second row
with col6:
    if st.button("Add/Edit Contact Details"):
        add_edit_contact_details()

with col7:
    if st.button("Download App"):
        webbrowser.open_new("https://play.google.com/store/apps/details?id=com.hm.goe")

with col8:
    if st.button("Customer Support"):
        webbrowser.open_new("https://www2.hm.com/en_us/customer-service.html")

with col9:
    if st.button("Fake al button"):
        webbrowser.open_new(
            "https://www2.hm.com/en_us/men/products/hoodies-sweatshirts.html?priceRange=%5B14.99%2C27.3%5D"
        )


# Function to display the 'Add to Cart' button
def add_to_cart(product):
    # Add the product to the cart when the button is clicked
    quantity = st.number_input(
        f"Quantity for {product['Name']}",
        min_value=1,
        max_value=10,
        value=1,
        key=f"quantity_{product['Name']}",
    )
    if st.button(f"Add {product['Name']} to Cart", key=f"add_{product['Name']}"):
        # Simulate adding to cart
        product["Quantity"] = quantity
        if "cart" not in st.session_state:
            st.session_state.cart = []
        st.session_state.cart.append(product)
        st.success(f"Added {quantity} of {product['Name']} to cart!")
        return True
    return False


def show_product_details(row):
    # Clean the color list and format it
    colors_list = row["Colors"].split(", ")
    cleaned_colors = [color.split("/")[0].strip() for color in colors_list]
    formatted_colors = ", ".join(cleaned_colors)

    # Display product details
    st.write(f"**Product Name:** {row['Name']}")
    st.write(f"**Price:** {row['Price']}")

    # Show images
    image_urls = row["Images"].split(",")  # Get all image URLs
    for image_url in image_urls:
        st.image(image_url.strip(), width=100)

    st.write(f"**Available Colors:** {formatted_colors}")
    st.write(f"[View Product Link]({row['Product Link']})")

    # Add to cart button
    quantity = st.number_input(
        f"Quantity for {row['Name']}",
        min_value=1,
        max_value=10,
        value=1,
        key=f"quantity_{row['Name']}",
    )
    if st.button(f"Add {row['Name']} to Cart", key=f"add_{row['Name']}"):
        product = {
            "Name": row["Name"],
            "Price": row["Price"],
            "Colors": formatted_colors,
            "Image Link": image_urls[
                0
            ].strip(),  # Taking the first image for cart display
            "Product Link": row["Product Link"],
            "Quantity": quantity,
        }
        add_to_cart(product, quantity)
        st.success(f"Added {quantity} of {row['Name']} to the cart!")


# Function to display fashion products in the new format
def display_fashion_products(user_input):
    fashion_keywords = [
        "dress",
        "shoes",
        "pants",
        "shirt",
        "jeans",
        "jacket",
        "sneakers",
        "coat",
        "blazer",
        "t-shirt",
        "sweater",
        "hoodie",
        "skirt",
        "shorts",
        "socks",
        "heels",
        "outfit",
        "accessories",
        "bag",
        "hat",
        "fashion",
    ]

    if any(keyword in user_input.lower() for keyword in fashion_keywords):
        query_embedding = embedding_model.encode([user_input], convert_to_numpy=True)
        distances, indices = index.search(query_embedding, 5)  # Retrieve top 5 products

        retrieved_products = df.iloc[indices[0]]  # Get the relevant product rows

        # Display the product images and prices first
        product_details = "Here are some fashion items I found:\n"

        # Create columns for images, prices, and buttons
        cols = st.columns(5)  # Assuming we are showing 5 products

        for i, (idx, row) in enumerate(retrieved_products.iterrows()):
            # Get the image URLs and display them
            image_urls = row["Images"].split(",")  # Get all image URLs

            # Display images side by side
            cols[i].image(
                image_urls[0].strip(), width=100
            )  # Display the first image as an example
            cols[i].write(f"**Price**: {row['Price']}")  # Display the price
            if cols[i].button(f"View Product", key=f"view_product_{i}"):
                # Show product details when the user clicks the button
                show_product_details(row, i)

    else:
        st.write(
            "I couldn’t understand. Please ask a different question related to H&M products or services."
        )


def display_message(role, content):
    cols = st.columns([1, 10])  # Create a layout with two columns
    if role == "assistant":
        with cols[0]:
            st.image(
                "https://thumbs.dreamstime.com/b/chat-bot-icon-speech-bubble-shape-background-chat-bot-icon-speech-bubble-shape-background-virtual-assistant-website-chat-100280898.jpg",
                width=30,
            )
            with cols[1]:
                st.write(content)
    else:
        st.write(content)


if user_input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": user_input})
    messages = [
        {"role": "system", "content": st.session_state.system_message}
    ] + st.session_state.messages
    if any(keyword in user_input.lower() for keyword in ["show", "display", "view"]):
        response = display_fashion_products(user_input)
    else:
        ai_msg = llm.invoke(messages)
        response = ai_msg.content
        st.session_state.messages.append({"role": "assistant", "content": response})
        display_message("assistant", response)
