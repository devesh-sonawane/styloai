import requests
from bs4 import BeautifulSoup
import pandas as pd
import random
import re

BASE_URL = "https://www2.hm.com/en_us/men/products/"
CATEGORIES = {
    "hoodies-sweatshirts": "Hoodies And Sweatshirts",
    "pants": "Pants",
    "jeans": "Jeans",
    "jackets-coats": "Jackets And Coats",
    "shirts": "Shirts",
    "t-shirts-tank-tops": "T-Shirts And Tank Tops",
    "polos": "Polos",
    "cardigans-sweaters": "Cardigans And Sweaters",
    "suits-blazers": "Suits And Blazers",
    "shorts": "Shorts",
    "nightwear-loungewear": "Nightwear And Loungewear",
}


def scrape_hm_products(category, formatted_category):
    products = []

    for page in range(1, 4):  # Scrape up to page 3
        url = f"{BASE_URL}{category}.html?page={page}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(
                f"Failed to retrieve {category}, page {page}, status code: {response.status_code}"
            )
            break

        soup = BeautifulSoup(response.text, "html.parser")
        product_list = soup.find_all("article", class_="hm-product-item")

        if not product_list:
            print(f"No products found for {category} on page {page}, stopping.")
            break

        for product in product_list:
            product_name_tag = product.find("h2", class_="item-heading")
            product_price_tag = product.find(
                "span", class_=["aeecde", "ac3d9e"]
            )  # Adjusted class for price
            product_link_tag = product.find("a", class_="link")

            product_name = (
                product_name_tag.text.strip() if product_name_tag else "Unknown"
            )
            product_link = product_link_tag.get("href") if product_link_tag else "#"
            if product_link and not product_link.startswith("http"):
                product_link = "https://www2.hm.com" + product_link

            # Extract price using regex to find numbers after the '$' sign, excluding 0.00
            product_price = 0
            if product_price_tag:
                product_price_text = product_price_tag.text.strip()
                match = re.search(r"\$\s*(\d+\.\d+)", product_price_text)
                if match and match.group(1) != "0.00":
                    product_price = match.group(
                        1
                    )  # Extracts the price value like "34.99"

            color_tags = product.find_all("li", class_="filter-item")
            colors = (
                [tag.find("a").get("title", "Unknown") for tag in color_tags[:3]]
                if color_tags
                else []
            )
            while len(colors) < 3:
                colors.append(
                    random.choice(["Black", "White", "Gray", "Navy", "Beige"])
                )

            products.append(
                {
                    "Type": "Men",
                    "Category": formatted_category,
                    "Sub Category": product_name,
                    "Price (in $)": product_price,
                    "Colour 1": colors[0],
                    "Colour 2": colors[1],
                    "Colour 3": colors[2],
                    "Link": product_link,
                }
            )

    return products


# Loop through each category and scrape data
all_products = []
for category, formatted_category in CATEGORIES.items():
    all_products.extend(scrape_hm_products(category, formatted_category))

# Save to CSV
if all_products:
    df = pd.DataFrame(all_products)
    df.to_csv("fashion_data.csv", index=False)
    print("Data saved to fashion_data.csv")

# Print scraped data
for product in all_products:
    print(
        f"{product['Type']} - {product['Category']} - {product['Sub Category']} - {product['Price (in $)']} - {product['Colour 1']}, {product['Colour 2']}, {product['Colour 3']} - {product['Link']}"
    )
