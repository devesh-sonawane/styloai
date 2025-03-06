import requests
import json
import pandas as pd

# API URL
API_URL = "https://api.hm.com/search-services/v1/en_us/listing/resultpage"

# Headers to mimic a browser request
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# API Parameters
params = {
    "pageSource": "PLP",
    "page": 1,
    "sort": "RELEVANCE",
    "pageId": "/men/shop-by-product/view-all",
    "page-size": 36,
    "categoryId": "men_viewall",
    "filters": "sale:false||oldSale:false",
    "touchPoint": "DESKTOP",
    "skipStockCheck": "false",
}

# List to store product data
all_products = []


def fetch_products(page):
    """Fetches product data from the API for a specific page."""
    params["page"] = page  # Set the current page number
    response = requests.get(API_URL, headers=HEADERS, params=params)

    print(f"Fetching Page {page}: Status Code {response.status_code}")

    if response.status_code == 200:
        try:
            data = response.json()
            return data
        except json.JSONDecodeError:
            print("Error decoding JSON. Full Response:")
            print(response.text)
            return None
    else:
        print(f"Failed to retrieve page {page}. Status code: {response.status_code}")
        return None


# Loop through multiple pages
page = 1
while True:
    data = fetch_products(page)
    if not data or "plpList" not in data or "productList" not in data["plpList"]:
        break  # Stop if no more products

    products = data["plpList"]["productList"]

    for product in products:
        name = product.get("productName", "").strip()

        # Extracting price
        price_list = product.get("prices", [])
        price = None
        if price_list:
            price = (
                price_list[0].get("formattedPrice", "").strip()
            )  # First price in list

        # Extract product URL
        link = f"https://www.hm.com{product.get('url', '')}"

        # Extract available colors
        colors = [color["colorName"] for color in product.get("swatches", [])]

        # Extract product images
        # images = [color["productImage"] for color in product.get("swatches", [])]

        all_products.append(
            {
                "Name": name,
                "Price": price,
                "Colors": ", ".join(colors),
                "Product Link": link,
                # ,"Images": ", ".join(images),
            }
        )

    # Pagination - Check if next page exists
    if "pagination" in data and "nextPageNum" in data["pagination"]:
        page += 1  # Move to the next page
    else:
        break  # Stop if there's no next page

# Save data to CSV
df = pd.DataFrame(all_products)
df.to_csv("hm_products2.csv", index=False)

print("✅ Scraping completed. Data saved to hm_products2.csv")
