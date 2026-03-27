import requests
from bs4 import BeautifulSoup
import pandas as pd

user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
_headers = {'User-Agent': user_agent}

url = 'https://www.example.com/products'

try:
    response = requests.get(url, headers=_headers)
    response.raise_for_status()  # Check if the request was successful
    print("Successful response received.")
except requests.exceptions.HTTPError as http_err:
    print(f"HTTP error occurred: {http_err}")
except requests.exceptions.RequestException as e:
    print(f"Error fetching the URL: {e}")
except Exception as err:
    print(f"An error occurred: {err}")

soup = BeautifulSoup(response.content, 'html.parser')
print("Webpage title:", soup.title.string)

# Find the table containing the product data
table = soup.find('table', {'id': 'product-table'})
# Check if the table was found
rows = table.find_all('tr')
data = []
# Loop through the rows of the table and extract product information
for row in rows[1:]:
    cols = row.find_all('td')
    if len(cols) == 3:
        product_name = cols[0].text.strip()
        price = cols[2].text.strip()
        rating = cols[3].text.strip()
        data.append([product_name, price, rating])
    else:
        print("Unexpected number of columns in row, skipping:", row)
# Create a DataFrame from the extracted data
df = pd.DataFrame(data, columns=['Product Name', 'Price', 'Rating'])
df.to_csv(f'{soup.title.string}.csv', index=False)
print(f"Data has been saved to {soup.title.string}.csv")

