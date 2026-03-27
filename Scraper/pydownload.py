import requests
from bs4 import BeautifulSoup
import ssl
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context
import os
from urllib.parse import urljoin, urlparse

user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
headers = {"User-Agent": user_agent}

class TLSAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        context = create_urllib3_context()
        context.set_ciphers('DEFAULT@SECLEVEL=1')
        kwargs['ssl_context'] = context
        super().init_poolmanager(*args, **kwargs)

def download_file(url, destination):
    try:
        session = requests.Session()
        session.mount('https://', TLSAdapter())
        response = session.get(url, headers=headers)
        response.raise_for_status()  # Check if the request was successful
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')  # Parse the content to ensure it's valid
        else:
            print(f"Failed to parse the webpage. Status code: {response.status_code}")
            return
        download_links = soup.find_all('a', {'class': class_name})  # Adjust the selector based on the actual webpage structure
        if not download_links:
            print("No download links found.")
            return
        download_counter = 0
        for i, link in enumerate(download_links):
            file_url = link.get('href')
            if not file_url:
                continue
            download_counter += 1
            document_url = urljoin(url, file_url) if not file_url.startswith(('http://', 'https://')) else file_url
            file_name = os.path.basename(urlparse(document_url).path)
            folder = os.path.dirname(destination)
            full_destination = os.path.join(folder, file_name)
            document_response = session.get(document_url, headers=headers)
            document_response.raise_for_status()  # Check if the document request was successful
            with open(full_destination, 'wb') as file:
                file.write(document_response.content)                
        print(f"{download_counter} File(s) downloaded successfully: {destination}")
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
    except Exception as err:
        print(f"An unexpected error occurred: {err}")

print("The Python file downloader is running...")
url = input("Enter the URL of the webpage: ")
while not url:
    print("URL cannot be empty. Please enter a valid URL.")
    url = input("Enter the URL of the webpage: ")

class_name = input("Enter the class name of the download link (default is 'download-link'): ") or "download-link"
destination = input("Enter the destination path (including filename): ")
while not destination:
    print("Destination cannot be empty. Please enter a valid destination.")
    destination = input("Enter the destination path (including filename): ")

if not os.path.exists(os.path.dirname(destination)):
    os.makedirs(os.path.dirname(destination))

download_file(url, destination)


