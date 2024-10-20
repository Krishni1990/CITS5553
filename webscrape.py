import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import io
import PyPDF2
import time
import csv
import random

# Set to store visited URLs to avoid revisiting
visited_urls = set()
data_to_save = []  # List to store the data for CSV

# Function to make HTTP requests with rate limit handling
def make_request(url):
    try:
        response = requests.get(url)
        if response.status_code == 429:
            # Extract Retry-After header if present
            retry_after = response.headers.get('Retry-After')
            if retry_after:
                wait_time = int(retry_after)
            else:
                wait_time = 60  # Default wait time if Retry-After is not present
            print(f"Rate limit exceeded. Waiting for {wait_time} seconds.")
            time.sleep(wait_time)
            response = requests.get(url)
        return response
    except requests.RequestException as e:
        print(f"Request error: {e}")
        return None

def scrape_website(url):
    # Skip if URL is already visited
    if url in visited_urls:
        return

    try:
        # Mark URL as visited
        visited_urls.add(url)

        # Send HTTP request to the URL
        response = make_request(url)
        if response is None:
            return
        
        content_type = response.headers.get('Content-Type')

        # Check if the content is HTML
        if 'text/html' in content_type:
            soup = BeautifulSoup(response.content, 'html.parser')
            print(f"Scraping HTML URL: {url}")

            # Get page title
            page_title = soup.title.string if soup.title else 'No title'
            print("Page Title:", page_title)

            # Find all links in the current page
            links = soup.find_all('a', href=True)

            # Loop through all links found
            for link in links:
                href = link['href']
                link_text = link.get_text(strip=True) or 'No text'  # Get link text; use 'No text' if empty

                # Convert relative URLs to absolute URLs
                full_url = urljoin(url, href)

                # Print the link URL and its title text
                print(f"Link Text: {link_text}, URL: {full_url}")

                # Add data to list for CSV output with Title and Link/Content format
                data_to_save.append({
                    'Title': link_text,
                    'Link/Content': full_url
                })

                # Ensure we only visit links from the same domain
                if urlparse(full_url).netloc == urlparse(url).netloc:
                    scrape_website(full_url)

        # Check if the content is a PDF
        elif 'application/pdf' in content_type:
            print(f"Scraping PDF URL: {url}")
            scrape_pdf(url)

        # Sleep to avoid overloading the server
        time.sleep(random.uniform(1, 3))  # Randomize delay between requests

    except Exception as e:
        print(f"Error scraping {url}: {e}")

def scrape_pdf(url):
    try:
        # Download the PDF file
        response = make_request(url)
        if response is None:
            return

        # Read the PDF content
        with io.BytesIO(response.content) as f:
            reader = PyPDF2.PdfReader(f)
            num_pages = len(reader.pages)
            pdf_text = ""

            # Extract text from each page
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                text = page.extract_text()
                if text:
                    pdf_text += text

        # Add PDF content to list for CSV output with Title and Link/Content format
        data_to_save.append({
            'Title': 'PDF Content',  # Static title indicating this is PDF content
            'Link/Content': pdf_text  # The extracted text from the PDF
        })

        # Print the extracted text or process as needed
        print("Extracted PDF Text:", pdf_text)

    except Exception as e:
        print(f"Error scraping PDF {url}: {e}")

def save_to_csv(filename, data):
    # Specify the field names for CSV
    fieldnames = ['Title', 'Link/Content']

    # Write to CSV file
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"Data saved to {filename}")

# Start scraping from the main page
start_url = 'https://australianchildatlas.com'
scrape_website(start_url)

# Save the collected data to CSV
save_to_csv('scraped_datav3.csv', data_to_save)
S