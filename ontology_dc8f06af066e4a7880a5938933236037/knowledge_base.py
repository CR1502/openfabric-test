import requests
from bs4 import BeautifulSoup
import csv

websites = [
    "https://en.wikipedia.org/wiki/Chemistry",
    "https://en.wikipedia.org/wiki/Matter",
    "https://en.wikipedia.org/wiki/Atom",
    "https://en.wikipedia.org/wiki/Chemical_compound",
    "https://en.wikipedia.org/wiki/Molecule",
    "https://en.wikipedia.org/wiki/Mole_(unit)",
    "https://en.wikipedia.org/wiki/Phase_(matter)",
    "https://en.wikipedia.org/wiki/Chemical_bond",
    "https://en.wikipedia.org/wiki/Energy",
    "https://en.wikipedia.org/wiki/Chemical_reaction",
    "https://en.wikipedia.org/wiki/History_of_chemistry",
    "https://en.wikipedia.org/wiki/Atomic_theory"
    "https://en.wikipedia.org/wiki/John_Dalton"
    "https://en.wikipedia.org/wiki/Niels_Bohr",
    "https://en.wikipedia.org/wiki/Bohr_model",
    "https://en.wikipedia.org/wiki/Quantum_chemistry",
    "https://en.wikipedia.org/wiki/Biology",
    "https://en.wikipedia.org/wiki/Cell_(biology)",
    "https://en.wikipedia.org/wiki/Prokaryote",
    "https://en.wikipedia.org/wiki/Eukaryote",
    "https://en.wikipedia.org/wiki/Organism",
    "https://en.wikipedia.org/wiki/Mitochondrion",
    "https://en.wikipedia.org/wiki/Plastid",
    "https://en.wikipedia.org/wiki/Cytoskeleton",
    "https://en.wikipedia.org/wiki/Cell_wall,"
    "https://en.wikipedia.org/wiki/Cell_membrane",
    "https://en.wikipedia.org/wiki/DNA",
    "https://en.wikipedia.org/wiki/RNA",
    "https://en.wikipedia.org/wiki/Cell_division",
    "https://en.wikipedia.org/wiki/Cell_nucleus",
    "https://en.wikipedia.org/wiki/Multicellular_organism",
    "https://en.wikipedia.org/wiki/Evolution_of_sexual_reproduction",
    "https://en.wikipedia.org/wiki/Physics",
    "https://en.wikipedia.org/wiki/Classical_physics",
    "https://en.wikipedia.org/wiki/Applied_physics",
    "https://en.wikipedia.org/wiki/Theoretical_physics",
    "https://en.wikipedia.org/wiki/Experimental_physics",
    "https://en.wikipedia.org/wiki/Isaac_Newton",
    "https://en.wikipedia.org/wiki/Thermodynamics",
    "https://en.wikipedia.org/wiki/Newton%27s_laws_of_motion",
    "https://en.wikipedia.org/wiki/Albert_Einstein",
    "https://en.wikipedia.org/wiki/Classical_electromagnetism"
]

data = []  # List to store the scraped data

# Scrape data from each website
for website in websites:
    # Send a GET request to the website
    response = requests.get(website)
    
    # Create a BeautifulSoup object to parse the HTML content
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Find the relevant elements on the page and extract the desired information
    title = soup.find(id="firstHeading").text
    content_paragraphs = soup.select("#mw-content-text > div > p")
    
    # Extract the text content from the paragraphs
    content = " ".join([paragraph.text for paragraph in content_paragraphs])
    
    # Create a dictionary to store the scraped data for this website
    website_data = {
        "Website": website,
        "Title": title,
        "Content": content
    }
    
    # Append the website data to the list
    data.append(website_data)

# Write the scraped data to a CSV file
csv_file = "scraped_data.csv"

fieldnames = ["Website", "Title", "Content"]

with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)

print("Data scraped and saved to", csv_file)
