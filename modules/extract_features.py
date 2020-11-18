import pickle

from tqdm import tqdm
from pathlib import Path
from preprocess import load_style_dict, find_features
from modules.load_data import book_data, map_book_ids

raw_dir = Path.cwd() / "data/gutenberg_text"
end_dir = Path.cwd() / "models"

# Extract features from whole book list - defining each book as a vector
# The vector consists of the 'six styles' values plus the book's word count
data = book_data()
numbers = data["Catalogue number"].to_list()
gutenberg_to_gr = map_book_ids("gute_to_gr")
styles = load_style_dict(end_dir / "sixstyleplus.txt")

features = {}
for book in tqdm(numbers):
    features[gutenberg_to_gr[book]] = find_features(book, raw_dir, styles)

# Save output to be used by style vectorisers
pickle.dump(features, open(end_dir / "features.pkl", "wb"))
