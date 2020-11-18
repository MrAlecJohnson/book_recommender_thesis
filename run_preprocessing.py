import time

from tqdm import tqdm
from pathlib import Path

from modules.preprocess import preprocess
from modules import load_data

raw_dir = Path.cwd() / "data/gutenberg_text"
end_dir = Path.cwd() / "data/gutenberg_processed"

# Preprocess whole book list
book_data = load_data.book_data()
numbers = book_data["Catalogue number"].to_list()
books = [book for book in raw_dir.rglob("*.txt") if int(book.stem) in numbers]

start = time.time()
for b in tqdm(books):
    preprocess(b, raw_dir, end_dir, use_id=False)
end = time.time()
print(f"Completed in {round((end - start)/60)} minutes")
