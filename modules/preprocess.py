import re
import spacy
import numpy as np


nlp = spacy.load("en_core_web_lg")
default_tokeniser = nlp.Defaults.create_tokenizer(nlp)

# Regexes for leaving out lines that aren't text of the actual books
skip_regex = re.compile(r"transcriber's note|http|www\.|etext|ebook|project gutenberg")
start_regex = re.compile(r"^(\*+\s*START OF TH|\*END[* ]THE SMALL PRINT|CHAPTER )")
end_regex = re.compile(r"^(THE END$|\*+END OF|End of the Project Gutenberg)")


def tokenise(text, tokeniser=default_tokeniser):
    """Uses a Spacy tokeniser to return lower cased lemmas for tokens
    that aren't punctuation, stopwords or spaces
    """
    return [
        t.lemma_.lower().strip()
        for t in tokeniser(text)
        if not t.is_punct and not t.is_stop and not t.is_space
    ]


def preprocess(
    book,
    start_dir,
    end_dir,
    skip=skip_regex,
    start=start_regex,
    end=end_regex,
    use_id=True,
):
    """Take a book file and produce a preprocessed version of it in a new file.
    New files have _processed appended to filename.

    Preprocessing will strip out Gutenberg metadata from the start and end, then
    lemmatise the text and drop punctuation, stopwords and excess spaces

    Parameters
    ----------
    book (int or str, depending on use_id):
        A way to find the book to preprocess - can be Gutenberg ID or Path to a file

    start_dir (Path):
        pathlib Path to the directory containing the raw text files

    end_dir (Path):
        pathlib Path to the directory where you want to save the processed text files

    skip (compiled regex):
        Drop lines matching this regex from the processed text

    start (compiled regex):
        Don't include any lines until you find one matching this regex

    end (compiled regex):
        Ignore all lines after finding one matching this regex

    use_id (bool):
        If True, expect the book parameter to be a numerical Gutenberg ID
        Otherwise expect a Path
    """
    # Find the book file
    if use_id:
        book_id = book
        book_path = start_dir / (str(book) + ".txt")
    else:
        book_id = book.stem
        book_path = book

    out_file = end_dir / (str(book_id) + "_processed.txt")

    # Open file and tokenise from start match to end match
    with open(book_path, "r", encoding="latin-1") as f_in, open(out_file, "w") as f_out:
        content = False
        for line in f_in:
            if not content:
                if re.match(start, line):
                    content = True
            else:
                if re.match(end, line):
                    break
                elif re.search(skip, line):
                    pass
                else:
                    output = re.sub(" +", " ", " ".join(tokenise(line.strip())))
                    print(output, end=" ", file=f_out)


def find_features(
    book, folder, style_dict, skip=skip_regex, start=start_regex, end=end_regex,
):
    """
    Using same start/end/skip approach as preprocess, create style vectors.
    Style vectors are based on get_style_average() from Gutentag
    (https://github.com/julianbrooke/GutenTag)

    Style vectors are created by mapping words to a predesigned dictionary of weights.
    The weights represent the 'six styles' research (see report) plus a sentiment value
    Includes word count as an extra element in the vector.

    Can be refactored to remove the repetition of preprocess
    """
    book_path = folder / (str(book) + ".txt")

    with open(book_path, "r", encoding="latin-1") as f:
        content = f.readlines()

    styles = np.array([0.0] * 7)
    length = 0
    in_dict = 0

    lines = len(content)
    i = 0
    while i < lines:
        i += 1
        if re.match(start, content[i]):
            break

    while i < lines:
        if re.match(end, content[i]):
            break
        elif re.search(skip, content[i]):
            i += 1
        else:
            # Check each word for its stylistic values
            for word in content[i].strip().split():
                length += 1
                values = style_dict.get(word, None)
                if values is not None:
                    styles += values
                    in_dict += 1
            i += 1

    styles /= in_dict
    return np.append(styles, length)


def load_style_dict(filename):
    """Adapted from Gutentag (https://github.com/julianbrooke/GutenTag)
    but converted to use Numpy arrays for efficiency.
    """
    style_dict = {}
    with open(filename, "r", encoding="utf-8") as f:
        f.readline()  # skip the header row
        for line in f:
            stuff = line.strip().split()
            word = stuff[0]
            style_dict[word] = np.array([float(num) for num in stuff[1:]])
    return style_dict
