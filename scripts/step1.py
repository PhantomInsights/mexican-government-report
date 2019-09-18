"""
This script reads the government report, extracts, cleans and saves
all the text so it can be analyzed in the next scripts.
"""

import re

import PyPDF2


CHARACTERS = {
    "ç": "Á",
    "⁄": "á",
    "…": "É",
    "”": "é",
    "ê": "Í",
    "™": 'í',
    "î": "Ó",
    "Š": "ó",
    "ı": "ö",
    "ò": "Ú",
    "œ": "ú",
    "Œ": "ñ",
    "Ô": "‘",
    "Õ": "’",
    "¥": "• ",
    "Ñ": "—",
    "¨": "®",
    "«": "´",
    "Ò": "“"
}


def extract_text():
    """Read the PDF contents and extract the text that we need."""

    reader = PyPDF2.PdfFileReader("informe.pdf")
    full_text = ""

    # The page numbers in the PDF are not the same as the reported
    # number of pages, we use this variable to keep track of both.
    pdf_page_number = 3

    # We will only retrieve the first 3 sections of the government report
    # which are between pages 14 and 326.
    for i in range(14, 327):

        # This block is used to remove the page number at the start of
        # each page. The first if removes page numbers with one digit.
        # The second if removes page numbers with 2 digits and the else
        # statement removes page numbers with 3 digits.
        if pdf_page_number <= 9:
            page_text = reader.getPage(i).extractText().strip()[1:]
        elif pdf_page_number >= 10 and pdf_page_number <= 99:
            page_text = reader.getPage(i).extractText().strip()[2:]
        else:
            page_text = reader.getPage(i).extractText().strip()[3:]

        full_text += page_text.replace("\n", "")
        pdf_page_number += 1

    # There's a small issue when decoding the PDF file.
    # We will manually fix all the weird characters
    # with their correct equivalents.
    for item, replacement in CHARACTERS.items():
        full_text = full_text.replace(item, replacement)

    # We remove all extra white spaces.
    full_text = re.sub(" +", " ", full_text)

    # Finally we save the cleaned text into a .txt file.
    with open("transcript_clean.txt", "w", encoding="utf-8") as temp_file:
        temp_file.write(full_text)


if __name__ == "__main__":

    extract_text()
