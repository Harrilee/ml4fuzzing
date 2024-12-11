from faker import Faker
import random

fake = Faker()


def generate_random_nav():
    nav_html = "<nav >\n<ul>\n"
    # Generate a few nav items
    for _ in range(random.randint(3, 6)):
        item_text = fake.word().capitalize()
        nav_html += f'  <li><a href="#{item_text.lower()}">{item_text}</a></li>\n'
    nav_html += "</ul>\n</nav>\n"
    return nav_html


def generate_random_paragraphs(num=3):
    paragraphs = ""
    for _ in range(num):
        paragraphs += f"<p>{fake.paragraph()}</p>\n"
    return paragraphs


def generate_random_table(rows=3, cols=3):
    table_html = "<table border='1' style='border-collapse:collapse; width:50%;'>\n"
    # Header row
    table_html += "  <tr>\n"
    for _ in range(cols):
        table_html += f"    <th>{fake.word().capitalize()}</th>\n"
    table_html += "  </tr>\n"
    # Data rows
    for _ in range(rows):
        table_html += "  <tr>\n"
        for _ in range(cols):
            table_html += f"    <td>{fake.word()}</td>\n"
        table_html += "  </tr>\n"
    table_html += "</table>\n"
    return table_html


def generate_random_image():
    # Since we're generating a random image tag, we can't fetch real images without external calls.
    # We'll just use a placeholder image service like https://via.placeholder.com/
    width = random.randint(100, 500)
    height = random.randint(100, 500)
    image_url = f"https://via.placeholder.com/{width}x{height}"
    alt_text = fake.sentence(nb_words=3)
    return f'<img src="{image_url}" alt="{alt_text}" style="max-width:100%; height:auto;">\n'


def generate_random_section():
    section_html = "<section>\n"
    # Add a heading
    section_html += f"<h2>{fake.catch_phrase()}</h2>\n"
    # Add some paragraphs
    section_html += generate_random_paragraphs(random.randint(2, 4))
    # Add an image
    section_html += generate_random_image()
    section_html += "</section>\n"
    return section_html


def generate_random_article():
    article_html = "<article>\n"
    # Add a heading
    article_html += f"<h3>{fake.bs()}</h3>\n"
    # Add paragraphs
    article_html += generate_random_paragraphs(random.randint(2, 3))
    # Add a table
    article_html += generate_random_table(random.randint(2, 4), random.randint(2, 4))
    article_html += "</article>\n"
    return article_html


def generate_random_footer():
    footer_html = "<footer>\n"
    footer_html += f"<p>&copy; {fake.year()} {fake.company()}. All rights reserved.</p>\n"
    footer_html += "</footer>\n"
    return footer_html


def generate_random_html():
    html = "<!DOCTYPE html>\n"
    html += "<html lang='en'>\n"
    html += "<head>\n"
    html += f"  <meta charset='UTF-8'>\n"
    html += "  <meta name='viewport' content='width=device-width, initial-scale=1.0'>\n"
    html += f"  <title>{fake.catch_phrase()}</title>\n"
    html += "  <style>\n"
    html += "    body { font-family: Arial, sans-serif; margin: 20px; }\n"
    html += "    nav ul { list-style: none; padding: 0; }\n"
    html += "    nav ul li { display: inline; margin-right: 10px; }\n"
    html += "    h1, h2, h3 { color: #333; }\n"
    html += "    footer { margin-top: 50px; font-size: 0.9em; color: #555; }\n"
    html += "  </style>\n"
    html += "</head>\n"
    html += "<body>\n"

    # Header
    html += "<header>\n"
    html += f"<h1>{fake.company()}</h1>\n"
    html += generate_random_nav()
    html += "</header>\n"

    # Main content
    html += "<main>\n"
    # Generate a few sections
    for _ in range(random.randint(2, 4)):
        html += generate_random_section()
    # Generate a few articles
    for _ in range(random.randint(1, 2)):
        html += generate_random_article()
    html += "</main>\n"

    # Footer
    html += generate_random_footer()

    html += "</body>\n"
    html += "</html>\n"

    return html



############################################################################################################

import sys
from bs4 import BeautifulSoup

lines = 0

def exec_trace_tracker(frame, event, arg):
    global lines
    code = frame.f_code
    func_name = code.co_name
    func_line_no = frame.f_lineno
    file_name = code.co_filename
    log = f"{func_line_no} | {event} | {func_name} | {file_name}"
    print(log)
    lines += 1




def test1():
    fakeHTML = generate_random_html()
    print(fakeHTML)
    elementToFind = "a"
    soup = BeautifulSoup(fakeHTML, "html.parser")

    sys.settrace(exec_trace_tracker)
    elements = soup.find_all(elementToFind)
    sys.settrace(None)

    print(lines)

if __name__ == "__main__":
    test1()