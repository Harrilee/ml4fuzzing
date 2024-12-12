import random
from faker import Faker

fake = Faker()

def random_id():
    return f"id-{random.randint(0, 3)}"

def random_class():
    return f"class-{random.randint(0, 3)}"

def generate_random_nav():
    nav_html = f'<nav id="{random_id()}" class="{random_class()}">\n<ul>\n'
    # Generate a few nav items with id/class as well
    for _ in range(random.randint(3, 5)):
        item_text = fake.word().capitalize()
        nav_html += f'  <li id="{random_id()}" class="{random_class()}"><a id="{random_id()}" class="{random_class()}" href="#{item_text.lower()}">{item_text}</a></li>\n'
    nav_html += "</ul>\n</nav>\n"
    return nav_html

def generate_random_paragraphs(num=3):
    paragraphs = ""
    for _ in range(num):
        paragraphs += f'<p id="{random_id()}" class="{random_class()}">{fake.paragraph()}</p>\n'
    return paragraphs

def generate_random_table(rows=3, cols=3):
    table_html = f'<table id="{random_id()}" class="{random_class()}" border="1" style="border-collapse:collapse; width:50%;">\n'
    # Header row
    table_html += "  <tr>\n"
    for _ in range(cols):
        table_html += f'    <th id="{random_id()}" class="{random_class()}">{fake.word().capitalize()}</th>\n'
    table_html += "  </tr>\n"
    # Data rows
    for _ in range(rows):
        table_html += "  <tr>\n"
        for _c in range(cols):
            table_html += f'    <td id="{random_id()}" class="{random_class()}">{fake.word()}</td>\n'
        table_html += "  </tr>\n"
    table_html += "</table>\n"
    return table_html

def generate_random_image():
    width = random.randint(100, 500)
    height = random.randint(100, 500)
    image_url = f"https://via.placeholder.com/{width}x{height}"
    alt_text = fake.sentence(nb_words=3)
    return f'<img id="{random_id()}" class="{random_class()}" src="{image_url}" alt="{alt_text}" style="max-width:100%; height:auto;">\n'

def generate_random_section():
    section_html = f'<section id="{random_id()}" class="{random_class()}">\n'
    # Add a heading
    section_html += f'<h2 id="{random_id()}" class="{random_class()}">{fake.catch_phrase()}</h2>\n'
    # Add some paragraphs
    section_html += generate_random_paragraphs(random.randint(2, 4))
    # Add an image
    section_html += generate_random_image()
    section_html += "</section>\n"
    return section_html

def generate_random_article():
    article_html = f'<article id="{random_id()}" class="{random_class()}">\n'
    # Add a heading
    article_html += f'<h3 id="{random_id()}" class="{random_class()}">{fake.bs()}</h3>\n'
    # Add paragraphs
    article_html += generate_random_paragraphs(random.randint(2, 3))
    # Add a table
    article_html += generate_random_table(random.randint(2, 4), random.randint(2, 4))
    article_html += "</article>\n"
    return article_html

def generate_random_footer():
    footer_html = f'<footer id="{random_id()}" class="{random_class()}">\n'
    footer_html += f'<p id="{random_id()}" class="{random_class()}">&copy; {fake.year()} {fake.company()}. All rights reserved.</p>\n'
    footer_html += "</footer>\n"
    return footer_html

def generate_random_html():
    html = "<!DOCTYPE html>\n"
    html += "<html lang='en' id='id-0' class='class-0'>\n"
    html += "<head>\n"
    html += "  <meta charset='UTF-8'>\n"
    html += "  <meta name='viewport' content='width=device-width, initial-scale=1.0'>\n"
    html += f"  <title id='{random_id()}' class='{random_class()}'>{fake.catch_phrase()}</title>\n"
    html += "  <style>\n"
    html += "    body { font-family: Arial, sans-serif; margin: 20px; }\n"
    html += "    nav ul { list-style: none; padding: 0; }\n"
    html += "    nav ul li { display: inline; margin-right: 10px; }\n"
    html += "    h1, h2, h3 { color: #333; }\n"
    html += "    footer { margin-top: 50px; font-size: 0.9em; color: #555; }\n"
    html += "  </style>\n"
    html += "</head>\n"
    html += f"<body id='{random_id()}' class='{random_class()}'>\n"

    # Header
    html += f'<header id="{random_id()}" class="{random_class()}">\n'
    html += f'<h1 id="{random_id()}" class="{random_class()}">{fake.company()}</h1>\n'
    html += generate_random_nav()
    html += "</header>\n"

    # Main content
    html += f'<main id="{random_id()}" class="{random_class()}">\n'
    # Generate multiple sections to ensure coverage
    for _ in range(random.randint(1, 3)):
        html += generate_random_section()
    # Generate multiple articles as well
    for _ in range(random.randint(1, 2)):
        html += generate_random_article()
    html += "</main>\n"

    # Footer
    html += generate_random_footer()

    html += "</body>\n"
    html += "</html>\n"

    return html

if __name__ == "__main__":
    # Generate and print random HTML
    print(generate_random_html())