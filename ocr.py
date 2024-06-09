# Function to crop and zoom an image at a specific point
# Function to crop and zoom an image at a specific point
# def crop_and_zoom(img, x, y, zoom_factor):
#         width, height = img.size
#         zoom_ratio = zoom_factor * 2
#         cropped_img = img.crop(
#             (x - width / zoom_ratio, y - height / zoom_ratio, x + width / zoom_ratio, y + height / zoom_ratio))
#         return cropped_img.resize((width, height), Image.LANCZOS)


# Function to preprocess images (resize and zoom)
# def preprocess_images(input_folder, ocr_model, target_size):
#         for filename in os.listdir(input_folder):
#             if filename not in already_exist_data():
#                try:
#                    ocr_model.ocr(os.path.join(input_folder, filename))
#                except Exception as e:
#                    print(f"Failed to preprocess {filename}: {e}")


# Function to extract text from images using OCR

from paddleocr import PaddleOCR
import os
import sqlite3
from flask import Flask, render_template, request
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__, template_folder='templates', static_folder='static', static_url_path='/')


def extract_image_text(ocr_model, input_folder='input_data'):
    img_name_dict = {}
    already = set(already_exist_data())

    def process_image(file):
        if file not in already:
            img_path = os.path.join(input_folder, file)
            try:
                result = ocr_model.ocr(img_path, cls=True, rec=True)
                if result[0] is not None:
                    img_name = [i[1][0] for data in result for i in data]
                    return file, img_name
            except Exception as e:
                print(f"Error processing {file}: {e}")
        return file, []

    with ThreadPoolExecutor() as executor:
        results = executor.map(process_image, os.listdir(input_folder))
        for file, img_name in results:
            if img_name:
                img_name_dict[file] = img_name

    return img_name_dict


def already_exist_data():
    db_path = "my_database.db"
    table_name_to_check = "images_file_database"
    already_exist = []

    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        if table_exists(conn, table_name_to_check):
            exist_data = c.execute("SELECT image_name FROM images_file_database").fetchall()
            already_exist = [data[0] for data in exist_data]
    except sqlite3.Error as e:
        print(f"Error: {e}")
    finally:
        conn.close()

    return already_exist


def initialize_database_with_text(image_text_dict):
    conn = sqlite3.connect('my_database.db')
    c = conn.cursor()

    # Create tables if not exist
    c.execute(
        "CREATE TABLE IF NOT EXISTS images_file_database (id INTEGER PRIMARY KEY AUTOINCREMENT, image_name TEXT);")
    c.execute(
        "CREATE TABLE IF NOT EXISTS images_database (id INTEGER PRIMARY KEY AUTOINCREMENT, image_id TEXT NOT NULL, image_text TEXT);")

    # Insert image names and text data into tables
    for image_name, image_text_list in image_text_dict.items():
        c.execute("INSERT INTO images_file_database (image_name) VALUES (?)", (image_name,))
        for image_text in image_text_list:
            c.execute("INSERT INTO images_database (image_id, image_text) VALUES (?, ?)", (image_name, image_text))

    conn.commit()
    conn.close()


def table_exists(conn, table_name):
    cursor = conn.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
    return cursor.fetchone() is not None


def train_ai_model():
    ocr_model = PaddleOCR(lang='en')
    image_text_data = extract_image_text(ocr_model)
    initialize_database_with_text(image_text_data)


@lru_cache(maxsize=128)
def search_images_by_text(input_text):
    conn = sqlite3.connect("my_database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT image_id FROM images_database WHERE image_text = ?", (input_text,))
    image_list = [row[0] for row in cursor.fetchall()]
    conn.close()
    return image_list


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    input_text = request.form.get('bib_number')
    if input_text:
        image_ids = search_images_by_text(input_text)
        image_paths = ["images/{}".format(image_id) for image_id in image_ids] if image_ids else []
        message = "" if image_ids else "No image found"
    else:
        image_paths = []
        message = "Please enter a BIB number"

    return render_template('index.html', images=image_paths, message=message)


if __name__ == '__main__':
    train_ai_model()
    app.run(debug=True)
