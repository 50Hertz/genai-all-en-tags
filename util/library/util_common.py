import json
import os
import shutil
from datetime import datetime, timedelta
from random import sample

from googletrans import Translator
from util.constants import ALLOWED_LANGUAGES


def copy_file(file_location, destination_folder):
    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    # Copy the file to the destination folder
    shutil.copy(file_location, destination_folder)


def copy_file_with_id(file_location, destination_folder, unique_id):
    os.makedirs(destination_folder, exist_ok=True)

    # Extract the file name and extension
    file_name, file_extension = os.path.splitext(os.path.basename(file_location))

    # Create the new file name by appending the unique ID
    new_file_name = f"{file_name}_{unique_id}{file_extension}"

    # Copy the file to the destination folder with the new name
    shutil.copy(file_location, os.path.join(destination_folder, new_file_name))


def nanoid(n=10):
    return ''.join(sample('abcdefghijklmnopqrstuvwxyz', n))

def detect_language(text):
    translator = Translator()
    detection = translator.detect(text)
    return detection.lang


def translate_query(query, source_lang, target_lang='en'):
    translator = Translator()
    translation = translator.translate(query, src=source_lang, dest=target_lang)
    return translation.text


def is_allowed_language(lang):
    return lang in ALLOWED_LANGUAGES


# Check if a string is not None, not empty, and not just whitespace
def has_text(s):
    return bool(s and s.strip())


# Subtract from the date based on the interval
def subtract_date(interval: str, date: datetime) -> datetime:
    amount = int(interval[:-1])  # Get the number part of the interval
    unit = interval[-1]  # Get the last character of the interval, which represents the unit

    if unit == 'h':
        return date - timedelta(hours=amount)
    elif unit == 'd':
        return date - timedelta(days=amount)
    elif unit == 'm':
        return date - timedelta(days=amount * 30)  # Approximate a month as 30 days
    elif unit == 'y':
        return date - timedelta(days=amount * 365)  # Approximate a year as 365 days
    else:
        raise ValueError("Invalid interval unit. Available options are: 'h', 'd', 'm', 'y'")


# Define a custom function to serialize datetime objects
def serialize_datetime(obj):
    if isinstance(obj, datetime):
        dt_str = obj.strftime("%Y-%m-%d %H:%M:%S")
        return json.dumps(dt_str)
    raise TypeError("Type not serializable")
