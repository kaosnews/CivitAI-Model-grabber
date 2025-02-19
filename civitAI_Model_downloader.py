import re
import json
import requests
import logging
import urllib.parse
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
import argparse

# Constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOGS_DIR, "civitAI_Model_downloader.txt")
OUTPUT_DIR = "model_downloads"
MAX_PATH_LENGTH = 200
VALID_DOWNLOAD_TYPES = ['Lora', 'Checkpoints', 'Embeddings', 'Training_Data', 'Other', 'All']
BASE_URL = "https://civitai.com/api/v1/models"

# Set up logging
logger_md = logging.getLogger('md')
logger_md.setLevel(logging.DEBUG)
file_handler_md = logging.FileHandler(LOG_FILE_PATH, encoding='utf-8')
file_handler_md.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler_md.setFormatter(formatter)
logger_md.addHandler(file_handler_md)

# Argument parsing
parser = argparse.ArgumentParser(description="Download model files and images from Civitai API.")
parser.add_argument("usernames", nargs='+', type=str, help="Enter one or more usernames you want to download from.")
parser.add_argument("--retry_delay", type=int, default=10, help="Retry delay in seconds.")
parser.add_argument("--max_tries", type=int, default=3, help="Maximum number of retries.")
parser.add_argument("--max_threads", type=int, default=5, help="Maximum number of concurrent threads. Too many produces API Failure.")
parser.add_argument("--token", type=str, default=None, help="API Token for Civitai.")

# Mutually exclusive group for filtering options
group = parser.add_mutually_exclusive_group()
group.add_argument(
    "--download_type",
    type=str,
    choices=VALID_DOWNLOAD_TYPES,
    help="Specify the type of content to download: 'Lora', 'Checkpoints', 'Embeddings', 'Training_Data', 'Other', or 'All'."
)
group.add_argument(
    "--exclude_type",
    type=str,
    choices=VALID_DOWNLOAD_TYPES,
    help="Download all content except the specified type (cannot use with --download_type)."
)

args = parser.parse_args()

# Prompt the user for the token if not provided via command line
if args.token is None:
    args.token = input("Please enter your Civitai API token: ")

# Determine filtering options. They are mutually exclusive.
download_type = None
exclude_type = None
if args.download_type:
    download_type = args.download_type
elif args.exclude_type:
    exclude_type = args.exclude_type
else:
    download_type = 'All'

# Initialize variables
usernames = args.usernames
retry_delay = args.retry_delay
max_tries = args.max_tries
max_threads = args.max_threads
token = args.token

# Function to sanitize directory names
def sanitize_directory_name(name):
    return name.rstrip()  # Remove trailing whitespace characters

# Create output directory
OUTPUT_DIR = sanitize_directory_name(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create session
session = requests.Session()

def read_summary_data(username):
    """Read summary data from a file in the logs subfolder."""
    summary_path = os.path.join(LOGS_DIR, f"{username}.txt")
    data = {}
    try:
        with open(summary_path, 'r', encoding='utf-8') as file:
            for line in file:
                if 'Total - Count:' in line:
                    total_count = int(line.strip().split(':')[1].strip())
                    data['Total'] = total_count
                elif ' - Count:' in line:
                    category, count = line.strip().split(' - Count:')
                    data[category.strip()] = int(count.strip())
    except FileNotFoundError:
        print(f"File {summary_path} not found.")
    return data

def sanitize_name(name, folder_name=None, max_length=MAX_PATH_LENGTH, subfolder=None, output_dir=None, username=None):
    """Sanitize a name for use as a file or folder name."""
    base_name, extension = os.path.splitext(name)

    if folder_name and base_name == folder_name:
        return name

    if folder_name:
        base_name = base_name.replace(folder_name, "").strip("_")

    # Remove problematic characters and control characters
    base_name = re.sub(r'[<>:"/\\|?*\x00-\x1f\x7f-\x9f]', '_', base_name)

    # Handle reserved names (Windows specific)
    reserved_names = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}
    if base_name.upper() in reserved_names:
        base_name = '_'

    # Reduce multiple underscores to single and trim leading/trailing underscores and dots
    base_name = re.sub(r'__+', '_', base_name).strip('_.')

    # Calculate max length of base name considering the path length
    if subfolder and output_dir and username:
        path_length = len(os.path.join(output_dir, username, subfolder))
        max_base_length = max_length - len(extension) - path_length
        base_name = base_name[:max_base_length].rsplit('_', 1)[0]

    sanitized_name = base_name + extension
    return sanitized_name.strip()

def download_file_or_image(url, output_path, retry_count=0, max_retries=max_tries):
    """Download a file or image from the provided URL and determine the correct file extension."""
    if os.path.exists(output_path):
        return False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    progress_bar = None
    try:
        response = session.get(url, stream=True, timeout=(20, 40))
        if response.status_code == 404:
            print(f"File not found: {url}")
            return False
        response.raise_for_status()

        # Determine the file extension based on the Content-Type header
        content_type = response.headers.get('Content-Type', '')
        if 'image' in content_type:
            file_extension = '.jpg'  # You can further refine this based on specific image types
        elif 'video' in content_type:
            file_extension = '.mp4'  # You can further refine this based on specific video types
        else:
            file_extension = os.path.splitext(output_path)[1]  # Keep the original extension if not image/video

        # Update the output path with the correct extension
        output_path = os.path.splitext(output_path)[0] + file_extension

        total_size = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, leave=False)
        with open(output_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    progress_bar.update(len(chunk))
                    file.write(chunk)
        progress_bar.close()

        if output_path.endswith('.safetensor') and os.path.getsize(output_path) < 4 * 1024 * 1024:  # 4MB
            if retry_count < max_retries:
                print(f"File {output_path} is smaller than expected. Retrying download (attempt {retry_count}).")
                time.sleep(retry_delay)
                return download_file_or_image(url, output_path, retry_count + 1, max_retries)
            else:
                # Save download error log in the logs subfolder
                download_errors_log = os.path.join(LOGS_DIR, f'{username}.download_errors.log')
                with open(download_errors_log, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"Failed to download {url} after {max_retries} attempts.\n")
                return False
        return True
    except (requests.RequestException, Exception) as e:
        if retry_count < max_retries:
            print(f"Error downloading {url}: {e}. Retrying in {retry_delay} seconds (attempt {retry_count}).")
            time.sleep(retry_delay)
            return download_file_or_image(url, output_path, retry_count + 1, max_retries)
        else:
            download_errors_log = os.path.join(LOGS_DIR, f'{username}.download_errors.log')
            with open(download_errors_log, 'a', encoding='utf-8') as log_file:
                log_file.write(f"Failed to download {url} after {max_retries} attempts. Error: {e}\n")
            return False
    except (TimeoutError, ConnectionResetError) as e:
        if progress_bar:
            progress_bar.close()
        if retry_count < max_retries:
            print(f"Error during download: {url}, attempt {retry_count + 1}")
            time.sleep(retry_delay)
            return download_file_or_image(url, output_path, retry_count + 1, max_retries)
        else:
            download_errors_log = os.path.join(LOGS_DIR, f'{username}.download_errors.log')
            with open(download_errors_log, 'a', encoding='utf-8') as log_file:
                log_file.write(f"Error downloading file {output_path} from URL {url}: {e} after {max_retries} attempts\n")
            return False
    return True


def download_model_files(item_name, model_version, item, download_type, exclude_type, failed_downloads_file):
    """Download related image and model files for each model version."""
    files = model_version.get('files', [])
    images = model_version.get('images', [])
    downloaded = False
    model_id = item['id']
    model_url = f"https://civitai.com/models/{model_id}"
    # Prepend the model ID to the model name
    model_name_with_id = f"{model_id} - {item_name}"
    item_name_sanitized = sanitize_name(model_name_with_id, max_length=MAX_PATH_LENGTH)
    model_images = {}
    item_dir = None

    description = item.get('description', '')
    base_model = item.get('baseModel')
    trigger_words = model_version.get('trainedWords', [])

    for file in files:
        file_name = file.get('name', '')
        file_url = file.get('downloadUrl', '')

        # Determine subfolder based on file extension and item type
        if file_name.endswith('.zip'):
            if 'type' in item and item['type'] == 'LORA':
                subfolder = 'Lora'
            elif 'type' in item and item['type'] == 'Training_Data':
                subfolder = 'Training_Data'
            else:
                subfolder = 'Other'
        elif file_name.endswith('.safetensors'):
            if 'type' in item:
                if item['type'] == 'Checkpoint':
                    subfolder = 'Checkpoints'
                elif item['type'] == 'TextualInversion':
                    subfolder = 'Embeddings'
                elif item['type'] in ['VAE', 'LoCon']:
                    subfolder = 'Other'
                else:
                    subfolder = 'Lora'
            else:
                subfolder = 'Lora'
        elif file_name.endswith('.pt'):
            if 'type' in item and item['type'] == 'TextualInversion':
                subfolder = 'Embeddings'
            else:
                subfolder = 'Other'
        else:
            subfolder = 'Other'

        # Apply filtering based on user options:
        if download_type is not None:
            if download_type != 'All' and subfolder != download_type:
                continue
        elif exclude_type is not None:
            if subfolder == exclude_type:
                continue

        # Create folder structure using the sanitized model name (with id)
        if base_model:
            item_dir = os.path.join(OUTPUT_DIR, username, subfolder, base_model, item_name_sanitized)
            logging.info(f"Using baseModel folder structure for {item_name} (ID: {model_id}): {base_model}")
        else:
            item_dir = os.path.join(OUTPUT_DIR, username, subfolder, item_name_sanitized)
            logging.info(f"No baseModel found for {item_name} (ID: {model_id}), using standard folder structure")

        try:
            os.makedirs(item_dir, exist_ok=True)
        except OSError as e:
            logging.error(f"Error creating directory for {item_name}: {str(e)}")
            with open(failed_downloads_file, "a", encoding='utf-8') as f:
                f.write(f"Item Name: {item_name}\n")
                f.write(f"Model URL: {model_url}\n")
                f.write("---\n")
            return item_name, False, model_images

        # Write description (with HTML tags removed) and trigger words to files
        description_file = os.path.join(item_dir, "description.txt")
        # Remove HTML tags from description
        clean_description = re.sub(r'<[^>]*>', '', description)
        with open(description_file, "w", encoding='utf-8') as f:
            f.write(clean_description)

        trigger_words_file = os.path.join(item_dir, "triggerWords.txt")
        with open(trigger_words_file, "w", encoding='utf-8') as f:
            for word in trigger_words:
                f.write(f"{word}\n")

        # Append token and nsfw parameter to URL
        if '?' in file_url:
            file_url += f"&token={token}&nsfw=true"
        else:
            file_url += f"?token={token}&nsfw=true"

        file_name_sanitized = sanitize_name(file_name, item_name, max_length=MAX_PATH_LENGTH, subfolder=subfolder)
        file_path = os.path.join(item_dir, file_name_sanitized)

        if not file_name or not file_url:
            print(f"Invalid file entry: {file}")
            continue

        success = download_file_or_image(file_url, file_path)
        if success:
            downloaded = True
        else:
            with open(failed_downloads_file, "a", encoding='utf-8') as f:
                f.write(f"Item Name: {item_name}\n")
                f.write(f"File URL: {file_url}\n")
                f.write("---\n")

        details_file = os.path.join(item_dir, "details.txt")
        with open(details_file, "a", encoding='utf-8') as f:
            f.write(f"Model URL: {model_url}\n")
            f.write(f"File Name: {file_name}\n")
            f.write(f"File URL: {file_url}\n")

    if item_dir is not None:
        for image in images:
            image_id = image.get('id', '')
            image_url = image.get('url', '')

            image_filename_raw = f"{item_name}_{image_id}_for_{file_name}.jpeg"
            image_filename_sanitized = sanitize_name(image_filename_raw, item_name, max_length=MAX_PATH_LENGTH, subfolder=subfolder)
            image_path = os.path.join(item_dir, image_filename_sanitized)
            if not image_id or not image_url:
                print(f"Invalid image entry: {image}")
                continue

            success = download_file_or_image(image_url, image_path)
            if success:
                downloaded = True
            else:
                with open(failed_downloads_file, "a", encoding='utf-8') as f:
                    f.write(f"Item Name: {item_name}\n")
                    f.write(f"Image URL: {image_url}\n")
                    f.write("---\n")

            details_file = os.path.join(item_dir, "details.txt")
            with open(details_file, "a", encoding='utf-8') as f:
                f.write(f"Image ID: {image_id}\n")
                f.write(f"Image URL: {image_url}\n")

    return item_name, downloaded, model_images

def process_username(username, download_type, exclude_type=None):
    """Process a username and download the specified type of content."""
    if download_type is not None:
        print(f"Processing username: {username}, Download type: {download_type}")
    elif exclude_type is not None:
        print(f"Processing username: {username}, Excluding type: {exclude_type}")

    fetch_user_data = fetch_all_models(token, username)
    summary_data = read_summary_data(username)
    total_items = summary_data.get('Total', 0)

    if download_type is not None:
        if download_type == 'All':
            selected_type_count = total_items
            intentionally_skipped = 0
        else:
            selected_type_count = summary_data.get(download_type, 0)
            intentionally_skipped = total_items - selected_type_count
    elif exclude_type is not None:
        selected_type_count = total_items - summary_data.get(exclude_type, 0)
        intentionally_skipped = summary_data.get(exclude_type, 0)

    # Save failed downloads log in the logs subfolder.
    failed_downloads_file = os.path.join(LOGS_DIR, f"failed_downloads_{username}.txt")
    with open(failed_downloads_file, "w", encoding='utf-8') as f:
        f.write(f"Failed Downloads for Username: {username}\n\n")

    params = {
        "username": username,
        "token": token
    }
    url = f"{BASE_URL}?{urllib.parse.urlencode(params)}&nsfw=true"

    headers = {
        "Content-Type": "application/json"
    }

    initial_url = url
    next_page = url
    first_next_page = None

    while True:
        if next_page is None:
            print("End of pagination reached: 'next_page' is None.")
            break

        retry_count = 0
        max_retries = max_tries
        current_retry_delay = args.retry_delay

        while retry_count < max_retries:
            try:
                response = session.get(next_page, headers=headers)
                response.raise_for_status()
                data = response.json()
                break
            except (requests.RequestException, TimeoutError, json.JSONDecodeError) as e:
                print(f"Error making API request or decoding JSON response: {e}")
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Retrying in {current_retry_delay} seconds...")
                    time.sleep(current_retry_delay)
                else:
                    print("Maximum retries exceeded. Exiting.")
                    exit()

        items = data['items']
        metadata = data.get('metadata', {})
        next_page = metadata.get('nextPage')

        if not metadata and not items:
            print("Termination condition met: 'metadata' is empty.")
            break

        if first_next_page is None:
            first_next_page = next_page

        executor = ThreadPoolExecutor(max_workers=max_threads)
        download_futures = []
        downloaded_item_names = set()

        for item in items:
            item_name = item['name']
            model_versions = item['modelVersions']
            if item_name in downloaded_item_names:
                continue
            downloaded_item_names.add(item_name)

            for version in model_versions:
                # Include baseModel in the item dictionary
                item_with_base_model = item.copy()
                item_with_base_model['baseModel'] = version.get('baseModel')

                future = executor.submit(
                    download_model_files,
                    item_name,
                    version,
                    item_with_base_model,
                    download_type,
                    exclude_type,
                    failed_downloads_file
                )
                download_futures.append(future)

        for future in tqdm(download_futures, desc="Downloading Files", unit="file", leave=False):
            future.result()

        executor.shutdown()

    # Calculate downloaded count based on filtering option
    if download_type is not None:
        if download_type == 'All':
            downloaded_count = sum(
                len(os.listdir(os.path.join(OUTPUT_DIR, username, category)))
                for category in ['Lora', 'Checkpoints', 'Embeddings', 'Training_Data', 'Other']
                if os.path.exists(os.path.join(OUTPUT_DIR, username, category))
            )
        else:
            downloaded_count = len(os.listdir(os.path.join(OUTPUT_DIR, username, download_type))) if os.path.exists(os.path.join(OUTPUT_DIR, username, download_type)) else 0
    elif exclude_type is not None:
        categories = [cat for cat in ['Lora', 'Checkpoints', 'Embeddings', 'Training_Data', 'Other'] if cat != exclude_type]
        downloaded_count = sum(
            len(os.listdir(os.path.join(OUTPUT_DIR, username, cat)))
            for cat in categories if os.path.exists(os.path.join(OUTPUT_DIR, username, cat))
        )

    failed_count = selected_type_count - downloaded_count

    print(f"Total items for username {username}: {total_items}")
    print(f"Downloaded items for username {username}: {downloaded_count}")
    print(f"Intentionally skipped items for username {username}: {intentionally_skipped}")
    print(f"Failed items for username {username}: {failed_count}")

def categorize_item(item):
    """Categorize the item based on JSON type."""
    item_type = item.get("type", "").upper()
    file_name = item.get("name", "")

    if item_type == 'CHECKPOINT':
        return 'Checkpoints'
    elif item_type == 'TEXTUALINVERSION':
        return 'Embeddings'
    elif item_type == 'LORA':
        return 'Lora'
    elif item_type == 'TRAINING_DATA':
        return 'Training_Data'
    else:
        return 'Other'

def search_for_training_data_files(item):
    """Search for files with type 'Training Data' in the item's model versions."""
    training_data_files = []
    model_versions = item.get("modelVersions", [])
    for version in model_versions:
        for file in version.get("files", []):
            if file.get("type") == "Training Data":
                training_data_files.append(file.get("name", ""))
    return training_data_files

def fetch_all_models(token, username):
    base_url = "https://civitai.com/api/v1/models"
    categorized_items = {
        'Checkpoints': [],
        'Embeddings': [],
        'Lora': [],
        'Training_Data': [],
        'Other': []
    }
    other_item_types = []

    next_page = f"{base_url}?username={username}&token={token}&nsfw=true"
    first_next_page = None

    while next_page:
        response = requests.get(next_page)
        data = response.json()
        for item in data.get("items", []):
            try:
                # Check for top-level categorization.
                category = categorize_item(item)
                categorized_items[category].append(item.get("name", ""))

                # Check for deep nested "Training Data" files.
                training_data_files = search_for_training_data_files(item)
                if training_data_files:
                    categorized_items['Training_Data'].extend(training_data_files)

                if category == 'Other':
                    other_item_types.append((item.get("name", ""), item.get("type", None)))
            except Exception as e:
                logger_md.error(f"Error categorizing item: {item} - {e}")

        metadata = data.get('metadata', {})
        next_page = metadata.get('nextPage')
        if first_next_page is None:
            first_next_page = next_page
        # Termination condition: break if the nextPage URL repeats or metadata is empty.
        if next_page and next_page == first_next_page and next_page != next_page or not metadata:
            logger_md.error("Termination condition met: first nextPage URL repeated.")
            break

    total_count = sum(len(items) for items in categorized_items.values())

    # Write the summary file to the logs subfolder.
    summary_file_path = os.path.join(LOGS_DIR, f"{username}.txt")
    with open(summary_file_path, "w", encoding='utf-8') as file:
        file.write("Summary:\n")
        file.write(f"Total - Count: {total_count}\n")
        for category, items in categorized_items.items():
            file.write(f"{category} - Count: {len(items)}\n")
        file.write("\nDetailed Listing:\n")

        # Write the detailed listing.
        for category, items in categorized_items.items():
            file.write(f"{category} - Count: {len(items)}\n")
            if category == 'Other':
                for item_name, item_type in other_item_types:
                    file.write(f"{category} - Item: {item_name} - Type: {item_type}\n")
            else:
                for item_name in items:
                    file.write(f"{category} - Item: {item_name}\n")
            file.write("\n")

    return categorized_items


if __name__ == "__main__":
    for username in usernames:
        process_username(username, download_type, exclude_type)
