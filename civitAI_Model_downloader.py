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

# Set up logging using our custom logger.
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

# Prompt for token if not provided.
if args.token is None:
    args.token = input("Please enter your Civitai API token: ")

# Determine filtering options.
download_type = None
exclude_type = None
if args.download_type:
    download_type = args.download_type
elif args.exclude_type:
    exclude_type = args.exclude_type
else:
    download_type = 'All'

# Initialize variables.
usernames = args.usernames
retry_delay = args.retry_delay
max_tries = args.max_tries
max_threads = args.max_threads
token = args.token

def sanitize_directory_name(name):
    return name.rstrip()

# Create output directory.
OUTPUT_DIR = sanitize_directory_name(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create a persistent session.
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
    base_name = re.sub(r'[<>:"/\\|?*\x00-\x1f\x7f-\x9f]', '_', base_name)
    reserved_names = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}
    if base_name.upper() in reserved_names:
        base_name = '_'
    base_name = re.sub(r'__+', '_', base_name).strip('_.')
    if subfolder and output_dir and username:
        path_length = len(os.path.join(output_dir, username, subfolder))
        max_base_length = max_length - len(extension) - path_length
        base_name = base_name[:max_base_length].rsplit('_', 1)[0]
    sanitized_name = base_name + extension
    return sanitized_name.strip()

def download_file_or_image(url, output_path, username, retry_count=0, max_retries=max_tries):
    """Download a file or image from the URL, adjusting the file extension if needed."""
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
        content_type = response.headers.get('Content-Type', '')
        if 'image' in content_type:
            file_extension = '.jpg'
        elif 'video' in content_type:
            file_extension = '.mp4'
        else:
            file_extension = os.path.splitext(output_path)[1]
        output_path = os.path.splitext(output_path)[0] + file_extension
        total_size = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, leave=False)
        with open(output_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    progress_bar.update(len(chunk))
                    file.write(chunk)
        progress_bar.close()
        if output_path.endswith('.safetensor') and os.path.getsize(output_path) < 4 * 1024 * 1024:
            if retry_count < max_retries:
                print(f"File {output_path} is smaller than expected. Retrying (attempt {retry_count}).")
                time.sleep(retry_delay)
                return download_file_or_image(url, output_path, username, retry_count + 1, max_retries)
            else:
                download_errors_log = os.path.join(LOGS_DIR, f'{username}.download_errors.log')
                with open(download_errors_log, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"Failed to download {url} after {max_retries} attempts.\n")
                return False
        return True
    except (requests.RequestException, Exception) as e:
        if retry_count < max_retries:
            print(f"Error downloading {url}: {e}. Retrying in {retry_delay} seconds (attempt {retry_count}).")
            time.sleep(retry_delay)
            return download_file_or_image(url, output_path, username, retry_count + 1, max_retries)
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
            return download_file_or_image(url, output_path, username, retry_count + 1, max_retries)
        else:
            download_errors_log = os.path.join(LOGS_DIR, f'{username}.download_errors.log')
            with open(download_errors_log, 'a', encoding='utf-8') as log_file:
                log_file.write(f"Error downloading file {output_path} from URL {url}: {e} after {max_retries} attempts\n")
            return False
    return True

def categorize_item(item):
    """Categorize the item based on its type."""
    item_type = item.get("type", "").upper()
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

def download_model_files(username, item_name, model_version, item, download_type, exclude_type, failed_downloads_file):
    """Download all files for one model version, saving them in a version subfolder.
    The preview image is saved in the version folder, and all additional images are saved in an 'examples' subfolder.
    """
    model_id = item['id']
    model_id_formatted = f"{model_id:07d}"
    model_name_with_id = f"{model_id_formatted} - {item_name}"
    item_name_sanitized = sanitize_name(model_name_with_id, max_length=MAX_PATH_LENGTH)
    
    # Use the model’s primary category for the parent folder.
    primary_category = categorize_item(item)
    base_model = item.get('baseModel')
    if base_model:
        model_folder = os.path.join(OUTPUT_DIR, username, primary_category, base_model, item_name_sanitized)
    else:
        model_folder = os.path.join(OUTPUT_DIR, username, primary_category, item_name_sanitized)
    
    # Create a subfolder for this version using the version's "name" field.
    version_folder_raw = model_version.get('name', 'Version Unknown')
    version_folder = sanitize_name(version_folder_raw)
    final_dir = os.path.join(model_folder, version_folder)
    os.makedirs(final_dir, exist_ok=True)
    
    model_url = f"https://civitai.com/models/{model_id}"
        
    # Determine a base file name from the first file entry (e.g. "kyl13-001" from "kyl13-001.pt").
    base_file_name = None
    files = model_version.get('files', [])
    for file in files:
        file_name = file.get('name', '')
        if file_name:
            base_file_name = os.path.splitext(file_name)[0]
            break
    
    downloaded = False
    # Download all model files.
    for file in files:
        file_name = file.get('name', '')
        file_url = file.get('downloadUrl', '')
        if not file_name or not file_url:
            print(f"Invalid file entry: {file}")
            continue
        if '?' in file_url:
            file_url += f"&token={token}&nsfw=true"
        else:
            file_url += f"?token={token}&nsfw=true"
        file_name_sanitized = sanitize_name(file_name, item_name, max_length=MAX_PATH_LENGTH)
        file_path = os.path.join(final_dir, file_name_sanitized)
        success = download_file_or_image(file_url, file_path, username)
        if success:
            downloaded = True
        else:
            with open(failed_downloads_file, "a", encoding='utf-8') as f:
                f.write(f"Item Name: {item_name}\nFile URL: {file_url}\n---\n")
    
    # Download the preview image.
    preview_url_used = None
    preview_filename = ""
    if base_file_name:
        preview_filename = f"{base_file_name}.preview.jpg"
    else:
        preview_filename = f"{item_name_sanitized}.preview.jpg"
    preview_path = os.path.join(final_dir, preview_filename)
    
    images = model_version.get('images', [])
    for image in images:
        if image.get("type", "image").lower() == "image":
            preview_url = image.get("url", "")
            if preview_url:
                if download_file_or_image(preview_url, preview_path, username):
                    preview_url_used = preview_url
                break  # Only use the first valid image as preview.
    
    # Create an "examples" subfolder for the other images.
    examples_dir = os.path.join(final_dir, "examples")
    os.makedirs(examples_dir, exist_ok=True)
    
    # Download remaining images (that are not the preview) into the examples folder.
    for image in images:
        image_url = image.get("url", "")
        if not image_url:
            continue
        # Skip the image if it was used as preview.
        if preview_url_used and image_url == preview_url_used:
            continue
        image_id = image.get('id', '')
        image_filename_raw = f"{item_name}_{image_id}.jpeg"
        image_filename_sanitized = sanitize_name(image_filename_raw, item_name, max_length=MAX_PATH_LENGTH)
        image_path = os.path.join(examples_dir, image_filename_sanitized)
        if not image_id or not image_url:
            print(f"Invalid image entry: {image}")
            continue
        success = download_file_or_image(image_url, image_path, username)
        if success:
            downloaded = True
        else:
            with open(failed_downloads_file, "a", encoding='utf-8') as f:
                f.write(f"Item Name: {item_name}\nImage URL: {image_url}\n---\n")
    
    # Save the info file with the model's JSON using the base file name.
    if base_file_name:
        info_filename = f"{base_file_name}.civitai.info"
    else:
        info_filename = f"{item_name_sanitized}.civitai.info"
    info_path = os.path.join(final_dir, info_filename)
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(item, f, indent=4)
    
    return item_name, downloaded, {}





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
    
    failed_downloads_file = os.path.join(LOGS_DIR, f"failed_downloads_{username}.txt")
    with open(failed_downloads_file, "w", encoding='utf-8') as f:
        f.write(f"Failed Downloads for Username: {username}\n\n")
    
    params = {
        "username": username,
        "token": token
    }
    url = f"{BASE_URL}?{urllib.parse.urlencode(params)}&nsfw=true"
    
    headers = {"Content-Type": "application/json"}
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
                item_with_base_model = item.copy()
                item_with_base_model['baseModel'] = version.get('baseModel')
                future = executor.submit(
                    download_model_files,
                    username,
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

def search_for_training_data_files(item):
    """Search for files with type 'Training Data' in the model versions."""
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
                category = categorize_item(item)
                categorized_items[category].append(item.get("name", ""))
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
        if next_page and next_page == first_next_page and next_page != next_page or not metadata:
            logger_md.error("Termination condition met: first nextPage URL repeated.")
            break
    total_count = sum(len(items) for items in categorized_items.values())
    summary_file_path = os.path.join(LOGS_DIR, f"{username}.txt")
    with open(summary_file_path, "w", encoding='utf-8') as file:
        file.write("Summary:\n")
        file.write(f"Total - Count: {total_count}\n")
        for category, items in categorized_items.items():
            file.write(f"{category} - Count: {len(items)}\n")
        file.write("\nDetailed Listing:\n")
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
