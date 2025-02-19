# CivitAI Model Downloader

CivitAI Model Downloader is a Python script that downloads model files and related images from CivitAI based on one or more provided usernames. The downloaded content is organized into a well-structured directory tree, and detailed metadata is stored in text files. The script supports multiple download types (Lora, Checkpoints, Embeddings, Training Data, Other, or All) and only downloads files that aren’t already present, ensuring that only new models are fetched when re-run.

## Features

- **Selective Downloading**:  
  Downloads only new files that aren’t already present in the user's folder. Running the script again will fetch only the newly uploaded models.

- **Customizable Download Types**:  
  Supports filtering by specific content types:
  - `--download_type`: Download only the specified type (e.g., Lora, Checkpoints, etc.).
  - `--exclude_type`: Download everything except the specified type.

- **Enhanced Description Files**:  
  The model description is cleaned of HTML tags and saved as a plain text file (`description.txt`), making it easier to read and process.

- **Detailed Logging in a Dedicated Subfolder**:  
  All log files are now stored in a `logs` subfolder:
  - **Main Log**: `civitAI_Model_downloader.txt` contains execution details and errors.
  - **Failed Downloads Log**: `failed_downloads_<username>.txt` stores any download errors.
  - **Helper Script Error Log**: `fetch_all_models_ERROR_LOG.txt` logs errors from the helper script.

- **Organized File Structure with Unique Model Folders**:  
  Each model is saved in a uniquely named folder (with the model ID prepended to the model name, e.g., `4576 - ModelName`) under directories corresponding to the download type.

## File Structure

The downloaded files are organized as follows:

```
model_downloads/
├── username1/
│   ├── Lora/
│   │   ├── SDXL 1.0/
│   │   │   └── 4576 - ModelName/
│   │   │       ├── file1.safetensors
│   │   │       ├── image1.jpeg
│   │   │       ├── details.txt
│   │   │       ├── triggerWords.txt
│   │   │       └── description.txt
│   │   └── SD 1.5/
│   │       └── 7890 - AnotherModel/
│   │           ├── file2.safetensors
│   │           ├── image2.jpeg
│   │           ├── details.txt
│   │           ├── triggerWords.txt
│   │           └── description.txt
│   ├── Checkpoints/
│   │   ├── FLUX/
│   │   │   └── 1234 - CheckpointModel/
│   │   │       ├── file.safetensors
│   │   │       ├── image.jpeg
│   │   │       ├── details.txt
│   │   │       ├── triggerWords.txt
│   │   │       └── description.txt       
│   ├── Embeddings/
│   ├── Training_Data/
│   └── Other/
└── username2/
    ├── Lora/
    ├── Checkpoints/
    ├── Embeddings/
    ├── Training_Data/
    └── Other/
```

## Example of `details.txt`

```
Model URL: https://civitai.com/models/ID
File Name: ModelName.ending
File URL: https://civitai.com/api/download/models/ID
Image ID: ID
Image URL: https://image.civitai.com/Random_characters/width=450/ID.jpeg
```

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/CivitAI-Model-Downloader.git
   cd CivitAI-Model-Downloader
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script by providing one or more CivitAI usernames:
```bash
python civitAI_Model_downloader.py username1 username2 --token YOUR_API_TOKEN --download_type Checkpoints
```
Or use the `--exclude_type` option to download everything except a specific type:
```bash
python civitAI_Model_downloader.py username1 --token YOUR_API_TOKEN --exclude_type Embeddings
```

### Additional Arguments

- `--retry_delay` (default: 10 seconds)  
  Delay between retry attempts.

- `--max_tries` (default: 3)  
  Maximum number of retries per download.

- `--max_threads` (default: 5)  
  Maximum number of concurrent threads. (Too many threads may lead to API failures.)

If no token is provided, the script will prompt you for one.

## API Key

To download models that are not public, you need an API token. Generate your API key in your CivitAI account settings under the API section.

## Updates & Bugfixes

- **Logging Enhancements:**  
  All logs (execution, errors, and failed downloads) are now saved in a dedicated `logs` subfolder for easier organization.

- **Improved Description Files:**  
  Model descriptions are now saved as plain text (`description.txt`), with HTML tags removed for clarity.

- **Unique Model Folder Naming:**  
  Model IDs are now prepended to the model names in the folder structure to avoid naming conflicts.
  
- **Images and Videos saved correctly:**
  It now ensures that the files are saved with the appropriate extensions based on their content type.

---

Enjoy using the CivitAI Model Downloader! Contributions, suggestions, and bug reports are welcome.