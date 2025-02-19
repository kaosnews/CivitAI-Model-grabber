# civitAI Model Downloader

This Python script downloads models, images, and associated data from the civitAI API based on one or more provided usernames. It supports various filtering options and organizes downloads into a clear, hierarchical folder structure.

## Features

- **Multi-User Support:** Supply one or more usernames from which to download models.
- **Filtering Options:**  
  - Use `--download_type` to download only a specific type (e.g., Lora, Checkpoints, Embeddings, Training_Data, Other, or All).  
  - Use `--exclude_type` to download all content except a specified type.
- **Folder Structure:**  
  - Models are grouped by username and primary category (determined by the model type).  
  - If available, the base model is included as an additional grouping level.
  - **Version Subfolders:**  
    Each model may have multiple versions. For each version, a subfolder is created _using the version’s “name” field. This keeps different iterations of a model neatly separated.
- **Image Organization:**  
  - The first valid image is downloaded as the preview and saved in the version folder.
  - All additional images (examples) are stored in an `examples` subfolder within the version folder.
- **Logging:**  
  - All logs (e.g., download errors and summary files) are saved in a dedicated `logs` folder.

## Usage

1. **Command-Line Arguments:**
   - **Usernames:** Provide one or more usernames (positional arguments).
   - **Token:** Use `--token` to provide your civitAI API token (if not provided, you will be prompted).
   - **Filtering:** Use either `--download_type` or `--exclude_type` (mutually exclusive) to control what content to download.
   - **Retries, Threads, etc.:** Other options include `--retry_delay`, `--max_tries`, and `--max_threads`.

2. **Example Command:**

   ```bash
   python civitAI_Model_downloader.py username1 username2 --token YOUR_API_TOKEN --download_type Checkpoints
   ```

3. **Output Structure:**

   The downloaded content is organized under the `model_downloads` folder following this hierarchy:

   ```
   model_downloads/
       └── username/
           └── PrimaryCategory/        # (e.g., Checkpoints, Lora, etc.)
               └── [BaseModel]/        # (if available)
                   └── "ID - ModelName"/
                       └── VersionName/  # e.g., "Version v1.0"
                           ├── mytiname-001.pt               # Model file(s)
                           ├── mytiname-001.preview.jpg      # Preview image (first valid image)
                           ├── examples/                  # Other example images
                           │    ├── <other_image_files>.jpeg
                           └── mytiname.civitai.info     # Full JSON info file
   ```

4. **Logs:**

   All log files (download errors, summaries, etc.) are saved in the `logs` subfolder.