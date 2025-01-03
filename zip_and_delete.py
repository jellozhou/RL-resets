import os
import zipfile
import argparse
import time
from datetime import datetime, timedelta

# function to create a zip file of all files in results/ over T time old, and then delete these files
# use to not violate the file number quota on della
def zip_and_delete(directory, time_limit):
    current_time = time.time()

    # create a zip file to store the old files
    # mark zip file with current age
    zip_filename = f"old_files_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                file_mod_time = os.path.getmtime(file_path)
                
                # If the file is older than the time limit, add it to the zip file
                if current_time - file_mod_time > time_limit:
                    zipf.write(file_path, os.path.relpath(file_path, directory))
                    os.remove(file_path)  # Delete the file after zipping it

    print(f"Zipped and deleted files older than the specified time. Zip file: {zip_filename}")

def parse_time(time_str):
    hours, minutes, seconds = map(int, time_str.split(':'))
    return timedelta(hours=hours, minutes=minutes, seconds=seconds).total_seconds()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zip and delete files older than a given time in the 'results/' directory.")
    parser.add_argument('--time', type=str, required=True, help="Time limit in the format HH:MM:SS.")
    args = parser.parse_args()
    time_limit_seconds = parse_time(args.time)

    # directory containing the files
    results_directory = 'results/'

    # ensure the directory exists
    if not os.path.exists(results_directory):
        print(f"Directory {results_directory} does not exist.")
    else:
        zip_and_delete(results_directory, time_limit_seconds)