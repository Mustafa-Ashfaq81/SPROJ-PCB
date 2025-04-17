import os


def rename_files_in_directory():
    for filename in os.listdir("."):
        if filename.endswith(".txt") and "ODI Series" in filename:
            # Replace " ODI Series_" with just "_"
            new_filename = filename.replace(" ODI Series_", "_")
            os.rename(filename, new_filename)
            print(f"Renamed: {filename} -> {new_filename}")


if __name__ == "__main__":
    rename_files_in_directory()
