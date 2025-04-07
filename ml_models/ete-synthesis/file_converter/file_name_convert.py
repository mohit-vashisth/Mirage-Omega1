import os

folder_path = r"D:\Mirage-Omega1\clips"
new_name = "mirage_"

files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

files.sort()

for i, file in enumerate(files, start=1):
    old_file_path = os.path.join(folder_path, file)
    _, ext = os.path.splitext(file)
    new_file_name = f"{new_name}{i}{ext}"
    new_file_path = os.path.join(folder_path, new_file_name)
    
    os.rename(old_file_path, new_file_path)
    print(f"Renamed {file} to {new_file_name}")
