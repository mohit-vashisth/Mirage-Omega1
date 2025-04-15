import os
import shutil  # Yeh import kar le

folder_path = r"C:\Users\Admin\Documents\Mirage-Omega1\ml_models\ete-synthesis\audios"
new_folder_path = r"C:\Users\Admin\Documents\Mirage-Omega1\ml_models\ete-synthesis\processed_audios"
new_name = "mirage_"

files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

files.sort()

for i, file in enumerate(files, start=1):
    old_file_path = os.path.join(folder_path, file)
    _, ext = os.path.splitext(file)
    new_file_name = f"{new_name}{i}{ext}"
    new_file_path = os.path.join(new_folder_path, new_file_name)
    
    shutil.copy(old_file_path, new_file_path)
    print(f"Copied and renamed {file} to {new_file_name}")