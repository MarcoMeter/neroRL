import os

def get_all_files_in_directory(root_folder):
    all_files = []
    
    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            all_files.append(os.path.join(foldername, filename))
            
    return all_files

root_folder = "results"
file_paths = get_all_files_in_directory(root_folder)

# save the paths to a .txt file
with open("file_paths.txt", "w") as f:
    for path in file_paths:
        f.write(path + "\n")

print("Paths have been saved to file_paths.txt")
