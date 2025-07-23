import os
import shutil
from tkinter import filedialog
from tqdm import tqdm


tower_path = filedialog.askdirectory(title="Select the labels directory")
os.makedirs(os.path.join(tower_path, "T"), exist_ok=True)
os.makedirs(os.path.join(tower_path, "V"), exist_ok=True)
list_files = os.listdir(tower_path)

for file in tqdm(list_files, desc="Procesando archivos"):
    #eliminar extension
    if file.endswith(".JPG"):
        filename = file.split(".")[0]
        termicas = filename.split("_")

        termica = termicas[(len(termicas)-2)]

        
        if termica == "T":
            shutil.move(os.path.join(tower_path, file), os.path.join(tower_path, "T", file))
        else:
            shutil.move(os.path.join(tower_path, file), os.path.join(tower_path, "V", file))