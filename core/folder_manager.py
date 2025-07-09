import tkinter as tk
from tkinter import filedialog

class FolderManager:
    
    # Función para seleccionar múltiples directorios
    def select_directories():
        
        list_folders = []
        path_root = filedialog.askdirectory(title='Seleccione el directorio raíz')
        while path_root:
            list_folders.append(path_root)
            print(f"Directorio seleccionado: {path_root}")
            path_root = filedialog.askdirectory(title='Seleccione otro directorio o cancele para continuar')
        if not list_folders:
            raise Exception("No se seleccionó ningún directorio")
        
        print(f"Se han seleccionado {len(list_folders)} directorios")
        return list_folders