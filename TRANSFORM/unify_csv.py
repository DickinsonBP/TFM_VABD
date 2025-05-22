import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def unify_csv(csv_path=None, out_path=None):
    df = pd.DataFrame()
    if os.path.exists(csv_path):
        for dir in os.listdir(csv_path):
            year = dir
            dir_path = os.path.join(csv_path, dir)
            if os.path.isdir(dir_path):
                for file in os.listdir(dir_path):
                    if file.startswith("pDIA") and file.endswith("PlAv.csv"):
                        file_path = os.path.join(dir_path, file)
                        aux = pd.read_csv(file_path, sep=";")
                        aux["Temporada"] = year
                        cols = list(aux.columns)
                        cols.insert(1, cols.pop(cols.index("Temporada")))
                        aux = aux[cols]
                        df = pd.concat([df, aux], ignore_index=True)
        # do transformations
        df["Nombre"] = df["Nombre"].astype(str).str.replace(r"^b'|\'$", "", regex=True)
        df["Equipo"] = df["Equipo"].astype(str).str.replace(r"^\['|'\]$", "", regex=True)
        df.to_csv(out_path, index=False, sep=";")
    else:
        print("This path does not exists")
    

if __name__ == "__main__":
    unify_csv(os.getenv("EXTRACT_DATA_PATH"),"TRANSFORM/DATA/LFENDESA_UnifiedData.csv")