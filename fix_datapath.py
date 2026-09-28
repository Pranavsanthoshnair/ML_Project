"""Fix DATA_PATH in Project_Code.ipynb to point to data/raw/train.csv"""
import json, os

NB_PATH = os.path.join(os.path.dirname(__file__),
                       "notebooks", "Project_Code.ipynb")

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "DATA_PATH" in src and "train.csv" in src:
        cell["source"] = [
            "import os\n",
            "\n",
            "# train.csv lives at data/raw/train.csv relative to the project root.\n",
            "# Launch Jupyter from the ML_Project folder (cd ML_Project, then jupyter notebook).\n",
            "DATA_PATH = os.path.join(os.getcwd(), \"data\", \"raw\", \"train.csv\")\n",
            "\n",
            "# Fallback: uncomment and edit the line below if the above path doesn't work:\n",
            "# DATA_PATH = r\"C:\\Users\\PRANAV\\ML_Project\\data\\raw\\train.csv\"\n",
            "\n",
            "df = pd.read_csv(DATA_PATH)\n",
            "\n",
            "print(\"Dataset shape:\", df.shape)\n",
            "display(df.head())",
        ]
        print("DATA_PATH cell updated.")
        break

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Notebook saved.")
