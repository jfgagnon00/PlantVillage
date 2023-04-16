import os


def array_to_str(a):
    return " ".join(a)


FINAL_REPORT = "final_report.ipynb"

notebooks_to_merge = [
    "00 - TOC.ipynb",
    "01 - Exploration.ipynb",
    "02 - Features Extraction.ipynb",
    "03 - Train Test Split.ipynb",
    "04 - Codebook.ipynb",
    "05 - Image To Visual Words.ipynb",
    "06 - Training.ipynb",
    ]

merge_command = [
    "nbmerge",
    "-o",
    FINAL_REPORT,
    "-v",
]

notebook_paths = [f'"{d}"' for d in notebooks_to_merge]
command_to_execute = array_to_str(merge_command + notebook_paths)

# fusionner les documents dans 1 seul notebook
os.system(command_to_execute)

# generer la table des matieres
if False:
    # ne fonctionne pas
    toc_command = [
        "jupyter",
        "nbconvert",
        FINAL_REPORT,
        "--template", "toc2",
        "--to", "notebook",
        "--output", "dude.ipynb"
    ]
    os.system( array_to_str(toc_command) )