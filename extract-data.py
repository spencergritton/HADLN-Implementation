import zipfile

# Extracts data.zip to data folder
with zipfile.ZipFile('./data.zip', 'r') as zip_ref:
    zip_ref.extractall('./data')