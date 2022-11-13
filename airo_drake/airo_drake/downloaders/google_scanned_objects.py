import os
import shutil
import tempfile
import zipfile

import requests


def move_texture_png(model_directory):
    texture_path = os.path.join(model_directory, "materials", "textures", "texture.png")
    texture_destination_path = os.path.join(model_directory, "meshes", "texture.png")
    if not os.path.exists(texture_destination_path):
        shutil.move(texture_path, texture_destination_path)


def download_model_zip_shutil(model_zip, model_zip_filepath_local):
    owner_name = "GoogleResearch"
    base_url = "https://fuel.gazebosim.org/"
    fuel_version = "1.0"
    model_zip_download_url = f"{base_url}/{fuel_version}/{owner_name}/models/{model_zip}"

    with requests.get(model_zip_download_url, stream=True) as r:
        with open(model_zip_filepath_local, "wb") as f:
            shutil.copyfileobj(r.raw, f)


def download_model_zip(model_zip, model_zip_filepath_local):
    owner_name = "GoogleResearch"
    base_url = "https://fuel.gazebosim.org/"
    fuel_version = "1.0"
    model_zip_download_url = f"{base_url}/{fuel_version}/{owner_name}/models/{model_zip}"
    response = requests.get(model_zip_download_url, stream=True)

    with open(model_zip_filepath_local, "wb") as fd:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            fd.write(chunk)


def unzip_model(model_zip_filepath_local, model_directory):
    with zipfile.ZipFile(model_zip_filepath_local, "r") as zip_ref:
        zip_ref.extractall(model_directory)


def get_google_scanned_object(model_name, save_directory=None, move_texture=True):
    if save_directory is None:
        save_directory = tempfile.gettempdir()

    model_zip = f"{model_name}.zip"
    model_zip_filepath_local = os.path.join(save_directory, model_zip)
    model_directory = os.path.join(save_directory, model_name)

    if os.path.exists(model_directory):
        print(f"{model_directory} already exists, not redownloading.")
        return model_directory

    download_model_zip(model_zip, model_zip_filepath_local)

    model_directory = os.path.join(save_directory, model_name)
    unzip_model(model_zip_filepath_local, model_directory)
    move_texture_png(model_directory)
    return model_directory
