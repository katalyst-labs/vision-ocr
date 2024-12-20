import io
import os
import shutil

import pandas as pd
from google.cloud import vision_v1
from tqdm import tqdm


class ImageTextExtractor:
    def __init__(self, credentials_file: str, data_dir: str, output_dir: str) -> None:
        """Initialize the ImageTextExtractor class.

        Args:
            credentials_file (str): Path to the Google Cloud credentials JSON file.
            data_dir (str): Directory containing input data.
            output_dir (str): Directory to save output data.
        """
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_file
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.client = vision_v1.ImageAnnotatorClient()

    def extract_text_from_image(self, image_file: str) -> str:
        """Extract text from an image using Google Vision API.

        Args:
            image_file (str): Path to the image file.

        Returns:
            str: Extracted text or an empty string if no text is found.
        """
        with io.open(image_file, "rb") as image_file_object:
            image = vision_v1.Image(content=image_file_object.read())

        response = self.client.text_detection(image=image)
        annotations = response.text_annotations

        if annotations:
            return annotations[0].description
        return ""

    def process_image(self, image_path: str) -> pd.DataFrame:
        """Process a single image to extract its text.

        Args:
            image_path (str): Path to the image file.

        Returns:
            pd.DataFrame: A DataFrame containing the image path and extracted text.
        """
        extracted_text = self.extract_text_from_image(image_path)
        return pd.DataFrame({"image_url": [image_path], "raw_text": [extracted_text]})

    def process_folder(self, folder_path: str) -> pd.DataFrame:
        """Process all images in a folder and return a DataFrame of results.

        Args:
            folder_path (str): Path to the folder containing images.

        Returns:
            pd.DataFrame: A DataFrame containing image paths and their extracted text.
        """
        data_frames = []
        for subfolder in tqdm(os.listdir(folder_path), desc="Processing folders"):
            subfolder_path = os.path.join(folder_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            for file_name in os.listdir(subfolder_path):
                if not file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    continue

                image_path = os.path.join(subfolder_path, file_name)
                try:
                    data_frames.append(self.process_image(image_path))
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

        if data_frames:
            return pd.concat(data_frames, ignore_index=True)
        return pd.DataFrame(columns=["image_url", "raw_text"])

    def clean_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate image URLs from the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with duplicates removed.
        """
        return df.drop_duplicates(subset="image_url", keep="first").reset_index(drop=True)

    def save_to_excel(self, df: pd.DataFrame, output_file: str):
        """Save the DataFrame to an Excel file.

        Args:
            df (pd.DataFrame): DataFrame to save.
            output_file (str): Path to the output Excel file.
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_excel(output_file, index=False)

    def execute(self):
        """Main execution method to process all folders and save results."""
        for folder in tqdm(os.listdir(self.data_dir), desc="Processing main folders"):
            folder_path = os.path.join(self.data_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            print(f"Processing folder: {folder}")
            try:
                result_df = self.process_folder(folder_path)
                result_df = self.clean_duplicates(result_df)

                output_file = os.path.join(self.output_dir, f"{folder}_raw_text_extracted.xlsx")
                self.save_to_excel(result_df, output_file)

                shutil.rmtree(folder_path)
            except Exception as e:
                print(f"Error processing folder {folder}: {e}")


extractor = ImageTextExtractor(
    credentials_file="token.json",
    data_dir="Data/lot/",
    output_dir="Data Extracted Raw/"
)
extractor.execute()
