"""
Coronahack dataset processor
"""
from typing import List
import pandas as pd

from pneumonia_dnn.dataset_processors.base_processor import DatasetProcessorBase, ImageLabel, ProcessedImage

class CoronahackProcessor(DatasetProcessorBase):
    """
    Dataset processor base
    """

    def is_multi_label(self) -> bool:
        """
        Returns true if the dataset contains bacterial vs viral
        labels of the pneumonia data

        Returns:
            bool: True if viral/bacterial labels exist
        """
        return True

    def get_output_path(self, base_output_path: str) -> str:
        """
        Get the output path to where the dataset was saved 

        Returns:
            str: Path to where the dataset was saved
        """
        return f"{base_output_path}/coronahack"

    def get_dataset_identifier(self) -> str:
        """
        Gets the kaggle identifier of the dataset
        (i.e., paultimothymooney/chest-xray-pneumonia)
        Returns:
            str: _description_
        """
        return "praveengovi/coronahack-chest-xraydataset"

    def get_images(self,
                base_output_path: str,
                num_of_pneumonia_train: int, 
                num_of_nonpneumonia_train: int,
                num_of_pneumonia_test: int, 
                num_of_nonpneumonia_test: int) -> List[ProcessedImage]:
        """
        Preprocess images

        Args:
            base_output_path (str): Base output path
            num_of_pneumonia_train (int): Total number of pneumonia images to
                                        pull from the dataset for trainining
            num_of_nonpneumonia_train (int): Total number of non-pneumonia images to 
                                        pull from the dataset for training
            num_of_pneumonia_test (float): Total number of pneumonia images to
                                        pull from the dataset for testing
            num_of_nonpneumonia_test (float): Total number of non-pneumonia images to
                                        pull from the dataset for testing

        Returns:
            List[ProcessedImage]: List of preprocessed images
        """
        output_path = self.get_output_path(base_output_path)
        metadata_df = pd.read_csv(f"{output_path}/Chest_xray_Corona_Metadata.csv")
        normal_df = metadata_df[metadata_df["Label"] == "Normal"]
        pneumonia_df = metadata_df[metadata_df["Label"] != "Normal"]

        processed_images = self.__get_processed_images(
            pneumonia_df.sample(num_of_pneumonia_train),
            True, output_path)
        processed_images.extend(
            self.__get_processed_images(normal_df.sample(num_of_nonpneumonia_train),
            True, output_path))
        processed_images.extend(
            self.__get_processed_images(pneumonia_df.sample(num_of_pneumonia_test),
            False, output_path))
        processed_images.extend(
            self.__get_processed_images(normal_df.sample(num_of_nonpneumonia_test),
            False, output_path))

        return processed_images

    def __get_processed_images(self, df: pd.DataFrame,
                               is_training: bool, output_path: str) -> List[ProcessedImage]:
        df_list = df.values.tolist()
        result: List[ProcessedImage] = []
        for row in df_list:
            file_name = row[1]
            label_str = row[2]
            dataset_type = row[3]
            category_str = row[5]
            has_pneumonia = True

            if label_str in ("Normal", "normal"):
                category = ImageLabel.NO_PNEUMONIA
                has_pneumonia= False
            elif category_str in ("Virus", "virus"):
                category =  ImageLabel.VIRAL_PNEUMONIA
            elif category_str in ("bacteria", "Bacteria"):
                category = ImageLabel.BACTERIAL_PNEUMONIA
            else:
                category = ImageLabel.NOT_CLASSIFIED

            if dataset_type in ("TRAIN", "train"):
                sub_path = "train"
            else:
                sub_path = "test"

            image_path = (f"{output_path}/Coronahack-Chest-XRay-Dataset"
                        + f"/Coronahack-Chest-XRay-Dataset/{sub_path}/{file_name}")

            result.append(ProcessedImage(image_path=image_path,
                                        has_pneumonia=has_pneumonia,
                                        label=category,
                                        is_training=is_training))
        return result
