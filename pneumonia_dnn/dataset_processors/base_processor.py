"""
Dataset metadata processor base
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List

class ImageLabel(Enum):
    """
    Image Label

    Args:
        Enum (_type_): Enum
    """
    NOT_CLASSIFIED = 0
    NO_PNEUMONIA = 1
    VIRAL_PNEUMONIA = 2
    BACTERIAL_PNEUMONIA = 3

@dataclass
class ProcessedImage():
    """
    Standardized Result of preprocessed data
    """
    image_path: str
    has_pneumonia: bool
    label: ImageLabel
    is_training: bool

class DatasetProcessorBase(ABC):
    """
    Dataset processor base

    Args:
        ABC (_type_): Abstract Base Class
    """
    @abstractmethod
    def is_multi_label(self) -> bool:
        """
        Returns true if the dataset contains bacterial vs viral
        labels of the pneumonia data

        Returns:
            bool: True if viral/bacterial labels exist
        """
        pass

    @abstractmethod
    def get_output_path(self, base_output_path: str) -> str:
        """
        Get the output path to where the dataset was saved 

        Returns:
            str: Path to where the dataset was saved
        """
        pass

    @abstractmethod
    def get_dataset_identifier(self) -> str:
        """
        Gets the kaggle identifier of the dataset
        (i.e., paultimothymooney/chest-xray-pneumonia)
        Returns:
            str: _description_
        """
        pass
    
    @abstractmethod
    def get_images(self,
                   base_output_path: str,
                   num_of_pneumonia_train: int,
                num_of_nonpneumonia_train: int,
                num_of_pneumonia_test: int, 
                num_of_nonpneumonia_test: int)  -> List[ProcessedImage]:
        pass
