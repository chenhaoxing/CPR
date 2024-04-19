import os

from .utils import Datum, DatasetBase
from .oxford_pets import OxfordPets


template = ['a photo of a {}.']


class Caltech101(DatasetBase):
    dataset_dir = 'caltech-101'
    def __init__(self, root, num_shots, subsample):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, '101_ObjectCategories')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_Caltech101.json')
        self.cupl_path = os.path.join('/ossfs/workspace/CPR/gpt3_prompts', 'CuPL_prompts_caltech101.json')

        self.template = template

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        val = self.generate_fewshot_dataset(val,num_shots=min(num_shots, 4))
        print("Sample from "+subsample)
        train, val, test = OxfordPets.subsample_classes(train,
                                                  val,
                                                  test,
                                                  subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)