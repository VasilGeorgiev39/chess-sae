# %%
%load_ext autoreload
%autoreload 2
import sys
import os
from leela_interp.tools.activations import ActivationCache
# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from dictionary_learning.dictionary import AutoEncoder
from dictionary_learning.trainers import TrainerTopK, TrainerBatchTopK
from dictionary_learning.training import trainSAE

import torch
from torch.utils.data import TensorDataset, DataLoader
# %%
activations = ActivationCache.load(os.path.join(os.path.dirname(__file__), '.', 'notebooks', 'residual_activations.zarr'))
# %%
layer = 13
# %%
l1_activations = activations[f'encoder{layer}/ln2']
# %%

# Create a PyTorch dataset from l1_activations
class ZarrDataset(torch.utils.data.Dataset):
    def __init__(self, zarr_array):
        # Convert Zarr array to numpy array before creating torch tensor
        numpy_array = zarr_array[:]
        self.zarr_array = torch.from_numpy(numpy_array).reshape(-1, numpy_array.shape[-1])
        self.total_samples = self.zarr_array.shape[0]

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        return self.zarr_array[idx]
# %%
# Create the dataset
dataset = ZarrDataset(l1_activations)
# %%
# Create a DataLoader for batch processing
batch_size = 4096 * 8 # Adjust as needed
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


# %%
trainer_cfg = {
    "trainer": TrainerBatchTopK,
    "activation_dim": 768,
    "dict_size": 768 * 32,
    #"lr": 1e-3,
    "k" : 64,
    "device": "cuda",
    "layer": f"encoder{layer}",
    "lm_name": "leela",
}
# %%
trainer_cfgs = [trainer_cfg.copy(), trainer_cfg.copy()]
trainer_cfgs[0]['dict_size'] = 768 * 32
trainer_cfgs[1]['dict_size'] = 768 * 64
# %%
ae = trainSAE(
    data=dataloader,
    trainer_configs=[trainer_cfg],
    #steps=,
    log_steps=100,
    save_dir=os.path.join(os.path.dirname(__file__), 'sae'),
    use_wandb=True,
    wandb_entity="",
    wandb_project=f"l{layer}x3264k64",
)
# %%