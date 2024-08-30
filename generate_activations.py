# %%
#%load_ext autoreload
#%autoreload 2
import torch
from leela_interp import Lc0sight, LeelaBoard, Lc0Model, get_lc0_pv_probabilities, ActivationCache
import os
import pandas as pd
import pickle
from tqdm import tqdm

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
current_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(current_folder)
file_path = os.path.join(parent_folder, "lc0.onnx")
# %%
model: Lc0Model = Lc0Model(file_path, device=device)
# %%
with open(os.path.join(parent_folder, 'unfiltered_puzzles.pkl'), 'rb') as f:
    puzzles = pickle.load(f)
# %%
puzzles['correct'] = puzzles.apply(lambda row: row['principal_variation'][0] == row['full_model_moves'][0], axis=1)

# %%
correct_puzzles = puzzles[puzzles['correct']]

# %%
with open(os.path.join(parent_folder, 'correct_puzzles.pkl'), "wb") as f:
    pickle.dump(correct_puzzles, f)

# # %%
# ten_k_puzzles = correct_puzzles#.iloc[:10000]

# # %%
# # Create LeelaBoards for each correct puzzle
# leela_boards = []
# for _, puzzle in tqdm(ten_k_puzzles.iterrows(), total=len(ten_k_puzzles), desc="Creating LeelaBoards"):
#     board = LeelaBoard.from_puzzle(puzzle)
#     leela_boards.append(board)

# print(f"Created {len(leela_boards)} LeelaBoards")
# # %%
# with open(os.path.join(parent_folder, 'leelaboards.pkl'), "wb") as f:
#     pickle.dump(leela_boards, f)
with open(os.path.join(parent_folder, 'leelaboards.pkl'), 'rb') as f:
    leela_boards = pickle.load(f) 
# %%
activations = ActivationCache.capture(
            model=model,
            boards=leela_boards[:400000],
            batch_size=4096,
            # There's a typo in Lc0, so we mirror it; "rehape" is deliberate
            names=[f"encoder{layer}/ln2" for layer in range(15)],
            n_samples=400_000,
            store_boards=False,
            # Uncomment to store activations on disk (they're about 70GB).
            # Without a path, they'll be kept in memory, which is faster but uses 70GB of RAM.
            path="residual_activations.zarr",
            overwrite=True,
        )
# %%
name = "encoder14/ln2"
act = activations[name]
act[0].shape
