# %%
%load_ext autoreload
%autoreload 2
import torch as t
import os
import json
import sys
import pickle
import pandas as pd
from leela_interp.tools.activations import ActivationCache
from leela_interp import LeelaBoard, Lc0Model, Lc0sight

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from dictionary_learning.trainers import BatchTopKSAE
# %%
dict_path = os.path.join(os.path.dirname(__file__), 'sae', 'trainer_0', 'ae.pt')
config_path = os.path.join(os.path.dirname(__file__), 'sae', 'trainer_0', 'config.json')
state_dict = t.load(dict_path)
config = json.load(open(config_path))
act_dim = config['trainer']['activation_dim']
dict_size = config['trainer']['dict_size']
k = config['trainer']['k']
# %%
ae = BatchTopKSAE(act_dim, dict_size, k)
ae.load_state_dict(state_dict)
ae.to('cuda')
# %%
layer = 13
# %%
activations = ActivationCache.load(os.path.join(os.path.dirname(__file__), '.', 'notebooks', 'residual_activations.zarr'))
stored_activations = activations[f'encoder{layer}/ln2']
# %%

# Create a PyTorch dataset from l1_activations
class ZarrDataset(t.utils.data.Dataset):
    def __init__(self, zarr_array):
        # Convert Zarr array to numpy array before creating torch tensor
        numpy_array = zarr_array[:]
        self.zarr_array = t.from_numpy(numpy_array).reshape(-1, numpy_array.shape[-1])
        self.total_samples = self.zarr_array.shape[0]

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        return self.zarr_array[idx]
# %%
# Create the dataset
activations_dataset = ZarrDataset(stored_activations)
# %%
# Find the top 5 activations that maximally activate each feature in the autoencoder
import torch as t
from torch.utils.data import DataLoader
from tqdm import tqdm

# Create a DataLoader for batch processing
batch_size = 4096 #* 4 # Adjust as needed
dataloader = DataLoader(activations_dataset, batch_size=batch_size, shuffle=False)
# %%
# Initialize tensors to store top 5 activations and indices for each feature
top_k = 5
top_activations = t.zeros((ae.dict_size, top_k), device='cuda')
top_indices = t.zeros((ae.dict_size, top_k), dtype=t.long, device='cuda')

# Process batches
with t.no_grad():
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        if batch_idx == 1000:
            break
        batch = batch.to('cuda')
        
        # Encode the batch
        features = ae.encode(batch)
        
        # Calculate batch indices
        batch_indices = t.arange(batch_idx * batch_size, (batch_idx + 1) * batch_size, device='cuda')
        
        # Update top activations and indices for all features at once
        combined = t.cat([top_activations, features.T], dim=1)
        combined_indices = t.cat([top_indices, batch_indices.unsqueeze(0).expand(ae.dict_size, -1)], dim=1)
        
        top_k_values, top_k_indices = combined.topk(top_k, dim=1)
        top_activations = top_k_values
        top_indices = t.gather(combined_indices, 1, top_k_indices)

# Get the activations that maximally activate each feature
top_activating_inputs = activations_dataset.zarr_array[top_indices.cpu()].to('cuda')

print(f"Shape of top_activating_inputs: {top_activating_inputs.shape}")
print(f"Top 5 activation values for first feature: {top_activations[0]}")
print(f"Indices of top 5 activations for first feature: {top_indices[0]}")

# %%
# Find features with all 5 activations greater than 0.7
threshold = 4
features_above_threshold = (top_activations > threshold).all(dim=1)
high_activation_features = t.where(features_above_threshold)[0]

print(f"Number of features with all activations > {threshold}: {len(high_activation_features)}")
print(f"Indices of these features: {high_activation_features.tolist()}")

# If you want to see the actual activation values for these features:
if len(high_activation_features) > 0:
    print("\nTop 5 activation values for these features:")
    for feature_idx in high_activation_features:
        print(f"Feature {feature_idx}: {top_activations[feature_idx].tolist()}")


# %%
# Calculate puzzle indices and square indices for each feature
puzzle_indices = top_indices // 64
square_indices = top_indices % 64

# Create a DataFrame to store the results for high activation features
results = {}

for feature_idx in high_activation_features:
    results[feature_idx.item()] = {
        'Top_Activations': top_activations[feature_idx].tolist(),
        'Puzzle_Indices': puzzle_indices[feature_idx].tolist(),
        'Square_Indices': square_indices[feature_idx].tolist()
    }

# Display statistics for high activation features
if len(high_activation_features) > 0:
    print("\nStatistics for high activation features:")
    for feature_idx, feature_data in results.items():
        print(f"Feature {feature_idx}:")
        print(f"  Top Activations: {feature_data['Top_Activations']}")
        print(f"  Puzzle Indices: {feature_data['Puzzle_Indices']}")
        print(f"  Square Indices: {feature_data['Square_Indices']}")
        print()

# %%

correct_puzzles_path = os.path.join(os.path.dirname(__file__), 'correct_puzzles.pkl')
correct_puzzles = pickle.load(open(correct_puzzles_path, 'rb'))
# %%
import iceberg as ice
def visualize_interesting_features(interesting_features: dict, correct_puzzles: pd.DataFrame):
    feature_plots = []
    for feature_idx, feature_data in interesting_features.items():
        data_indexes = feature_data[0]
        feature_activations = feature_data[1]
        puzzle_indices = [data_idx // 64 for data_idx in data_indexes]
        square_indices = [data_idx % 64 for data_idx in data_indexes]
        #print(f"Feature {feature_idx}:")
        board_plots = []
        for puzzle_idx, square_idx, activation in zip(puzzle_indices, square_indices, feature_activations):
            puzzle = correct_puzzles.iloc[puzzle_idx]
            
            board = LeelaBoard.from_puzzle(puzzle)
            
            model_move = puzzle['full_model_moves'][0]
            labels = puzzle['Themes']
            principal_variation = puzzle['principal_variation']
            square_str = board.idx2sq(square_idx)
            heatmap = {square_str: 'red'}
            board.pc_board.move_stack = None
            board_plots.append(board.plot(moves=model_move, heatmap=heatmap, caption=f"Feature {feature_idx}\nPuzzleID: {puzzle['PuzzleId']}\nActivation: {activation}\nLabels: {labels}\nPrincipal Variation: {principal_variation}"))
        feature_plots.append(ice.Arrange(board_plots, gap=10))
    display(ice.Arrange(feature_plots, gap=10, arrange_direction=ice.Arrange.Direction.VERTICAL))

    
# %%

from tqdm import tqdm

def find_interesting_features(ae: BatchTopKSAE, dataloader: DataLoader, threshold: float = 4, top_k: int = 5):
    frequent_features = []
    sparse_features = []
    strongly_activated_features = []
    features_above_threshold = []

    device = "cuda"
    feature_activations = t.zeros(ae.dict_size, device=device)
    feature_squared_activations = t.zeros(ae.dict_size, device=device)
    feature_counts = t.zeros(ae.dict_size, device=device)
    total_samples = 0

    # Initialize tensors to store top k activations and indices for each feature
    top_activations = t.zeros((ae.dict_size, top_k), device=device)
    top_indices = t.zeros((ae.dict_size, top_k), dtype=t.long, device=device)

    ae.eval()
    with t.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            batch = batch.to(device)
            features = ae.encode(batch)
            
            # Count activations and compute statistics
            active_features = (features != 0).float()
            feature_activations += active_features.sum(0)
            feature_squared_activations += (features ** 2).sum(0)
            feature_counts += active_features.sum(0)
            total_samples += batch.size(0)

            # Update top activations and indices
            batch_indices = t.arange(batch_idx * dataloader.batch_size, 
                                     (batch_idx + 1) * dataloader.batch_size, 
                                     device=device)
            combined = t.cat([top_activations, features.T], dim=1)
            combined_indices = t.cat([top_indices, 
                                      batch_indices.unsqueeze(0).expand(ae.dict_size, -1)], 
                                     dim=1)
            top_k_values, top_k_indices = combined.topk(top_k, dim=1)
            top_activations = top_k_values
            top_indices = t.gather(combined_indices, 1, top_k_indices)

            del batch
            del features

    # Compute average activation and sparsity
    avg_activation = feature_activations / total_samples
    sparsity = 1 - (feature_counts / total_samples)
    
    # Compute average squared activation (for strongly activated features)
    avg_squared_activation = feature_squared_activations / feature_counts.clamp(min=1)

    # Find interesting features
    frequent_features = avg_activation.argsort(descending=True)[:top_k].tolist()
    sparse_features = (sparsity * avg_squared_activation).argsort(descending=True)[:top_k].tolist()
    strongly_activated_features = avg_squared_activation.argsort(descending=True)[:top_k].tolist()
    features_above_threshold = ((top_activations > threshold).sum(dim=1) >= top_k).nonzero().squeeze().tolist()

    # Create dictionaries to store feature indices, their top activations, and highest activation value
    frequent_features_dict = {idx: (top_indices[idx].tolist(), top_activations[idx].tolist()) for idx in frequent_features}
    sparse_features_dict = {idx: (top_indices[idx].tolist(), top_activations[idx].tolist()) for idx in sparse_features}
    strongly_activated_features_dict = {idx: (top_indices[idx].tolist(), top_activations[idx].tolist()) for idx in strongly_activated_features}
    features_above_threshold_dict = {idx: (top_indices[idx].tolist(), top_activations[idx].tolist()) for idx in features_above_threshold}

    return frequent_features_dict, sparse_features_dict, strongly_activated_features_dict, features_above_threshold_dict

# %%

frequent_features, sparse_features, strongly_activated_features, features_above_threshold = find_interesting_features(ae, dataloader)
# %%
visualize_interesting_features(features_above_threshold, correct_puzzles)
# %%
device = "cuda"
file_path = os.path.join(os.path.dirname(__file__), "lc0.onnx")
model: Lc0sight = Lc0sight(file_path, device=device)

# %%
def ablate_feature(feature_id: int, layer: int, patch_square: int):
    board = LeelaBoard.from_fen("rnbqkb1r/pppp1ppp/5n2/4p2Q/4P3/1P6/P1PP1PPP/RNB1KBNR b KQkq - 0 1")
    with model.trace(board):
        features_per_square = []
        activations_for_board = model.residual_stream(layer).output
        for square_index in range(64):
            activations_at_square = activations_for_board[0][square_index]
            features_per_square.append(ae.encode(activations_at_square))
        feature_activations = t.stack([features[feature_id] for features in features_per_square])
        max_activation_square = feature_activations.argmax().item().save()
        #max_activation_square = patch_square
        new_features_for_max_square = features_per_square[patch_square]
        new_features_for_max_square[feature_id] = 0

        new_activations_for_square = ae.decode(new_features_for_max_square)
        activations_for_board[0][patch_square] = new_activations_for_square
        output = model.output.save()
    
    probs = model.logits_to_probs(board, output[0])[0]
    policy = model.top_moves(board, probs, top_k=3)
    square_str = board.idx2sq(patch_square)
    heatmap = {square_str: 'red'}
    display(board.plot(caption=f"Ablating feature {feature_id} at square {square_str}", heatmap=heatmap))
    print(policy)
    print(max_activation_square)

# %%

ablate_feature(1, layer, 1)
# %%
ablate_feature(3889, layer, 23)
ablate_feature(1000, layer, 53)

# %%
