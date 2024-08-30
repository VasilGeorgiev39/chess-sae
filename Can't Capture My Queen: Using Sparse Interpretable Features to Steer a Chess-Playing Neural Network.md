# Intro

Humans have been playing chess for thousands of years, building up a rich catalog of concepts like forks, pins, and passed pawns. But what about neural networks? When they play at a grandmaster level, do they develop similar ideas? Or do their internal representations look completely different from ours? Let’s dive in and find out!

# Motivation

-   As a huge chess fan and former professional player, it really bugs me that I don’t fully understand how a chess model plays.
-   Most of the cutting-edge models today are multi-modal, and that trend is only going to grow. But the majority of interpretability work is still focused on text. I think it would be cool to experiment with how well current interpretability techniques, like SAEs, work on different modalities (and chess models are basically vision transformers).
-   A toy setting like this makes more sense for a time-limited experimental project.
-   Chess has a clear ground truth, which we can use to automatically scale up SAEs.
-   I haven’t trained SAEs before, so it makes sense to start in an environment I’m more familiar with.
-   Mostly, I just wanted an excuse to do something with chess. 😊

# Related work

["Acquisition of Chess Knowledge in AlphaZero" (McGrath et al, 2021)](https://arxiv.org/abs/2111.09259) showed that chess playing neural networks have representations of human concepts. However their approach was to create a list of concepts they want to look for and probe for them. I want to take an unsupervised approach and find as many concepts as possible. Would it be possible to find a concept that humans have not given a name to yet?

[Towards Multimodal Interpretability: Learning Sparse Interpretable Features in Vision Transformers](https://www.lesswrong.com/posts/bCtbuWraqYTDtuARg/towards-multimodal-interpretability-learning-sparse-2) is previous work using SAE with vision transformers.

In our previous paper [Evidence of Learned Look-Ahead in a Chess-Playing Neural Network (Jenner at al, 2024)](https://leela-interp.github.io/) we did mechanistic interpretability work on a chess transformer and found evidence of learned look-ahead. 

# Setup

I decided to use [Leela Chess Zero](https://lczero.org/) (Lc0), the strongest neural network engine [citation needed]. Lc0 takes a board state as input and produces a distribution over moves and a value (win/draw/loss probability) for that state. It has a policy and a value head, which share a transformer as the main body of the network. Each square of the chess board is represented as a sequence position in this transformer.

<sup>Note: Initially I wanted to do the experiment on DeepMind's transformer from the paper [Grandmaster-Level Chess Without Search (Ruoss et al., 2024)](https://arxiv.org/abs/2402.04494) as it has cleaner architecture (although I think there could have been further improvements like flipping the board if it is black's turn), however after reviewing the github repo I noticed that DeepMind open sourced only the weights to the 'action-value' model (a model that takes a pair of a position and move and produces a distribution over the value of the board after that move), which I thought would be quite messy for my setup, I would much rather prefer to have what the paper calls 'behavioral-clonning' model (a model that takes a board and out produces a distribution over all possible moves). I think my work would transfer easily if this model gets open sourced as well.</sup>

For the dataset I used puzzles from Lichess. I think using puzzles as a dataset will help labeling the features as I expect puzzle positions will have more distinguishable features like 'fork' and 'mate' which will be easier to distinguish from a feature like 'improve the safety of the king'. Also the puzzles come with labels which can help in automatically labeling the features.

# Progress

## Dataset

I filtered 800k puzzles that Lc0 gets right. I expect that common motifs in puzzles have corresponding features and I assume that if the model solves the puzzle correctly it is more likely that the feature will be present. From those 800k puzzles I took the first 400k and generated activations for all layers. As the board has 64 squares I ended up with 25M activations. This took ~2 hours on a 3090 and when stored as a .zarr file was taking 1TB.

I think that for chess transformers it would be interesting to train SAEs on the whole flattened residual stream (as opposed for every square), however this exploded the memory requirements.

<sup>Note: I decided to store them as a .zarr so that I can easily read them, however I realised that was too slow and basically ended up loading them in memory so .zarr was possibly not the right choice here. I am a bit confused but it seems that .zarr has huge overhead as loading the activations in memory for 1 layer was taking 10GB. I need to read up on chunking with .zarr as I think this was what was slowing me down massively</sup>

## SAE Training

I decided to use BatchTopK SAEs (as seen by (link for Bart post) which builds upon (link for Gao) on the resid post. I realised that I wouldn't have time to do a full parameter sweep so decided to fiddle with the parameters manually on a couple of layers.

I think one of the important decisions I faced was after I have a trained autoencoder, how do I look for 'interesting' features? I was not familiar with the literature and after some brainstorming I decided to fill features based on 4 criteria:
* Activation frequency
* Activation sparsity
* Activation strength
* Activating above some threshold T at least K number of times (I chose those values somewhat arbitrarily to 4 and 5 after observing the results).

Other approaches that I would love to explore:
* Maximum reconstruction loss (this was too slow to calculate)
* Maximum effect on the logit of the predicted move (too slow as well)
* Activating when a puzzle label is present - this would have allowed me to take advantage of the labeled data and 'automatically' label features (the caveat is that the labels of this dataset are not that great actually)
* To improve on that we can train some probes (or algorithmically detect when a concept is present like 'an-passant') and find the features that have high correlation with the probes. In general, with chess we have ground truth available and we can use that.

### Layer 10

I started with layer 10 as it is 2/3rds in the model and was expecting to see some specific features formed already. I decided to use an expansion factor of 32 and fiddled with the expected L0 - I ran 3 runs with k in [16, 32, 64]

#### k = 16

**Most frequent features**

Features in general are not very interpretable
Feature 17718: Something about the pawn in front of the enemy king that is going to be checkmated
Feature 22366: Seems maybe fork related but stores the info on weird squares

**Most sparse features**

A bit more interpretable but not clear cut

Feature 8441, 2311 and 22551: Vulnerable enemy king (with 2311 and 22551 specifically about giving horizontal check to the enemy king)
Feature 3945: Something about our queen but not super clear

**Most strongly activated**
Exactly the same as the sparse features

**Threshold**

More clear-cut features

**Feature 1655: We are taking the enemy queen! However the information is stored on weird unrelated squares, usually around the corner**
Feature 2294: Taking the queen again all 5/5 top activations are from the same position (meaning the information appears to be copied over multiple squares)
Feature 5809: Something about pawn ending but not clear
Feature 10764: Information stored on our king's square that we are giving check to the opponent's king with a rook (weird place to store that information)
Feature 17771: Pawn ending again

#### k = 32

**Most frequent features**

Not interpretable features with one exception

Feature 12545: Square where a rook would normally love to go but the file is protected. Very niche, humans don't have a term for this

**Most sparse features**
Mostly piece related features
Feature 17363: We are giving a check on the enemy king with a queen on F7 (mirrored C2). Very specific
Feature 2739: Something about the enemy king but not specific at all
Feature 6465: Something about the enemy knight
Feature 7392: Something about our knight
**Most strongly activated**
pretty much same
**Threshold**
Same as the features from k = 16 (with different ids of course)
#### k = 64

**Most frequent features**
Feature 17484: Same as feature 12545 from k = 32
Feature 8046: We are giving a check to the enemy king with a queen or a rook. Information stored on a weird square
Feature 13087: We are moving the queen, information stored on a weird square
**Most sparse features**
Feature 10755: Giving a check on the last rank with a rook, queen and rook mate incoming
Feature 14697: Long live the queen feature
Feature 12558: Someting about the enemy bishop
Feature 8392: Someting about the enemy knight
Feature 8392: Someting about the enemy rook

**Most strongly activated**
Same
**Threshold**
Feature 460: Our King holds information that we are giving a check with a rook
Feature 6623: Something about pawn ending stored in the corner
Feature 23151: Another we are taking a queen feature


Based on those observations it seemed to me that the higher the K the more interpretable the features are so I decided to fix K at 64, move up a couple of layers and do two runs with expanding factor of 32 and 64

### Layer 12

#### x = 32

**Most sparse features**
Mostly piece related features again, this seems like a trend for sparse features

Feature 22493: Something about our king
Feature 21783: Something about an enemy pawn
Feature 11316: Something about an enemy pawn near their king
Feature 23988: Something about the enemy king
Feature 23656: Something about our rook

**Most strongly activated**
Very similar

**Threshold**
Feature 4793: Same square colour bishop ending (information on a square opposite the bishops)
Feature 15906: We are giving a check, information stored on a weird square
Feature 17922: Something about our pawn

#### x = 64

**Most frequent features**
Nothing interesting

**Most sparse features**
Feature 2526: Something about our king
Feature 2439: We are giving a check with a queen and information is stored in our pawn
Feature 34197: Something about enemy pawn

**Most strongly activated**
Feature 4955: Back rank mate stored with our king

**Threshold**
**Features 18618 and 21530: An passant features!**

In general I was disappointed that increasing the expansion to 64 didn't lead to noticeably more interpretable features (although we got the an passant feature!)

### Layer 13

I was excited to layer 13 because our previous work showed that L12H12 was crucial for lookahead so was hoping to see some interesting features

#### x = 32, k = 32

**Most frequent features**
Feature 23769: Back rank mate stored in a random square

**Most sparse features**
Feature 4688: Queen captures for back rank mate
Feature 19298: Something about our king when we capture
Feature 19702: Our knight supports the moving piece
Feature 5912: Enemy queen 

**Most strongly activated**
Very similar

### Observations

The higher up the layers I go, the less interpretable the features become
I was surprised that increasing the expansion factor didn't increase the number of interpretable features given that there were many 'close to interpretable' features before
Lot's of features for specific piece types but in clear in what situations
Most interpretable features came from the sparse features filter
Very often the model stores information in the corner squares or other unrelated squares

## Causal intervention

To provide stronger evidence that the SAEs have learned actual features I wanted to do a causal intervention experiment where ablating (or potentially increasing) that specific feature would change the model's behaviour (ideally just for things related for that feature and nothing else).

I choose the feature which I thought was the cleanest - Feature 1655 from Layer 10: Queen capture. I created the following board:

Then I performed ablation on the highest activation square for that feature:

Vioala, the model doesn't want to take the queen!

To verify the result, I performed ablation on a different square and also on a different feature. In both cases the model still wants to capture the queen.

Further work: Verify that the ablation didn't otherwise ruin the performance of the model

# Conclusions

I trained my first SAEs on a chess playing neural network and found a handful of interpretable features. I picked one of those features and performed causal intervention (although definitely not conclusive) to prove that the feature is indeed used in the model's decision making. I now have a good pipeline and would be excited to scale up the experiment. I am mostly excited to try to use the labels on the puzzles to try to automatically label SAE features. I think chess can be a nice testbed for testing and evaluating different SAE techniques as there is ground truth that can be used.
