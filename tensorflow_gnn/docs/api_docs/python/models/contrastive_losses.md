# Module: contrastive_losses

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/contrastive_losses/__init__.py">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

Contrastive losses.

Users of TF-GNN can use these layers by importing them next to the core library:

```python
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import contrastive_losses
```

## Classes

[`class AllSvdMetrics`](./contrastive_losses/AllSvdMetrics.md): Computes
multiple metrics for representations using one SVD call.

[`class BarlowTwinsTask`](./contrastive_losses/BarlowTwinsTask.md): A Barlow
Twins (BT) Task.

[`class ContrastiveLossTask`](./contrastive_losses/ContrastiveLossTask.md): Base
class for unsupervised contrastive representation learning tasks.

[`class CorruptionSpec`](./contrastive_losses/CorruptionSpec.md): Class for
defining corruption specification.

[`class Corruptor`](./contrastive_losses/Corruptor.md): Base class for graph
corruptor.

[`class DeepGraphInfomaxLogits`](./contrastive_losses/DeepGraphInfomaxLogits.md):
Computes clean and corrupted logits for Deep Graph Infomax (DGI).

[`class DeepGraphInfomaxTask`](./contrastive_losses/DeepGraphInfomaxTask.md): A
Deep Graph Infomax (DGI) Task.

[`class DropoutFeatures`](./contrastive_losses/DropoutFeatures.md): Base class
for graph corruptor.

[`class ShuffleFeaturesGlobally`](./contrastive_losses/ShuffleFeaturesGlobally.md):
A corruptor that shuffles features.

[`class TripletEmbeddingSquaredDistances`](./contrastive_losses/TripletEmbeddingSquaredDistances.md):
Computes embeddings distance between positive and negative pairs.

[`class TripletLossTask`](./contrastive_losses/TripletLossTask.md): The triplet
loss task.

[`class VicRegTask`](./contrastive_losses/VicRegTask.md): A VICReg Task.

## Functions

[`coherence(...)`](./contrastive_losses/coherence.md): Coherence metric
implementation.

[`numerical_rank(...)`](./contrastive_losses/numerical_rank.md): Numerical rank
implementation.

[`pseudo_condition_number(...)`](./contrastive_losses/pseudo_condition_number.md):
Pseudo-condition number metric implementation.

[`rankme(...)`](./contrastive_losses/rankme.md): RankMe metric implementation.

[`self_clustering(...)`](./contrastive_losses/self_clustering.md):
Self-clustering metric implementation.
