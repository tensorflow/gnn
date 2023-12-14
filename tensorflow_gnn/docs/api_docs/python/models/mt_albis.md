# Module: mt_albis

<!-- Insert buttons and diff -->

<a target="_blank" href="https://github.com/tensorflow/gnn/tree/master/tensorflow_gnn/models/mt_albis/__init__.py">
<img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View source
on GitHub </a>

TF-GNN's Model Template "Albis".

The TF-GNN Model Template "Albis" provides a small selection of field-tested GNN
architectures through the
<a href="./mt_albis/MtAlbisGraphUpdate.md"><code>mt_albis.MtAlbisGraphUpdate</code></a>
class.

Users of TF-GNN can use it by importing it next to the core library as

```python
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import mt_albis
```

## Functions

[`MtAlbisGraphUpdate(...)`](./mt_albis/MtAlbisGraphUpdate.md): Returns
GraphUpdate layer for message passing with Model Template "Albis".

[`graph_update_from_config_dict(...)`](./mt_albis/graph_update_from_config_dict.md):
Constructs a MtAlbisGraphUpdate from a ConfigDict.

[`graph_update_get_config_dict(...)`](./mt_albis/graph_update_get_config_dict.md):
Returns ConfigDict for graph_update_from_config_dict() with defaults.
