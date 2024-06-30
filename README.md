# MarkovPoints

MarkovPoints is a Python package for generating state embeddings from Markov chains. It provides a Python implementation of the algorithms described in the article "On the Representation of Sparse Stochastic Matrices with State Embedding". This package allows users to work with Markov chains, compute state embeddings, and explore intrinsic dimensions efficiently.

## Installation

To install the package, clone the repository from GitHub and install it using `pip`:

```sh
git clone https://github.com/ataidegualberto/Markov_Points.git
cd Markov_Points
pip install .
```

Alternatively, you can install directly from PyPI once it's published:
```sh
pip install markov_points
```
## Usage

Here is a basic example of how to use the package:

```python
import numpy as np
from markov_points import MarkovPoints, mean_perplexity

# Example Markov chain matrix P and initial probabilities P0
P = np.array([[0, 0.8, 0.2], [0.5, 0.5, 0], [0.5, 0.4, 0.1]])
P0 = np.array([9/27, 16/27, 2/27])

# Initialize MarkovPoints object
mkpts = MarkovPoints(P, P0, 2)

# Calculate mean perplexity
print(mean_perplexity(P))

# Fit the model
mkpts.fit()

# Calculate Q0 and Q for each state
print(mkpts.calculate_Q0())
for i in range(3):
    print(mkpts.calculate_Q(i))
```
## Contributing

Contributions are welcome! Feel free to submit issues or pull requests on [GitHub](https://github.com/ataidegualberto/Markov_Points).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
