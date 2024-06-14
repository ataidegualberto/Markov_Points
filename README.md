# MarkovPoints

MarkovPoints is a Python package for generating state embeddings from Markov chains. It provides a Python implementation of the algorithms described in the article "On the Representation of Sparse Stochastic Matrices with State Embedding". This package allows users to work with Markov chains, compute state embeddings, and explore intrinsic dimensions efficiently.

## Installation

You can install the package using `pip`:

```sh
pip install markov_points
```
## Usage

Here is a basic example of how to use the package:

```python
import numpy as np
from markov_points import MarkovPoints, intrinsic_dimension

# Example Markov chain matrix P and initial probabilities P0
P = np.array([[0, 0.8, 0.2], [0.5, 0.5, 0], [0.5, 0.4, 0.1]])
P0 = np.array([9/27, 16/27, 2/27])

# Initialize MarkovPoints object
mkpts = MarkovPoints(P, P0, 2)

# Calculate intrinsic dimension
print(intrinsic_dimension(P))

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
