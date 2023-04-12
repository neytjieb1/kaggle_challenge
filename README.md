# Classifying Biological Graph Data using Kernel Methods.

The final data may be found in `wflehman.ipynb` with precomputed kernel matrices in the `Data/` folder and the implementation of the logistic regression classifier 
in `LogisticRegression_NC.py`, as well as the SVM classifier in `???.py`

## To run our code
1. In the `Data` folder, add train and test files in `.pkl` format.
2. Run `wflehman.ipynb`

## Preprocessing
1. To ensure a unique label alphabet from the get-go, we first relabel the edge labels to $\{50, \dots, 53\}$. (The graphs provided have node labels ranging from values $\{0,\dots, 49\}$ and edge labels from $\{0,\dots, 3\}$)
2. For each graph $G$, we calculate the Weisfeiler-Lehman Subtree Kernel feature vector $phi(G)$, where we set the number of hops to $h=5$. 
3. The kernel Gram matrix $K$ is calculated as $[K]_{i,j}:=\langle{\phi(G),\phi(G')}\rangle$.
4. We then center the matrix, using the formulation as described in the course notes:
$$K^{c} = (I-U)K(I-U)$$ for $I\in\mathbb{R}^n$ is the identity matrix and U defined by $[U]_{i,j}:=\frac{1}{n}$
5. As a final step, we perform principal component analysis by calculating all eigenvectors of $K^c$ and projecting back onto them.

## WL-Algorithm
1. Assign to $v$ the multiset-label consisting of its own and neighbouring nodes' labels, $M_i(v) = {l_{i-1}(u) | u \in N(v)} \cup {l_{i-1}(v)}$
2. Sort $M_i(v)$ in ascending order and concatenate elements into a string $s_i(v)$
3. Append the node's current value  as prefix, ie  $s_i(v) \gets l_{i-1}(v)  s_i(v) $
4. Compress $s_i(v)$ by applying the hashing function $f$ and 
5. Finally, assign the new label: $l_i(v) := f(s_i(v))$

## Caveat
The SVM algorithm as implemented in _HW-2_ failed to converge in reasonable time, and the Logistic Regression algorithm we implemented somehow had a bug in. We therefore opted to use `sklearn` in order to succeed with at least some of the project. 
