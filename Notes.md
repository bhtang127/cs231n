# Notes

## Loss Function

__Multivariate Hinge Loss__:
$Lost(x,y) = \sum_{j\neq y} \max(0 , score\ of\ j - score\ of\ y + 1)$

```python
Loss_of_i(model, x, y):
    scores = model(x)
    loss_aux = np.max(0, scores - scores[y] + 1)
    loss_aux[y] = 0
    return np.sum(loss_aux)
``` 

__Softmax Loss__:
$Lost(x,y) = -\log(\frac{e^{s_y}}{\sum_j e^{s_j}})$

```python
Loss_of_i(model, x, y):
    scores = model(x)
    softmax_scores = np.exp(scores)
    return -np.log(softmax_scores[y] / np.sum(softmax_scores))
```