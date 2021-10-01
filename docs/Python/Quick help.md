# Full solutions

## Cosine Similarity in python with help of scipy

```python
from scipy import spatial
dataSetI = [3, 14, 18, 23]
dataSetII = [12, 16, 21, 29]
result = 1 - spatial.distance.cosine(dataSetI, dataSetII)
```




