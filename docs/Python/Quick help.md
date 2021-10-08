# Full solutions

## Cosine Similarity in python with help of scipy

```python
from scipy import spatial
dataSetI = [3, 14, 18, 23]
dataSetII = [12, 16, 21, 29]
result = 1 - spatial.distance.cosine(dataSetI, dataSetII)
```


## #SnowballStemmer
Stems and lemmatize strings
#Lemmatization is the process of grouping together the inflected forms of a word so they can be analysed as a single item
#Stemming  just removes or stems the last few characters of a word




## Indexes and not indexes
tilde sign shifts all bits / use all "other" values
```python
test_idx = [156, 233, 203]
test_idx = df.index.isin(test_idx)

test = df[test_idx]
train = df[~test_idx]
```


## #iloc

`.iloc[]` is primarily integer position based (from `0` to `length-1` of the axis


## #ArgMin
is **typically used to find the smallest possible values given constraints** returning its index in the array of the arguement

for value just use #min(args)



