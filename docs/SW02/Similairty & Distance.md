# #Distance & #Similarity

The smaller the distance => more similar but only for values that don't correlate.


Similairty is the opposite of distance

# #Euclidiane-Distance
Calculate distance between 2 points with help of pythagoras
![[Pasted image 20211008134411.png]]

```python
def euclidean_distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2)**2))
    # other implementations available:
    #return scipy.spatial.distance.euclidean(v1, v2)
    #return np.linalg.norm(v1 - v2)
	
```


# #Cosine-Similarity

Vektor between 2 vectors. The smaller the angle the more correlation (cos value closer to 1)

`````python
def cosine_similarity(v1, v2):
    sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return sim
`````
	
	
# #Cosin-Distance

1 - cosine similarity = distance


# #Normalize-Data

When comparing distances its important to use normalized data!
(Basically making all values have the same weight being able to compare for examlpe Kg of rocks with Mileage in a car )

## Normalizing strings
normalize the strings by tokenizing all the words and then applying lemmatization and stemming. For examlpe with #SnowballStemmer 



# #NearestNeighbour
Find most similar dataset with help of distances. Remember to use normalized 

### our implementation
`````python
def get_nearest_neighbor(source_car, cars, distance):
    distances = np.array([distance(source_car, car) for car in tqdm(cars)])
    idx = np.argmin(distances)
    min_distance = np.min(distances)
    return distance, idx
`````



#### Scikit
`````python
knn = NearestNeighbors(n_neighbors=1, metric="euclidean")
knn.fit(X_train_transform)
`````




# #Jaccard-Similarity
Similirity by looking at how the intersaction count to union count ratio

`````python
def jaccard_similarity(list1, list2):
    #Mike
    intersection = (float)(len(list(set(list1).intersection(set(list2)))))
    union = (float)((len(set(list1)) + len(set(list2))) - intersection)
    similarity = (float)(intersection / union);
    print('Mike '+ str(intersection) + ' ' + str(union))
    #Dave
    similarity = 0
    whole_list = list(set(list1 + list2))
    intersection = []
    for i in list1:
        for j in list2:
            if (i == j):
                intersection.append(i)
    similarity = len(intersection) / len(whole_list)
    print('David '+ str(len(intersection)) + ' ' + str(len(whole_list)))
    return similarity
`````

## using nearest neighbour with jaccard example

````python
def nearest_neighbor(wine, wines):
    idx = -1
    similarity = -1
   
    '''sol1 asarray dm
    biggest = -1;
    for poss_idx,w in enumerate(wines):
        _ = jaccard_similarity(wine, w)
        if _>0 and _>biggest and _<=1:
            biggest = _
            idx = poss_idx
    similarity = biggest
    '''
    #sol 2 with asarray
    dm = np.asarray([[jaccard_similarity(p1, wine) 
                  for p1 in wines]])
    idx = np.argmax(dm)
    similarity = np.max(dm)
    
    return idx, similarity
```