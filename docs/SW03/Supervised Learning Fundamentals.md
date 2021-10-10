# Supervised Learning Fundamentals

Load Data -> Prepare Data -> Data Analysis & Assessement -> Prediction

## Load Data
load data with prefered language, in our example #python

```python
df = pd.read_csv("cars.csv", parse_dates=['Registration'])
```

check after load if data has been correctly loaded


## Prepare Data
Second step prepare data. Here we 
- set necessary datatype and indices
- join tables


### Datatypes
Specifc datatypes can either be set during read oder with the funciton astype(..)

```python
#direct
df = pd.read_csv("cars.csv", parse_dates=['Registration'])
#indirect
df.Color.astype('category')
```


#### Handle categorical variables
set categorical values with  the panda function astype(...)

```python
df.Color = df.Color.astype('category')
df.dtypes #prints type
```

## Data Analysis & Assessment
Before modeling data we need to analyse the data and get familiar with it.

### Duplicates
Exact duplicates should never occur.

```python
# check if they are any
df.duplicated().any()
# if there are, print them...
df[df.duplicated(keep=False)].head(n=10)
# ... and drop them
df.drop_duplicates(inplace=True)
#df = df.drop_duplicates() # same result
```

### Null Values
Check for null values and if need be, replace them with default values or remove the set

```python
df.isna().any() # add another .any() to aggregate to a single Boolean
```

### Suspicious Ranges
Certain data ranges can be checked by making sure they make sense. **For example** a registration date should always be in the past.

df.Year.min(), df.Year.max()

### Redundant Data
Try minimizing data without losing **important** information. For example, having both registraionDate and registrationYear is absolute 

```python
if (df.Registration.dt.year == df.Year).all():
	# remove if true because all data is the same$
	df.drop('Registration', axis='columns', inplace=True)
```

you can count the different values

```python
df.Registration.dt.day.value_counts()
```

### Outliers
#outliers acan affect the performance of algorithms that are based on #Similarity & #Distance . Algorithms like #Linear-Regression are heavily influenced by #outliers, in such cases the outliers should almost always be removed even if theyre valid data.

To find visualize outliers we can create a boxplot

```python
numerical_cols = ['Price', 'Mileage', 'Horsepower', 'EngineSize']
df.loc[:, numerical_cols].plot(kind='box', subplots=True, layout=(2, 2), figsize=(10, 10), sharex=False)
```
![[Pasted image 20211010190005.png]]

#outliers: at least 1.5 * IQR above the 3rd quartile or below the 1st quartile

```python
# The following code can be used to calculate an upper bound.
# If applied, this bound must be calculated only on the training set, not on the complete dataset.
# In a dataset where there are outliers above as well as below the two quartiles, the lower bound
# would have to be calculated accordingly
q3 = df.loc[:, numerical_cols].describe().loc['75%']
iqr = q3 - df.loc[:, numerical_cols].describe().loc['25%']
upper_boundary = q3 + 1.5*iqr
upper_boundary
# And here the outliers are removed
df = df[(df.Price <= upper_boundary.Price) &
(df.Mileage <= upper_boundary.Mileage) &
(df.Horsepower <= upper_boundary.Horsepower) &
(df.EngineSize <= upper_boundary.EngineSize)]
```


### Data distribution and pairwise relation
Viewing the pairwise relation is easiest with a #pairplot

```python
sns.pairplot(df.loc[:, numerical_cols], diag_kind = "kde", kind = "scatter")
# or with just pandas:
# pd.plotting.scatter_matrix(df.loc[:, numerical_cols], diagonal='kde')
```
![[Pasted image 20211010190520.png]]


## Prediction
Once we finish our assessement we can use #sckit-learn to fit a model.
Algorithms:
- #KNN 

Before training our model we need to do some feature engineering

### Feature Engineering
We have categorical column which we need to transform.

#### #One-Hot encode
Making a each categorical value represented by a number

**before**
![[Pasted image 20211010190938.png]]

**after**
![[Pasted image 20211010190950.png]]

```python
df = pd.concat([df, pd.get_dummies(df.Color)], axis='columns')
df.drop('Color', axis='columns', inplace=True)
df.head() #output check
```


### #K-NearestNeighbor #KNN
TODO: regression problem







