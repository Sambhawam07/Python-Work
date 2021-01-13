# <font color = blue> IMDb Movie Assignment </font>

You have the data for the 100 top-rated movies from the past decade along with various pieces of information about the movie, its actors, and the voters who have rated these movies online. In this assignment, you will try to find some interesting insights into these movies and their voters, using Python.


```python
# Importing the required libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# Filtering out the warnings
import warnings
warnings.filterwarnings('ignore')
```

## Reading the data

_Read the movies data file provided and store it in a dataframe `movies`._


```python
# Read the csv file using 'read_csv'. Please write your dataset location here.
movies=pd.read_csv('Movie Assignment Data.csv')
movies.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>title_year</th>
      <th>budget</th>
      <th>Gross</th>
      <th>actor_1_name</th>
      <th>actor_2_name</th>
      <th>actor_3_name</th>
      <th>actor_1_facebook_likes</th>
      <th>actor_2_facebook_likes</th>
      <th>actor_3_facebook_likes</th>
      <th>...</th>
      <th>Votes3044M</th>
      <th>Votes3044F</th>
      <th>Votes45A</th>
      <th>Votes45AM</th>
      <th>Votes45AF</th>
      <th>Votes1000</th>
      <th>VotesUS</th>
      <th>VotesnUS</th>
      <th>content_rating</th>
      <th>Country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>La La Land</td>
      <td>2016</td>
      <td>30000000</td>
      <td>151101803</td>
      <td>Ryan Gosling</td>
      <td>Emma Stone</td>
      <td>Amiée Conn</td>
      <td>14000</td>
      <td>19000.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>7.9</td>
      <td>7.8</td>
      <td>7.6</td>
      <td>7.6</td>
      <td>7.5</td>
      <td>7.1</td>
      <td>8.3</td>
      <td>8.1</td>
      <td>PG-13</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Zootopia</td>
      <td>2016</td>
      <td>150000000</td>
      <td>341268248</td>
      <td>Ginnifer Goodwin</td>
      <td>Jason Bateman</td>
      <td>Idris Elba</td>
      <td>2800</td>
      <td>28000.0</td>
      <td>27000.0</td>
      <td>...</td>
      <td>7.8</td>
      <td>8.1</td>
      <td>7.8</td>
      <td>7.8</td>
      <td>8.1</td>
      <td>7.6</td>
      <td>8.0</td>
      <td>8.0</td>
      <td>PG</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lion</td>
      <td>2016</td>
      <td>12000000</td>
      <td>51738905</td>
      <td>Dev Patel</td>
      <td>Nicole Kidman</td>
      <td>Rooney Mara</td>
      <td>33000</td>
      <td>96000.0</td>
      <td>9800.0</td>
      <td>...</td>
      <td>7.9</td>
      <td>8.2</td>
      <td>8.0</td>
      <td>7.9</td>
      <td>8.4</td>
      <td>7.1</td>
      <td>8.1</td>
      <td>8.0</td>
      <td>PG-13</td>
      <td>Australia</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arrival</td>
      <td>2016</td>
      <td>47000000</td>
      <td>100546139</td>
      <td>Amy Adams</td>
      <td>Jeremy Renner</td>
      <td>Forest Whitaker</td>
      <td>35000</td>
      <td>5300.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>7.8</td>
      <td>7.8</td>
      <td>7.6</td>
      <td>7.6</td>
      <td>7.7</td>
      <td>7.3</td>
      <td>8.0</td>
      <td>7.9</td>
      <td>PG-13</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Manchester by the Sea</td>
      <td>2016</td>
      <td>9000000</td>
      <td>47695371</td>
      <td>Casey Affleck</td>
      <td>Michelle Williams</td>
      <td>Kyle Chandler</td>
      <td>518</td>
      <td>71000.0</td>
      <td>3300.0</td>
      <td>...</td>
      <td>7.7</td>
      <td>7.7</td>
      <td>7.6</td>
      <td>7.6</td>
      <td>7.6</td>
      <td>7.1</td>
      <td>7.9</td>
      <td>7.8</td>
      <td>R</td>
      <td>USA</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 62 columns</p>
</div>



_Inspect the dataframe for dimensions, null-values, and summary of different numeric columns._


```python
# Check the number of rows and columns in the dataframe
movies.shape
```




    (100, 62)




```python
# Check the column-wise info of the dataframe
movies.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 100 entries, 0 to 99
    Data columns (total 62 columns):
     #   Column                  Non-Null Count  Dtype  
    ---  ------                  --------------  -----  
     0   Title                   100 non-null    object 
     1   title_year              100 non-null    int64  
     2   budget                  100 non-null    int64  
     3   Gross                   100 non-null    int64  
     4   actor_1_name            100 non-null    object 
     5   actor_2_name            100 non-null    object 
     6   actor_3_name            100 non-null    object 
     7   actor_1_facebook_likes  100 non-null    int64  
     8   actor_2_facebook_likes  99 non-null     float64
     9   actor_3_facebook_likes  98 non-null     float64
     10  IMDb_rating             100 non-null    float64
     11  genre_1                 100 non-null    object 
     12  genre_2                 97 non-null     object 
     13  genre_3                 74 non-null     object 
     14  MetaCritic              95 non-null     float64
     15  Runtime                 100 non-null    int64  
     16  CVotes10                100 non-null    int64  
     17  CVotes09                100 non-null    int64  
     18  CVotes08                100 non-null    int64  
     19  CVotes07                100 non-null    int64  
     20  CVotes06                100 non-null    int64  
     21  CVotes05                100 non-null    int64  
     22  CVotes04                100 non-null    int64  
     23  CVotes03                100 non-null    int64  
     24  CVotes02                100 non-null    int64  
     25  CVotes01                100 non-null    int64  
     26  CVotesMale              100 non-null    int64  
     27  CVotesFemale            100 non-null    int64  
     28  CVotesU18               100 non-null    int64  
     29  CVotesU18M              100 non-null    int64  
     30  CVotesU18F              100 non-null    int64  
     31  CVotes1829              100 non-null    int64  
     32  CVotes1829M             100 non-null    int64  
     33  CVotes1829F             100 non-null    int64  
     34  CVotes3044              100 non-null    int64  
     35  CVotes3044M             100 non-null    int64  
     36  CVotes3044F             100 non-null    int64  
     37  CVotes45A               100 non-null    int64  
     38  CVotes45AM              100 non-null    int64  
     39  CVotes45AF              100 non-null    int64  
     40  CVotes1000              100 non-null    int64  
     41  CVotesUS                100 non-null    int64  
     42  CVotesnUS               100 non-null    int64  
     43  VotesM                  100 non-null    float64
     44  VotesF                  100 non-null    float64
     45  VotesU18                100 non-null    float64
     46  VotesU18M               100 non-null    float64
     47  VotesU18F               100 non-null    float64
     48  Votes1829               100 non-null    float64
     49  Votes1829M              100 non-null    float64
     50  Votes1829F              100 non-null    float64
     51  Votes3044               100 non-null    float64
     52  Votes3044M              100 non-null    float64
     53  Votes3044F              100 non-null    float64
     54  Votes45A                100 non-null    float64
     55  Votes45AM               100 non-null    float64
     56  Votes45AF               100 non-null    float64
     57  Votes1000               100 non-null    float64
     58  VotesUS                 100 non-null    float64
     59  VotesnUS                100 non-null    float64
     60  content_rating          100 non-null    object 
     61  Country                 100 non-null    object 
    dtypes: float64(21), int64(32), object(9)
    memory usage: 48.6+ KB
    

- `actor_2_facebook_likes` & `actor_3_facebook_likes` is of `float64` type which should be changed to `int`

- `title_year` is of `integer` type it should be changed to `Datetime format` 


```python
# Check the summary for the numeric columns 
movies.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title_year</th>
      <th>budget</th>
      <th>Gross</th>
      <th>actor_1_facebook_likes</th>
      <th>actor_2_facebook_likes</th>
      <th>actor_3_facebook_likes</th>
      <th>IMDb_rating</th>
      <th>MetaCritic</th>
      <th>Runtime</th>
      <th>CVotes10</th>
      <th>...</th>
      <th>Votes1829F</th>
      <th>Votes3044</th>
      <th>Votes3044M</th>
      <th>Votes3044F</th>
      <th>Votes45A</th>
      <th>Votes45AM</th>
      <th>Votes45AF</th>
      <th>Votes1000</th>
      <th>VotesUS</th>
      <th>VotesnUS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>100.000000</td>
      <td>1.000000e+02</td>
      <td>1.000000e+02</td>
      <td>100.000000</td>
      <td>99.000000</td>
      <td>98.000000</td>
      <td>100.000000</td>
      <td>95.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>...</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.00000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2012.820000</td>
      <td>7.838400e+07</td>
      <td>1.468679e+08</td>
      <td>13407.270000</td>
      <td>7377.303030</td>
      <td>3002.153061</td>
      <td>7.883000</td>
      <td>78.252632</td>
      <td>126.420000</td>
      <td>73212.160000</td>
      <td>...</td>
      <td>7.982000</td>
      <td>7.732000</td>
      <td>7.723000</td>
      <td>7.780000</td>
      <td>7.65100</td>
      <td>7.624000</td>
      <td>7.770000</td>
      <td>7.274000</td>
      <td>7.958000</td>
      <td>7.793000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.919491</td>
      <td>7.445295e+07</td>
      <td>1.454004e+08</td>
      <td>10649.037862</td>
      <td>13471.568216</td>
      <td>6940.301133</td>
      <td>0.247433</td>
      <td>9.122066</td>
      <td>19.050799</td>
      <td>82669.594746</td>
      <td>...</td>
      <td>0.321417</td>
      <td>0.251814</td>
      <td>0.260479</td>
      <td>0.282128</td>
      <td>0.21485</td>
      <td>0.213258</td>
      <td>0.301344</td>
      <td>0.361987</td>
      <td>0.232327</td>
      <td>0.264099</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2010.000000</td>
      <td>3.000000e+06</td>
      <td>2.238380e+05</td>
      <td>39.000000</td>
      <td>12.000000</td>
      <td>0.000000</td>
      <td>7.500000</td>
      <td>62.000000</td>
      <td>91.000000</td>
      <td>6420.000000</td>
      <td>...</td>
      <td>7.300000</td>
      <td>7.300000</td>
      <td>7.200000</td>
      <td>7.200000</td>
      <td>7.10000</td>
      <td>7.100000</td>
      <td>7.000000</td>
      <td>6.400000</td>
      <td>7.500000</td>
      <td>7.300000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2011.000000</td>
      <td>1.575000e+07</td>
      <td>4.199752e+07</td>
      <td>1000.000000</td>
      <td>580.000000</td>
      <td>319.750000</td>
      <td>7.700000</td>
      <td>72.000000</td>
      <td>114.750000</td>
      <td>30587.000000</td>
      <td>...</td>
      <td>7.700000</td>
      <td>7.600000</td>
      <td>7.500000</td>
      <td>7.600000</td>
      <td>7.50000</td>
      <td>7.475000</td>
      <td>7.500000</td>
      <td>7.100000</td>
      <td>7.800000</td>
      <td>7.600000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2013.000000</td>
      <td>4.225000e+07</td>
      <td>1.070266e+08</td>
      <td>13000.000000</td>
      <td>1000.000000</td>
      <td>626.500000</td>
      <td>7.800000</td>
      <td>78.000000</td>
      <td>124.000000</td>
      <td>54900.500000</td>
      <td>...</td>
      <td>8.000000</td>
      <td>7.700000</td>
      <td>7.700000</td>
      <td>7.800000</td>
      <td>7.65000</td>
      <td>7.600000</td>
      <td>7.800000</td>
      <td>7.300000</td>
      <td>7.950000</td>
      <td>7.750000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2014.000000</td>
      <td>1.500000e+08</td>
      <td>2.107548e+08</td>
      <td>20000.000000</td>
      <td>11000.000000</td>
      <td>1000.000000</td>
      <td>8.100000</td>
      <td>83.500000</td>
      <td>136.250000</td>
      <td>80639.000000</td>
      <td>...</td>
      <td>8.200000</td>
      <td>7.900000</td>
      <td>7.900000</td>
      <td>8.000000</td>
      <td>7.80000</td>
      <td>7.800000</td>
      <td>7.925000</td>
      <td>7.500000</td>
      <td>8.100000</td>
      <td>7.925000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2016.000000</td>
      <td>2.600000e+08</td>
      <td>9.366622e+08</td>
      <td>35000.000000</td>
      <td>96000.000000</td>
      <td>46000.000000</td>
      <td>8.800000</td>
      <td>100.000000</td>
      <td>180.000000</td>
      <td>584839.000000</td>
      <td>...</td>
      <td>8.800000</td>
      <td>8.700000</td>
      <td>8.700000</td>
      <td>8.500000</td>
      <td>8.10000</td>
      <td>8.100000</td>
      <td>8.500000</td>
      <td>8.200000</td>
      <td>8.700000</td>
      <td>8.800000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 53 columns</p>
</div>



___

## Data Analysis

Now that we have loaded the dataset and inspected it, we see that most of the data is in place. As of now, no data cleaning is required, so let's start with some data manipulation, analysis, and visualisation to get various insights about the data. 

-  ###  Reduce those Digits!

These numbers in the `budget` and `gross` are too big, compromising its readability. Let's convert the unit of the `budget` and `gross` columns from `$` to `million $` first.


```python
# Divide the 'gross' and 'budget' columns by 1000000 to convert '$' to 'million $'

movies.Gross = movies.Gross/1000000
movies.budget = movies.budget/1000000
movies[['Gross','budget']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gross</th>
      <th>budget</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>151.101803</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>341.268248</td>
      <td>150.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>51.738905</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100.546139</td>
      <td>47.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>47.695371</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>13.092000</td>
      <td>3.3</td>
    </tr>
    <tr>
      <th>96</th>
      <td>8.114507</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>97</th>
      <td>936.662225</td>
      <td>245.0</td>
    </tr>
    <tr>
      <th>98</th>
      <td>296.347721</td>
      <td>150.0</td>
    </tr>
    <tr>
      <th>99</th>
      <td>0.223838</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 2 columns</p>
</div>



___

###  Let's Talk Profit!

    1. Create a new column called `profit` which contains the difference of the two columns: `gross` and `budget`.
    2. Sort the dataframe using the `profit` column as reference.
    3. Extract the top ten profiting movies in descending order and store them in a new dataframe - `top10`.
    4. Plot a scatter or a joint plot between the columns `budget` and `profit` and write a few words on what you observed.
    5. Extract the movies with a negative profit and store them in a new dataframe - `neg_profit`


```python
# Creating the new column named 'profit' by subtracting the 'budget' column from the 'gross' column
movies['profit'] = movies.Gross-movies.budget
movies.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>title_year</th>
      <th>budget</th>
      <th>Gross</th>
      <th>actor_1_name</th>
      <th>actor_2_name</th>
      <th>actor_3_name</th>
      <th>actor_1_facebook_likes</th>
      <th>actor_2_facebook_likes</th>
      <th>actor_3_facebook_likes</th>
      <th>...</th>
      <th>Votes3044F</th>
      <th>Votes45A</th>
      <th>Votes45AM</th>
      <th>Votes45AF</th>
      <th>Votes1000</th>
      <th>VotesUS</th>
      <th>VotesnUS</th>
      <th>content_rating</th>
      <th>Country</th>
      <th>profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>La La Land</td>
      <td>2016</td>
      <td>30.0</td>
      <td>151.101803</td>
      <td>Ryan Gosling</td>
      <td>Emma Stone</td>
      <td>Amiée Conn</td>
      <td>14000</td>
      <td>19000.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>7.8</td>
      <td>7.6</td>
      <td>7.6</td>
      <td>7.5</td>
      <td>7.1</td>
      <td>8.3</td>
      <td>8.1</td>
      <td>PG-13</td>
      <td>USA</td>
      <td>121.101803</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Zootopia</td>
      <td>2016</td>
      <td>150.0</td>
      <td>341.268248</td>
      <td>Ginnifer Goodwin</td>
      <td>Jason Bateman</td>
      <td>Idris Elba</td>
      <td>2800</td>
      <td>28000.0</td>
      <td>27000.0</td>
      <td>...</td>
      <td>8.1</td>
      <td>7.8</td>
      <td>7.8</td>
      <td>8.1</td>
      <td>7.6</td>
      <td>8.0</td>
      <td>8.0</td>
      <td>PG</td>
      <td>USA</td>
      <td>191.268248</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lion</td>
      <td>2016</td>
      <td>12.0</td>
      <td>51.738905</td>
      <td>Dev Patel</td>
      <td>Nicole Kidman</td>
      <td>Rooney Mara</td>
      <td>33000</td>
      <td>96000.0</td>
      <td>9800.0</td>
      <td>...</td>
      <td>8.2</td>
      <td>8.0</td>
      <td>7.9</td>
      <td>8.4</td>
      <td>7.1</td>
      <td>8.1</td>
      <td>8.0</td>
      <td>PG-13</td>
      <td>Australia</td>
      <td>39.738905</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arrival</td>
      <td>2016</td>
      <td>47.0</td>
      <td>100.546139</td>
      <td>Amy Adams</td>
      <td>Jeremy Renner</td>
      <td>Forest Whitaker</td>
      <td>35000</td>
      <td>5300.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>7.8</td>
      <td>7.6</td>
      <td>7.6</td>
      <td>7.7</td>
      <td>7.3</td>
      <td>8.0</td>
      <td>7.9</td>
      <td>PG-13</td>
      <td>USA</td>
      <td>53.546139</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Manchester by the Sea</td>
      <td>2016</td>
      <td>9.0</td>
      <td>47.695371</td>
      <td>Casey Affleck</td>
      <td>Michelle Williams</td>
      <td>Kyle Chandler</td>
      <td>518</td>
      <td>71000.0</td>
      <td>3300.0</td>
      <td>...</td>
      <td>7.7</td>
      <td>7.6</td>
      <td>7.6</td>
      <td>7.6</td>
      <td>7.1</td>
      <td>7.9</td>
      <td>7.8</td>
      <td>R</td>
      <td>USA</td>
      <td>38.695371</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 63 columns</p>
</div>




```python
# Sorting the dataframe with the 'profit' column as reference using the 'sort_values' function in descending ordde

movies.sort_values(by='profit', ascending=False,inplace=True)
```


```python
movies.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>title_year</th>
      <th>budget</th>
      <th>Gross</th>
      <th>actor_1_name</th>
      <th>actor_2_name</th>
      <th>actor_3_name</th>
      <th>actor_1_facebook_likes</th>
      <th>actor_2_facebook_likes</th>
      <th>actor_3_facebook_likes</th>
      <th>...</th>
      <th>Votes3044F</th>
      <th>Votes45A</th>
      <th>Votes45AM</th>
      <th>Votes45AF</th>
      <th>Votes1000</th>
      <th>VotesUS</th>
      <th>VotesnUS</th>
      <th>content_rating</th>
      <th>Country</th>
      <th>profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>97</th>
      <td>Star Wars: Episode VII - The Force Awakens</td>
      <td>2015</td>
      <td>245.0</td>
      <td>936.662225</td>
      <td>Doug Walker</td>
      <td>Rob Walker</td>
      <td>0</td>
      <td>131</td>
      <td>12.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>8.2</td>
      <td>7.9</td>
      <td>7.8</td>
      <td>8.2</td>
      <td>7.7</td>
      <td>8.2</td>
      <td>7.9</td>
      <td>PG-13</td>
      <td>USA</td>
      <td>691.662225</td>
    </tr>
    <tr>
      <th>11</th>
      <td>The Avengers</td>
      <td>2012</td>
      <td>220.0</td>
      <td>623.279547</td>
      <td>Chris Hemsworth</td>
      <td>Robert Downey Jr.</td>
      <td>Scarlett Johansson</td>
      <td>26000</td>
      <td>21000.0</td>
      <td>19000.0</td>
      <td>...</td>
      <td>8.1</td>
      <td>7.9</td>
      <td>7.9</td>
      <td>8.1</td>
      <td>7.4</td>
      <td>8.3</td>
      <td>7.9</td>
      <td>PG-13</td>
      <td>USA</td>
      <td>403.279547</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Deadpool</td>
      <td>2016</td>
      <td>58.0</td>
      <td>363.024263</td>
      <td>Ryan Reynolds</td>
      <td>Ed Skrein</td>
      <td>Stefan Kapicic</td>
      <td>16000</td>
      <td>805.0</td>
      <td>361.0</td>
      <td>...</td>
      <td>7.9</td>
      <td>7.8</td>
      <td>7.8</td>
      <td>7.9</td>
      <td>7.3</td>
      <td>8.1</td>
      <td>7.9</td>
      <td>R</td>
      <td>USA</td>
      <td>305.024263</td>
    </tr>
    <tr>
      <th>32</th>
      <td>The Hunger Games: Catching Fire</td>
      <td>2013</td>
      <td>130.0</td>
      <td>424.645577</td>
      <td>Jennifer Lawrence</td>
      <td>Josh Hutcherson</td>
      <td>Sandra Ellis Lafferty</td>
      <td>34000</td>
      <td>14000.0</td>
      <td>523.0</td>
      <td>...</td>
      <td>7.9</td>
      <td>7.3</td>
      <td>7.2</td>
      <td>7.9</td>
      <td>6.7</td>
      <td>7.7</td>
      <td>7.4</td>
      <td>PG-13</td>
      <td>USA</td>
      <td>294.645577</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Toy Story 3</td>
      <td>2010</td>
      <td>200.0</td>
      <td>414.984497</td>
      <td>Tom Hanks</td>
      <td>John Ratzenberger</td>
      <td>Don Rickles</td>
      <td>15000</td>
      <td>1000.0</td>
      <td>721.0</td>
      <td>...</td>
      <td>8.3</td>
      <td>8.1</td>
      <td>8.1</td>
      <td>8.1</td>
      <td>8.1</td>
      <td>8.5</td>
      <td>8.3</td>
      <td>G</td>
      <td>USA</td>
      <td>214.984497</td>
    </tr>
    <tr>
      <th>8</th>
      <td>The Dark Knight Rises</td>
      <td>2012</td>
      <td>250.0</td>
      <td>448.130642</td>
      <td>Tom Hardy</td>
      <td>Christian Bale</td>
      <td>Joseph Gordon-Levitt</td>
      <td>27000</td>
      <td>23000.0</td>
      <td>23000.0</td>
      <td>...</td>
      <td>8.2</td>
      <td>7.9</td>
      <td>7.9</td>
      <td>7.9</td>
      <td>7.8</td>
      <td>8.4</td>
      <td>8.4</td>
      <td>PG-13</td>
      <td>USA</td>
      <td>198.130642</td>
    </tr>
    <tr>
      <th>45</th>
      <td>The Lego Movie</td>
      <td>2014</td>
      <td>60.0</td>
      <td>257.756197</td>
      <td>Morgan Freeman</td>
      <td>Will Ferrell</td>
      <td>Alison Brie</td>
      <td>11000</td>
      <td>8000.0</td>
      <td>2000.0</td>
      <td>...</td>
      <td>7.5</td>
      <td>7.4</td>
      <td>7.4</td>
      <td>7.4</td>
      <td>7.2</td>
      <td>8.0</td>
      <td>7.6</td>
      <td>PG</td>
      <td>Australia</td>
      <td>197.756197</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Zootopia</td>
      <td>2016</td>
      <td>150.0</td>
      <td>341.268248</td>
      <td>Ginnifer Goodwin</td>
      <td>Jason Bateman</td>
      <td>Idris Elba</td>
      <td>2800</td>
      <td>28000.0</td>
      <td>27000.0</td>
      <td>...</td>
      <td>8.1</td>
      <td>7.8</td>
      <td>7.8</td>
      <td>8.1</td>
      <td>7.6</td>
      <td>8.0</td>
      <td>8.0</td>
      <td>PG</td>
      <td>USA</td>
      <td>191.268248</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Despicable Me</td>
      <td>2010</td>
      <td>69.0</td>
      <td>251.501645</td>
      <td>Steve Carell</td>
      <td>Miranda Cosgrove</td>
      <td>Jack McBrayer</td>
      <td>7000</td>
      <td>2000.0</td>
      <td>975.0</td>
      <td>...</td>
      <td>7.9</td>
      <td>7.6</td>
      <td>7.5</td>
      <td>7.9</td>
      <td>7.0</td>
      <td>7.6</td>
      <td>7.6</td>
      <td>PG</td>
      <td>USA</td>
      <td>182.501645</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Inside Out</td>
      <td>2015</td>
      <td>175.0</td>
      <td>356.454367</td>
      <td>Amy Poehler</td>
      <td>Mindy Kaling</td>
      <td>Phyllis Smith</td>
      <td>1000</td>
      <td>767.0</td>
      <td>384.0</td>
      <td>...</td>
      <td>8.1</td>
      <td>7.9</td>
      <td>7.9</td>
      <td>7.9</td>
      <td>7.6</td>
      <td>8.2</td>
      <td>8.1</td>
      <td>PG</td>
      <td>USA</td>
      <td>181.454367</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 63 columns</p>
</div>




```python
# Get the top 10 profitable movies by using position based indexing. Specify the rows till 10 (0-9)

top10 = movies.iloc[:10]
top10
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>title_year</th>
      <th>budget</th>
      <th>Gross</th>
      <th>actor_1_name</th>
      <th>actor_2_name</th>
      <th>actor_3_name</th>
      <th>actor_1_facebook_likes</th>
      <th>actor_2_facebook_likes</th>
      <th>actor_3_facebook_likes</th>
      <th>...</th>
      <th>Votes3044F</th>
      <th>Votes45A</th>
      <th>Votes45AM</th>
      <th>Votes45AF</th>
      <th>Votes1000</th>
      <th>VotesUS</th>
      <th>VotesnUS</th>
      <th>content_rating</th>
      <th>Country</th>
      <th>profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>97</th>
      <td>Star Wars: Episode VII - The Force Awakens</td>
      <td>2015</td>
      <td>245.0</td>
      <td>936.662225</td>
      <td>Doug Walker</td>
      <td>Rob Walker</td>
      <td>0</td>
      <td>131</td>
      <td>12.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>8.2</td>
      <td>7.9</td>
      <td>7.8</td>
      <td>8.2</td>
      <td>7.7</td>
      <td>8.2</td>
      <td>7.9</td>
      <td>PG-13</td>
      <td>USA</td>
      <td>691.662225</td>
    </tr>
    <tr>
      <th>11</th>
      <td>The Avengers</td>
      <td>2012</td>
      <td>220.0</td>
      <td>623.279547</td>
      <td>Chris Hemsworth</td>
      <td>Robert Downey Jr.</td>
      <td>Scarlett Johansson</td>
      <td>26000</td>
      <td>21000.0</td>
      <td>19000.0</td>
      <td>...</td>
      <td>8.1</td>
      <td>7.9</td>
      <td>7.9</td>
      <td>8.1</td>
      <td>7.4</td>
      <td>8.3</td>
      <td>7.9</td>
      <td>PG-13</td>
      <td>USA</td>
      <td>403.279547</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Deadpool</td>
      <td>2016</td>
      <td>58.0</td>
      <td>363.024263</td>
      <td>Ryan Reynolds</td>
      <td>Ed Skrein</td>
      <td>Stefan Kapicic</td>
      <td>16000</td>
      <td>805.0</td>
      <td>361.0</td>
      <td>...</td>
      <td>7.9</td>
      <td>7.8</td>
      <td>7.8</td>
      <td>7.9</td>
      <td>7.3</td>
      <td>8.1</td>
      <td>7.9</td>
      <td>R</td>
      <td>USA</td>
      <td>305.024263</td>
    </tr>
    <tr>
      <th>32</th>
      <td>The Hunger Games: Catching Fire</td>
      <td>2013</td>
      <td>130.0</td>
      <td>424.645577</td>
      <td>Jennifer Lawrence</td>
      <td>Josh Hutcherson</td>
      <td>Sandra Ellis Lafferty</td>
      <td>34000</td>
      <td>14000.0</td>
      <td>523.0</td>
      <td>...</td>
      <td>7.9</td>
      <td>7.3</td>
      <td>7.2</td>
      <td>7.9</td>
      <td>6.7</td>
      <td>7.7</td>
      <td>7.4</td>
      <td>PG-13</td>
      <td>USA</td>
      <td>294.645577</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Toy Story 3</td>
      <td>2010</td>
      <td>200.0</td>
      <td>414.984497</td>
      <td>Tom Hanks</td>
      <td>John Ratzenberger</td>
      <td>Don Rickles</td>
      <td>15000</td>
      <td>1000.0</td>
      <td>721.0</td>
      <td>...</td>
      <td>8.3</td>
      <td>8.1</td>
      <td>8.1</td>
      <td>8.1</td>
      <td>8.1</td>
      <td>8.5</td>
      <td>8.3</td>
      <td>G</td>
      <td>USA</td>
      <td>214.984497</td>
    </tr>
    <tr>
      <th>8</th>
      <td>The Dark Knight Rises</td>
      <td>2012</td>
      <td>250.0</td>
      <td>448.130642</td>
      <td>Tom Hardy</td>
      <td>Christian Bale</td>
      <td>Joseph Gordon-Levitt</td>
      <td>27000</td>
      <td>23000.0</td>
      <td>23000.0</td>
      <td>...</td>
      <td>8.2</td>
      <td>7.9</td>
      <td>7.9</td>
      <td>7.9</td>
      <td>7.8</td>
      <td>8.4</td>
      <td>8.4</td>
      <td>PG-13</td>
      <td>USA</td>
      <td>198.130642</td>
    </tr>
    <tr>
      <th>45</th>
      <td>The Lego Movie</td>
      <td>2014</td>
      <td>60.0</td>
      <td>257.756197</td>
      <td>Morgan Freeman</td>
      <td>Will Ferrell</td>
      <td>Alison Brie</td>
      <td>11000</td>
      <td>8000.0</td>
      <td>2000.0</td>
      <td>...</td>
      <td>7.5</td>
      <td>7.4</td>
      <td>7.4</td>
      <td>7.4</td>
      <td>7.2</td>
      <td>8.0</td>
      <td>7.6</td>
      <td>PG</td>
      <td>Australia</td>
      <td>197.756197</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Zootopia</td>
      <td>2016</td>
      <td>150.0</td>
      <td>341.268248</td>
      <td>Ginnifer Goodwin</td>
      <td>Jason Bateman</td>
      <td>Idris Elba</td>
      <td>2800</td>
      <td>28000.0</td>
      <td>27000.0</td>
      <td>...</td>
      <td>8.1</td>
      <td>7.8</td>
      <td>7.8</td>
      <td>8.1</td>
      <td>7.6</td>
      <td>8.0</td>
      <td>8.0</td>
      <td>PG</td>
      <td>USA</td>
      <td>191.268248</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Despicable Me</td>
      <td>2010</td>
      <td>69.0</td>
      <td>251.501645</td>
      <td>Steve Carell</td>
      <td>Miranda Cosgrove</td>
      <td>Jack McBrayer</td>
      <td>7000</td>
      <td>2000.0</td>
      <td>975.0</td>
      <td>...</td>
      <td>7.9</td>
      <td>7.6</td>
      <td>7.5</td>
      <td>7.9</td>
      <td>7.0</td>
      <td>7.6</td>
      <td>7.6</td>
      <td>PG</td>
      <td>USA</td>
      <td>182.501645</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Inside Out</td>
      <td>2015</td>
      <td>175.0</td>
      <td>356.454367</td>
      <td>Amy Poehler</td>
      <td>Mindy Kaling</td>
      <td>Phyllis Smith</td>
      <td>1000</td>
      <td>767.0</td>
      <td>384.0</td>
      <td>...</td>
      <td>8.1</td>
      <td>7.9</td>
      <td>7.9</td>
      <td>7.9</td>
      <td>7.6</td>
      <td>8.2</td>
      <td>8.1</td>
      <td>PG</td>
      <td>USA</td>
      <td>181.454367</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 63 columns</p>
</div>




```python
#Plot profit vs budget

sns.jointplot(movies.budget, movies.profit, color='green')
plt.show()
```


![png](./images/output_20_0.png)


 


```python
plt.figure(figsize=(10,9))
sns.scatterplot(movies.budget, movies.profit,palette='rainbow',style=movies.content_rating
                ,hue=movies.content_rating,size=movies.content_rating, 
                sizes=(30,400)).set_title('Profit Vs Budget Analysis of 100 Movies')
plt.show()
```


![png](./images/output_22_0.png)


#### Analysis
- Though the movie datasets contain the top 100 movies of the year 2010 to 2016, still some of the movies are in loss
- Most of the top movies falls under `PG-13 Content Rating which is inappropriate for children under 13 and parental guidence required`

The dataset contains the 100 best performing movies from the year 2010 to 2016. However scatter plot tells a different story. We  can notice that there are some movies with negative profit. Although good movies do incur losses, but there appear to be quite a few movie with losses. What can be the reason behind this? Lets have a closer look at this by finding the movies with negative profit.


```python
#Find the movies with negative profit
plt.figure(figsize=(10,4))
sns.scatterplot('budget','profit',data= movies[movies.profit<0], palette='rainbow',
                size=movies.IMDb_rating,
                hue=movies.IMDb_rating,sizes=(90,200)).set_title('Profit Vs Budget Analysis of Top 100 Movies')
plt.show()
```


![png](./images/output_25_0.png)


**`Checkpoint 1:`** Can you spot the movie `Tangled` in the dataset? You may be aware of the movie 'Tangled'. Although its one of the highest grossing movies of all time, it has negative profit as per this result. If you cross check the gross values of this movie (link: https://www.imdb.com/title/tt0398286/), you can see that the gross in the dataset accounts only for the domestic gross and not the worldwide gross. This is true for may other movies also in the list.

_________

### The General Audience and the Critics

You might have noticed the column `MetaCritic` in this dataset. This is a very popular website where an average score is determined through the scores given by the top-rated critics. Second, you also have another column `IMDb_rating` which tells you the IMDb rating of a movie. This rating is determined by taking the average of hundred-thousands of ratings from the general audience. 

Lets find out the highest rated movies which have been liked by critics and audiences alike.
1. Firstly we will notice that the `MetaCritic` score is on a scale of `100` whereas the `IMDb_rating` is on a scale of 10.So, converting the `MetaCritic` column to a scale of 10.
2. Now, to find out the movies which have been liked by both critics and audiences alike and also have a high rating overall, we need to -
    - Create a new column `Avg_rating` which will have the average of the `MetaCritic` and `Rating` columns
    - Retain only the movies in which the absolute difference(using abs() function) between the `IMDb_rating` and `Metacritic` columns is less than 0.5.
    - Sort these values in a descending order of `Avg_rating` and retain only the movies with a rating equal to higher than `8` and store these movies in a new dataframe `UniversalAcclaim`.
    


```python
# Change the scale of MetaCritic
movies.MetaCritic=movies.MetaCritic/10
```


```python
# Find the average ratings
movies['Avg_rating']=(movies.MetaCritic + movies.IMDb_rating)/2
```


```python
#Sort in descending order of average rating

movies.sort_values(by='Avg_rating',ascending=False, inplace=True)
```


```python
# Find the movies with metacritic-rating < 0.5 and also with the average rating of >8

UniversalAcclaim = movies[(abs(movies.MetaCritic - movies.IMDb_rating) < 0.5) & (movies.Avg_rating>=8)]
UniversalAcclaim
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>title_year</th>
      <th>budget</th>
      <th>Gross</th>
      <th>actor_1_name</th>
      <th>actor_2_name</th>
      <th>actor_3_name</th>
      <th>actor_1_facebook_likes</th>
      <th>actor_2_facebook_likes</th>
      <th>actor_3_facebook_likes</th>
      <th>...</th>
      <th>Votes45A</th>
      <th>Votes45AM</th>
      <th>Votes45AF</th>
      <th>Votes1000</th>
      <th>VotesUS</th>
      <th>VotesnUS</th>
      <th>content_rating</th>
      <th>Country</th>
      <th>profit</th>
      <th>Avg_rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>95</th>
      <td>Whiplash</td>
      <td>2014</td>
      <td>3.3</td>
      <td>13.092000</td>
      <td>J.K. Simmons</td>
      <td>Melissa Benoist</td>
      <td>Chris Mulkey</td>
      <td>24000</td>
      <td>970.0</td>
      <td>535.0</td>
      <td>...</td>
      <td>8.1</td>
      <td>8.1</td>
      <td>8.2</td>
      <td>8.0</td>
      <td>8.6</td>
      <td>8.4</td>
      <td>R</td>
      <td>USA</td>
      <td>9.792000</td>
      <td>8.65</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Django Unchained</td>
      <td>2012</td>
      <td>100.0</td>
      <td>162.804648</td>
      <td>Leonardo DiCaprio</td>
      <td>Christoph Waltz</td>
      <td>Ato Essandoh</td>
      <td>29000</td>
      <td>11000.0</td>
      <td>265.0</td>
      <td>...</td>
      <td>8.0</td>
      <td>8.0</td>
      <td>8.1</td>
      <td>7.8</td>
      <td>8.4</td>
      <td>8.4</td>
      <td>R</td>
      <td>USA</td>
      <td>62.804648</td>
      <td>8.25</td>
    </tr>
    <tr>
      <th>93</th>
      <td>Dallas Buyers Club</td>
      <td>2013</td>
      <td>5.0</td>
      <td>27.296514</td>
      <td>Matthew McConaughey</td>
      <td>Jennifer Garner</td>
      <td>Denis O'Hare</td>
      <td>11000</td>
      <td>3000.0</td>
      <td>896.0</td>
      <td>...</td>
      <td>7.8</td>
      <td>7.8</td>
      <td>8.0</td>
      <td>7.2</td>
      <td>8.0</td>
      <td>7.9</td>
      <td>R</td>
      <td>USA</td>
      <td>22.296514</td>
      <td>8.20</td>
    </tr>
    <tr>
      <th>97</th>
      <td>Star Wars: Episode VII - The Force Awakens</td>
      <td>2015</td>
      <td>245.0</td>
      <td>936.662225</td>
      <td>Doug Walker</td>
      <td>Rob Walker</td>
      <td>0</td>
      <td>131</td>
      <td>12.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>7.9</td>
      <td>7.8</td>
      <td>8.2</td>
      <td>7.7</td>
      <td>8.2</td>
      <td>7.9</td>
      <td>PG-13</td>
      <td>USA</td>
      <td>691.662225</td>
      <td>8.10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arrival</td>
      <td>2016</td>
      <td>47.0</td>
      <td>100.546139</td>
      <td>Amy Adams</td>
      <td>Jeremy Renner</td>
      <td>Forest Whitaker</td>
      <td>35000</td>
      <td>5300.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>7.6</td>
      <td>7.6</td>
      <td>7.7</td>
      <td>7.3</td>
      <td>8.0</td>
      <td>7.9</td>
      <td>PG-13</td>
      <td>USA</td>
      <td>53.546139</td>
      <td>8.05</td>
    </tr>
    <tr>
      <th>33</th>
      <td>The Martian</td>
      <td>2015</td>
      <td>108.0</td>
      <td>228.430993</td>
      <td>Matt Damon</td>
      <td>Donald Glover</td>
      <td>Benedict Wong</td>
      <td>13000</td>
      <td>801.0</td>
      <td>372.0</td>
      <td>...</td>
      <td>8.0</td>
      <td>7.9</td>
      <td>8.2</td>
      <td>7.8</td>
      <td>8.1</td>
      <td>7.9</td>
      <td>PG-13</td>
      <td>USA</td>
      <td>120.430993</td>
      <td>8.00</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Gone Girl</td>
      <td>2014</td>
      <td>61.0</td>
      <td>167.735396</td>
      <td>Patrick Fugit</td>
      <td>Sela Ward</td>
      <td>Emily Ratajkowski</td>
      <td>835</td>
      <td>812.0</td>
      <td>625.0</td>
      <td>...</td>
      <td>7.7</td>
      <td>7.7</td>
      <td>7.7</td>
      <td>7.6</td>
      <td>8.1</td>
      <td>8.1</td>
      <td>R</td>
      <td>USA</td>
      <td>106.735396</td>
      <td>8.00</td>
    </tr>
  </tbody>
</table>
<p>7 rows × 64 columns</p>
</div>



**`Checkpoint 2:`** Can you spot a `Star Wars` movie in your final dataset?

___

### Find the Most Popular Trios - I

Now, lets assume that a producer looking to make a blockbuster movie. There will primarily be three lead roles in that movie and he wish to cast the most popular actors for it. Now, since he don't want to take a risk, he will cast a trio which has already acted in together in a movie before. The metric that you've chosen to check the popularity is the Facebook likes of each of these actors.

The dataframe has three columns to help the producer out for the same, viz. `actor_1_facebook_likes`, `actor_2_facebook_likes`, and `actor_3_facebook_likes`. Your objective is to find the trios which has the most number of Facebook likes combined. That is, the sum of `actor_1_facebook_likes`, `actor_2_facebook_likes` and `actor_3_facebook_likes` should be maximum.
Find out the top 5 popular trios, and output their names in a list.



```python
# Fetching the subset of data in new dataframe to perform the above task by grouping on the actors and summing the likes
trios_on_fbLikes=movies.groupby(
    ['actor_1_name','actor_2_name','actor_3_name'], as_index=False).agg(
    {'actor_1_facebook_likes':'sum','actor_2_facebook_likes':'sum','actor_3_facebook_likes':'sum'})

# Creating new columns 'sum_likes' in the 'best_trios' dataframe
trios_on_fbLikes['sum_likes'] = trios_on_fbLikes['actor_1_facebook_likes'] + trios_on_fbLikes['actor_2_facebook_likes'] + trios_on_fbLikes['actor_3_facebook_likes']

# Using nlargest method to get top 5 trios on sum of facbook likes
top5_trios_on_fbLikes=trios_on_fbLikes.nlargest(5, 'sum_likes', keep='all')

# converting the trios into list
top5_trios_on_fbLikes[['actor_1_name','actor_2_name','actor_3_name']].values.tolist()
```




    [['Dev Patel', 'Nicole Kidman', 'Rooney Mara'],
     ['Leonardo DiCaprio', 'Tom Hardy', 'Joseph Gordon-Levitt'],
     ['Jennifer Lawrence', 'Peter Dinklage', 'Hugh Jackman'],
     ['Casey Affleck', 'Michelle Williams ', 'Kyle Chandler'],
     ['Tom Hardy', 'Christian Bale', 'Joseph Gordon-Levitt']]



### Find the Most Popular Trios - II

Previously we found the popular trio based on the total number of facebook likes. Let's add a small condition to it and make sure that all three actors are popular. The condition is **none of the three actors' Facebook likes should be less than half of the other two**. For example, the following is a valid combo:
- actor_1_facebook_likes: 70000
- actor_2_facebook_likes: 40000
- actor_3_facebook_likes: 50000

But the below one is not:
- actor_1_facebook_likes: 70000
- actor_2_facebook_likes: 40000
- actor_3_facebook_likes: 30000

since in this case, `actor_3_facebook_likes` is 30000, which is less than half of `actor_1_facebook_likes`.

Having this condition ensures that the producer aren't getting any unpopular actor in our trio (since the total likes calculated in the previous question doesn't tell anything about the individual popularities of each actor in the trio.).


- **`No. of trios that satisfy the above condition:`** 

- **`Most popular trio after applying the condition:`**


```python

best_trios_with_above_cond = top5_trios_on_fbLikes[(top5_trios_on_fbLikes['actor_1_facebook_likes']>top5_trios_on_fbLikes[['actor_2_facebook_likes','actor_3_facebook_likes']].max(axis=1)/2) & 
                                          (top5_trios_on_fbLikes['actor_2_facebook_likes']>top5_trios_on_fbLikes[['actor_1_facebook_likes','actor_3_facebook_likes']].max(axis=1)/2) & 
                                          (top5_trios_on_fbLikes['actor_3_facebook_likes']>top5_trios_on_fbLikes[['actor_1_facebook_likes','actor_2_facebook_likes']].max(axis=1)/2)]
print("No. of trios that satisfy the above condition: " + str(best_trios_with_above_cond.shape[0]))
print ("Most popular trios after applying the condition: ")
best_trios_with_above_cond[['actor_1_name','actor_2_name','actor_3_name']].values.tolist()
```

    No. of trios that satisfy the above condition: 3
    Most popular trios after applying the condition: 
    




    [['Leonardo DiCaprio', 'Tom Hardy', 'Joseph Gordon-Levitt'],
     ['Jennifer Lawrence', 'Peter Dinklage', 'Hugh Jackman'],
     ['Tom Hardy', 'Christian Bale', 'Joseph Gordon-Levitt']]




```python
top5_actuall_trios_with_cond=trios_on_fbLikes[(trios_on_fbLikes['actor_1_facebook_likes']>trios_on_fbLikes[['actor_2_facebook_likes','actor_3_facebook_likes']].max(axis=1)/2) & 
                                          (trios_on_fbLikes['actor_2_facebook_likes']>trios_on_fbLikes[['actor_1_facebook_likes','actor_3_facebook_likes']].max(axis=1)/2) & 
                                          (trios_on_fbLikes['actor_3_facebook_likes']>trios_on_fbLikes[['actor_1_facebook_likes','actor_2_facebook_likes']].max(axis=1)/2)]

# No. of trios that satisfy the above condition and are individually popular on the basis of FB Likes
print("No. of trios that satisfy the above condition in the actuall dataset of movies: "+
      str(top5_actuall_trios_with_cond.shape[0]))

# Finding Top5 Trios who are individually popular on the basis of their FB likes
print("Most popular trio after applying the condition on actuall dataset of top100 movies: ")
top5_actuall_trios_with_cond.nlargest(5, 'sum_likes', keep='all')
```

    No. of trios that satisfy the above condition in the actuall dataset of movies: 18
    Most popular trio after applying the condition on actuall dataset of top100 movies: 
    




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>actor_1_name</th>
      <th>actor_2_name</th>
      <th>actor_3_name</th>
      <th>actor_1_facebook_likes</th>
      <th>actor_2_facebook_likes</th>
      <th>actor_3_facebook_likes</th>
      <th>sum_likes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>63</th>
      <td>Leonardo DiCaprio</td>
      <td>Tom Hardy</td>
      <td>Joseph Gordon-Levitt</td>
      <td>29000</td>
      <td>27000.0</td>
      <td>23000.0</td>
      <td>79000.0</td>
    </tr>
    <tr>
      <th>53</th>
      <td>Jennifer Lawrence</td>
      <td>Peter Dinklage</td>
      <td>Hugh Jackman</td>
      <td>34000</td>
      <td>22000.0</td>
      <td>20000.0</td>
      <td>76000.0</td>
    </tr>
    <tr>
      <th>96</th>
      <td>Tom Hardy</td>
      <td>Christian Bale</td>
      <td>Joseph Gordon-Levitt</td>
      <td>27000</td>
      <td>23000.0</td>
      <td>23000.0</td>
      <td>73000.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Chris Hemsworth</td>
      <td>Robert Downey Jr.</td>
      <td>Scarlett Johansson</td>
      <td>26000</td>
      <td>21000.0</td>
      <td>19000.0</td>
      <td>66000.0</td>
    </tr>
    <tr>
      <th>77</th>
      <td>Robert Downey Jr.</td>
      <td>Scarlett Johansson</td>
      <td>Chris Evans</td>
      <td>21000</td>
      <td>19000.0</td>
      <td>11000.0</td>
      <td>51000.0</td>
    </tr>
  </tbody>
</table>
</div>



### Runtime Analysis

There is a column named `Runtime` in the dataframe which primarily shows the length of the movie. It might be intersting to see how this variable is distributed. Let's plot a `histogram` or `distplot` of seaborn to find the `Runtime` range most of the movies fall into.


```python
# Runtime histogram/density plot
movies.Runtime.plot(kind='hist', colormap='rainbow',figsize=(16,6), title='Movies Duration Histogram')
plt.xlabel('Runtime')
plt.ylabel("No's of Movies With Same Duration")
plt.show()
```


![png](./images/output_42_0.png)


**`Checkpoint 3:`** Most of the movies appear to be sharply 2 hour-long.

#### optional plot using seaborn to show each distribution point in the histogram plot
- we can see how most of the movies lies near the 2 hours bucket


```python
plt.figure(figsize=[16,6])
sns.distplot(movies.Runtime,rug=True, rug_kws={"color": "r"}, kde_kws={"color": "b", "lw": 2, "label": "kde"},
             hist_kws={"histtype": "step", "linewidth": 4,"alpha": 1, "color": "g"})
plt.title('Movies Duration with Distribution points')
plt.show()
```


![png](./images/output_45_0.png)


___

### R-Rated Movies

Although R rated movies are restricted movies for the under 18 age group, still there are vote counts from that age group. Among all the R rated movies that have been voted by the under-18 age group, find the top 10 movies that have the highest number of votes i.e.`CVotesU18` from the `movies` dataframe. Store these in a dataframe named `PopularR`.


```python
# Write your code here

PopularR = movies[movies.content_rating == 'R'].groupby(
    ['Title'], as_index=False).agg({'CVotesU18':'sum'}).nlargest(10,'CVotesU18', keep='all')
PopularR
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>CVotesU18</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>Deadpool</td>
      <td>4598</td>
    </tr>
    <tr>
      <th>39</th>
      <td>The Wolf of Wall Street</td>
      <td>3622</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Django Unchained</td>
      <td>3250</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Mad Max: Fury Road</td>
      <td>3159</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Whiplash</td>
      <td>2878</td>
    </tr>
    <tr>
      <th>37</th>
      <td>The Revenant</td>
      <td>2619</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Shutter Island</td>
      <td>2321</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Gone Girl</td>
      <td>2286</td>
    </tr>
    <tr>
      <th>35</th>
      <td>The Grand Budapest Hotel</td>
      <td>2083</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Birdman or (The Unexpected Virtue of Ignorance)</td>
      <td>1891</td>
    </tr>
  </tbody>
</table>
</div>



**`Checkpoint 4:`** Are these kids watching `Deadpool` a lot?

 

___

## Demographic analysis

If you take a look at the last columns in the dataframe, most of these are related to demographics of the voters. We also have three genre columns indicating the genres of a particular movie. We will extensively use these columns for the third and the final stage of our assignment wherein we will analyse the voters across all demographics and also see how these vary across various genres. So without further ado, let's get started with `demographic analysis`.

### Combine the Dataframe by Genres

There are 3 columns in the dataframe - `genre_1`, `genre_2`, and `genre_3`. As a part of this task, we need to aggregate a few values over these 3 columns. 
1. First create a new dataframe `df_by_genre` that contains `genre_1`, `genre_2`, and `genre_3` and all the columns related to **CVotes/Votes** from the `movies` data frame. There are 47 columns to be extracted in total.
2. Now, Add a column called `cnt` to the dataframe `df_by_genre` and initialize it to one.
3. First group the dataframe `df_by_genre` by `genre_1` and find the sum of all the numeric columns such as `cnt`, columns related to CVotes and Votes columns and store it in a dataframe `df_by_g1`.
4. Perform the same operation for `genre_2` and `genre_3` and store it dataframes `df_by_g2` and `df_by_g3` respectively. 
5. Now that we have 3 dataframes performed by grouping over `genre_1`, `genre_2`, and `genre_3` separately, it's time to combine them. For this, add the three dataframes and store it in a new dataframe `df_add`, so that the corresponding values of Votes/CVotes get added for each genre.
6. The column `cnt` on aggregation has basically kept the track of the number of occurences of each genre.Subset the genres that have atleast 10 movies into a new dataframe `genre_top10` based on the `cnt` column value.
7. Now, take the mean of all the numeric columns by dividing them with the column value `cnt` and store it back to the same dataframe. We will be using this dataframe for further analysis in this task unless it is explicitly mentioned to use the dataframe `movies`.
8. Since the number of votes can't be a fraction, type cast all the CVotes related columns to integers. Also, round off all the Votes related columns upto two digits after the decimal point.



```python
# Create the dataframe df_by_genre

df_by_genre = movies.iloc[:,np.r_[11:14, 16:60]]
df_by_genre.head(2)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>genre_1</th>
      <th>genre_2</th>
      <th>genre_3</th>
      <th>CVotes10</th>
      <th>CVotes09</th>
      <th>CVotes08</th>
      <th>CVotes07</th>
      <th>CVotes06</th>
      <th>CVotes05</th>
      <th>CVotes04</th>
      <th>...</th>
      <th>Votes1829F</th>
      <th>Votes3044</th>
      <th>Votes3044M</th>
      <th>Votes3044F</th>
      <th>Votes45A</th>
      <th>Votes45AM</th>
      <th>Votes45AF</th>
      <th>Votes1000</th>
      <th>VotesUS</th>
      <th>VotesnUS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>94</th>
      <td>Drama</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>49673</td>
      <td>62055</td>
      <td>76838</td>
      <td>52238</td>
      <td>23789</td>
      <td>10431</td>
      <td>4906</td>
      <td>...</td>
      <td>7.8</td>
      <td>7.8</td>
      <td>7.8</td>
      <td>7.6</td>
      <td>7.7</td>
      <td>7.7</td>
      <td>7.7</td>
      <td>7.2</td>
      <td>8.0</td>
      <td>7.9</td>
    </tr>
    <tr>
      <th>69</th>
      <td>Biography</td>
      <td>Drama</td>
      <td>History</td>
      <td>75556</td>
      <td>126223</td>
      <td>161460</td>
      <td>83070</td>
      <td>27231</td>
      <td>9603</td>
      <td>4021</td>
      <td>...</td>
      <td>8.2</td>
      <td>8.0</td>
      <td>7.9</td>
      <td>8.0</td>
      <td>7.8</td>
      <td>7.8</td>
      <td>8.1</td>
      <td>7.7</td>
      <td>8.3</td>
      <td>8.0</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 47 columns</p>
</div>




```python
# Create a column cnt and initialize it to 1

df_by_genre['cnt']=1
df_by_genre.head(2)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>genre_1</th>
      <th>genre_2</th>
      <th>genre_3</th>
      <th>CVotes10</th>
      <th>CVotes09</th>
      <th>CVotes08</th>
      <th>CVotes07</th>
      <th>CVotes06</th>
      <th>CVotes05</th>
      <th>CVotes04</th>
      <th>...</th>
      <th>Votes3044</th>
      <th>Votes3044M</th>
      <th>Votes3044F</th>
      <th>Votes45A</th>
      <th>Votes45AM</th>
      <th>Votes45AF</th>
      <th>Votes1000</th>
      <th>VotesUS</th>
      <th>VotesnUS</th>
      <th>cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>94</th>
      <td>Drama</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>49673</td>
      <td>62055</td>
      <td>76838</td>
      <td>52238</td>
      <td>23789</td>
      <td>10431</td>
      <td>4906</td>
      <td>...</td>
      <td>7.8</td>
      <td>7.8</td>
      <td>7.6</td>
      <td>7.7</td>
      <td>7.7</td>
      <td>7.7</td>
      <td>7.2</td>
      <td>8.0</td>
      <td>7.9</td>
      <td>1</td>
    </tr>
    <tr>
      <th>69</th>
      <td>Biography</td>
      <td>Drama</td>
      <td>History</td>
      <td>75556</td>
      <td>126223</td>
      <td>161460</td>
      <td>83070</td>
      <td>27231</td>
      <td>9603</td>
      <td>4021</td>
      <td>...</td>
      <td>8.0</td>
      <td>7.9</td>
      <td>8.0</td>
      <td>7.8</td>
      <td>7.8</td>
      <td>8.1</td>
      <td>7.7</td>
      <td>8.3</td>
      <td>8.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 48 columns</p>
</div>




```python
# Group the movies by individual genres

df_by_g1 = df_by_genre.groupby(['genre_1']).sum()
df_by_g2 = df_by_genre.groupby(['genre_2']).sum()
df_by_g3 = df_by_genre.groupby(['genre_3']).sum()
```


```python
# Add the grouped data frames and store it in a new data frame

df_add = df_by_g1.add(df_by_g2, axis='index', fill_value=0).add(df_by_g3, axis='index', fill_value=0)
df_add.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CVotes10</th>
      <th>CVotes09</th>
      <th>CVotes08</th>
      <th>CVotes07</th>
      <th>CVotes06</th>
      <th>CVotes05</th>
      <th>CVotes04</th>
      <th>CVotes03</th>
      <th>CVotes02</th>
      <th>CVotes01</th>
      <th>...</th>
      <th>Votes3044</th>
      <th>Votes3044M</th>
      <th>Votes3044F</th>
      <th>Votes45A</th>
      <th>Votes45AM</th>
      <th>Votes45AF</th>
      <th>Votes1000</th>
      <th>VotesUS</th>
      <th>VotesnUS</th>
      <th>cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Action</th>
      <td>3166467.0</td>
      <td>3547429.0</td>
      <td>4677755.0</td>
      <td>2922126.0</td>
      <td>1075354.0</td>
      <td>393484.0</td>
      <td>166970.0</td>
      <td>95004.0</td>
      <td>65573.0</td>
      <td>171247.0</td>
      <td>...</td>
      <td>240.0</td>
      <td>239.5</td>
      <td>241.8</td>
      <td>237.0</td>
      <td>236.4</td>
      <td>240.4</td>
      <td>226.2</td>
      <td>247.6</td>
      <td>240.6</td>
      <td>31.0</td>
    </tr>
    <tr>
      <th>Adventure</th>
      <td>3594659.0</td>
      <td>4014192.0</td>
      <td>5262328.0</td>
      <td>3281981.0</td>
      <td>1212075.0</td>
      <td>438970.0</td>
      <td>183070.0</td>
      <td>103318.0</td>
      <td>69737.0</td>
      <td>173858.0</td>
      <td>...</td>
      <td>294.6</td>
      <td>293.7</td>
      <td>299.2</td>
      <td>291.7</td>
      <td>290.4</td>
      <td>298.0</td>
      <td>280.6</td>
      <td>303.5</td>
      <td>296.2</td>
      <td>38.0</td>
    </tr>
    <tr>
      <th>Animation</th>
      <td>681562.0</td>
      <td>798227.0</td>
      <td>1153214.0</td>
      <td>722782.0</td>
      <td>251076.0</td>
      <td>83069.0</td>
      <td>30718.0</td>
      <td>15733.0</td>
      <td>10026.0</td>
      <td>25193.0</td>
      <td>...</td>
      <td>85.4</td>
      <td>84.9</td>
      <td>87.8</td>
      <td>84.5</td>
      <td>84.1</td>
      <td>86.7</td>
      <td>80.0</td>
      <td>87.6</td>
      <td>86.1</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>Biography</th>
      <td>852003.0</td>
      <td>1401608.0</td>
      <td>2231078.0</td>
      <td>1332980.0</td>
      <td>425595.0</td>
      <td>138648.0</td>
      <td>53718.0</td>
      <td>29510.0</td>
      <td>20613.0</td>
      <td>51297.0</td>
      <td>...</td>
      <td>139.1</td>
      <td>138.9</td>
      <td>139.8</td>
      <td>138.5</td>
      <td>137.9</td>
      <td>141.7</td>
      <td>130.1</td>
      <td>142.7</td>
      <td>139.9</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>Comedy</th>
      <td>1383616.0</td>
      <td>1774987.0</td>
      <td>2506851.0</td>
      <td>1591069.0</td>
      <td>600287.0</td>
      <td>226852.0</td>
      <td>97469.0</td>
      <td>56218.0</td>
      <td>39391.0</td>
      <td>88367.0</td>
      <td>...</td>
      <td>177.4</td>
      <td>177.4</td>
      <td>178.3</td>
      <td>175.0</td>
      <td>174.7</td>
      <td>177.1</td>
      <td>165.4</td>
      <td>182.6</td>
      <td>178.9</td>
      <td>23.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 45 columns</p>
</div>




```python
# Extract genres with atleast 10 occurences

genre_top10 = df_add[df_add['cnt']>=10]
genre_top10
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CVotes10</th>
      <th>CVotes09</th>
      <th>CVotes08</th>
      <th>CVotes07</th>
      <th>CVotes06</th>
      <th>CVotes05</th>
      <th>CVotes04</th>
      <th>CVotes03</th>
      <th>CVotes02</th>
      <th>CVotes01</th>
      <th>...</th>
      <th>Votes3044</th>
      <th>Votes3044M</th>
      <th>Votes3044F</th>
      <th>Votes45A</th>
      <th>Votes45AM</th>
      <th>Votes45AF</th>
      <th>Votes1000</th>
      <th>VotesUS</th>
      <th>VotesnUS</th>
      <th>cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Action</th>
      <td>3166467.0</td>
      <td>3547429.0</td>
      <td>4677755.0</td>
      <td>2922126.0</td>
      <td>1075354.0</td>
      <td>393484.0</td>
      <td>166970.0</td>
      <td>95004.0</td>
      <td>65573.0</td>
      <td>171247.0</td>
      <td>...</td>
      <td>240.0</td>
      <td>239.5</td>
      <td>241.8</td>
      <td>237.0</td>
      <td>236.4</td>
      <td>240.4</td>
      <td>226.2</td>
      <td>247.6</td>
      <td>240.6</td>
      <td>31.0</td>
    </tr>
    <tr>
      <th>Adventure</th>
      <td>3594659.0</td>
      <td>4014192.0</td>
      <td>5262328.0</td>
      <td>3281981.0</td>
      <td>1212075.0</td>
      <td>438970.0</td>
      <td>183070.0</td>
      <td>103318.0</td>
      <td>69737.0</td>
      <td>173858.0</td>
      <td>...</td>
      <td>294.6</td>
      <td>293.7</td>
      <td>299.2</td>
      <td>291.7</td>
      <td>290.4</td>
      <td>298.0</td>
      <td>280.6</td>
      <td>303.5</td>
      <td>296.2</td>
      <td>38.0</td>
    </tr>
    <tr>
      <th>Animation</th>
      <td>681562.0</td>
      <td>798227.0</td>
      <td>1153214.0</td>
      <td>722782.0</td>
      <td>251076.0</td>
      <td>83069.0</td>
      <td>30718.0</td>
      <td>15733.0</td>
      <td>10026.0</td>
      <td>25193.0</td>
      <td>...</td>
      <td>85.4</td>
      <td>84.9</td>
      <td>87.8</td>
      <td>84.5</td>
      <td>84.1</td>
      <td>86.7</td>
      <td>80.0</td>
      <td>87.6</td>
      <td>86.1</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>Biography</th>
      <td>852003.0</td>
      <td>1401608.0</td>
      <td>2231078.0</td>
      <td>1332980.0</td>
      <td>425595.0</td>
      <td>138648.0</td>
      <td>53718.0</td>
      <td>29510.0</td>
      <td>20613.0</td>
      <td>51297.0</td>
      <td>...</td>
      <td>139.1</td>
      <td>138.9</td>
      <td>139.8</td>
      <td>138.5</td>
      <td>137.9</td>
      <td>141.7</td>
      <td>130.1</td>
      <td>142.7</td>
      <td>139.9</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>Comedy</th>
      <td>1383616.0</td>
      <td>1774987.0</td>
      <td>2506851.0</td>
      <td>1591069.0</td>
      <td>600287.0</td>
      <td>226852.0</td>
      <td>97469.0</td>
      <td>56218.0</td>
      <td>39391.0</td>
      <td>88367.0</td>
      <td>...</td>
      <td>177.4</td>
      <td>177.4</td>
      <td>178.3</td>
      <td>175.0</td>
      <td>174.7</td>
      <td>177.1</td>
      <td>165.4</td>
      <td>182.6</td>
      <td>178.9</td>
      <td>23.0</td>
    </tr>
    <tr>
      <th>Crime</th>
      <td>574526.0</td>
      <td>967118.0</td>
      <td>1419495.0</td>
      <td>821390.0</td>
      <td>278391.0</td>
      <td>98690.0</td>
      <td>42271.0</td>
      <td>24713.0</td>
      <td>16985.0</td>
      <td>37217.0</td>
      <td>...</td>
      <td>84.9</td>
      <td>85.4</td>
      <td>83.7</td>
      <td>83.9</td>
      <td>83.8</td>
      <td>84.5</td>
      <td>81.3</td>
      <td>87.8</td>
      <td>85.8</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>Drama</th>
      <td>3404438.0</td>
      <td>4935375.0</td>
      <td>7107053.0</td>
      <td>4319700.0</td>
      <td>1529356.0</td>
      <td>552312.0</td>
      <td>235475.0</td>
      <td>135126.0</td>
      <td>94185.0</td>
      <td>211308.0</td>
      <td>...</td>
      <td>501.3</td>
      <td>501.1</td>
      <td>501.8</td>
      <td>496.8</td>
      <td>495.3</td>
      <td>503.2</td>
      <td>469.5</td>
      <td>515.9</td>
      <td>506.0</td>
      <td>65.0</td>
    </tr>
    <tr>
      <th>Romance</th>
      <td>549959.0</td>
      <td>689492.0</td>
      <td>1069280.0</td>
      <td>712841.0</td>
      <td>281289.0</td>
      <td>110901.0</td>
      <td>48913.0</td>
      <td>27698.0</td>
      <td>19200.0</td>
      <td>40075.0</td>
      <td>...</td>
      <td>98.9</td>
      <td>98.9</td>
      <td>99.6</td>
      <td>97.8</td>
      <td>97.5</td>
      <td>98.9</td>
      <td>89.9</td>
      <td>101.8</td>
      <td>100.1</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>Sci-Fi</th>
      <td>2325284.0</td>
      <td>2530855.0</td>
      <td>3002994.0</td>
      <td>1802098.0</td>
      <td>671811.0</td>
      <td>254175.0</td>
      <td>111925.0</td>
      <td>65904.0</td>
      <td>46171.0</td>
      <td>114435.0</td>
      <td>...</td>
      <td>133.6</td>
      <td>133.5</td>
      <td>133.2</td>
      <td>131.1</td>
      <td>130.8</td>
      <td>131.5</td>
      <td>127.9</td>
      <td>137.5</td>
      <td>134.0</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>Thriller</th>
      <td>1081701.0</td>
      <td>1465491.0</td>
      <td>1993378.0</td>
      <td>1175799.0</td>
      <td>416046.0</td>
      <td>149953.0</td>
      <td>65281.0</td>
      <td>37940.0</td>
      <td>25767.0</td>
      <td>57630.0</td>
      <td>...</td>
      <td>100.6</td>
      <td>100.7</td>
      <td>100.1</td>
      <td>99.6</td>
      <td>99.3</td>
      <td>100.7</td>
      <td>96.2</td>
      <td>103.1</td>
      <td>101.5</td>
      <td>13.0</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 45 columns</p>
</div>




```python
# Take the mean for every column by dividing with cnt 
genre_top10.iloc[:,0:44] = genre_top10.iloc[:,0:44].div(genre_top10.cnt, axis='index')
```


```python
# Rounding off the columns of Votes to two decimals

genre_top10.iloc[:,27:44] = round(genre_top10.iloc[:,27:44],2)
genre_top10.head(5)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CVotes10</th>
      <th>CVotes09</th>
      <th>CVotes08</th>
      <th>CVotes07</th>
      <th>CVotes06</th>
      <th>CVotes05</th>
      <th>CVotes04</th>
      <th>CVotes03</th>
      <th>CVotes02</th>
      <th>CVotes01</th>
      <th>...</th>
      <th>Votes3044</th>
      <th>Votes3044M</th>
      <th>Votes3044F</th>
      <th>Votes45A</th>
      <th>Votes45AM</th>
      <th>Votes45AF</th>
      <th>Votes1000</th>
      <th>VotesUS</th>
      <th>VotesnUS</th>
      <th>cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Action</th>
      <td>102144.096774</td>
      <td>114433.193548</td>
      <td>150895.322581</td>
      <td>94262.129032</td>
      <td>34688.838710</td>
      <td>12693.032258</td>
      <td>5386.129032</td>
      <td>3064.645161</td>
      <td>2115.258065</td>
      <td>5524.096774</td>
      <td>...</td>
      <td>7.74</td>
      <td>7.73</td>
      <td>7.80</td>
      <td>7.65</td>
      <td>7.63</td>
      <td>7.75</td>
      <td>7.30</td>
      <td>7.99</td>
      <td>7.76</td>
      <td>31.0</td>
    </tr>
    <tr>
      <th>Adventure</th>
      <td>94596.289474</td>
      <td>105636.631579</td>
      <td>138482.315789</td>
      <td>86367.921053</td>
      <td>31896.710526</td>
      <td>11551.842105</td>
      <td>4817.631579</td>
      <td>2718.894737</td>
      <td>1835.184211</td>
      <td>4575.210526</td>
      <td>...</td>
      <td>7.75</td>
      <td>7.73</td>
      <td>7.87</td>
      <td>7.68</td>
      <td>7.64</td>
      <td>7.84</td>
      <td>7.38</td>
      <td>7.99</td>
      <td>7.79</td>
      <td>38.0</td>
    </tr>
    <tr>
      <th>Animation</th>
      <td>61960.181818</td>
      <td>72566.090909</td>
      <td>104837.636364</td>
      <td>65707.454545</td>
      <td>22825.090909</td>
      <td>7551.727273</td>
      <td>2792.545455</td>
      <td>1430.272727</td>
      <td>911.454545</td>
      <td>2290.272727</td>
      <td>...</td>
      <td>7.76</td>
      <td>7.72</td>
      <td>7.98</td>
      <td>7.68</td>
      <td>7.65</td>
      <td>7.88</td>
      <td>7.27</td>
      <td>7.96</td>
      <td>7.83</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>Biography</th>
      <td>47333.500000</td>
      <td>77867.111111</td>
      <td>123948.777778</td>
      <td>74054.444444</td>
      <td>23644.166667</td>
      <td>7702.666667</td>
      <td>2984.333333</td>
      <td>1639.444444</td>
      <td>1145.166667</td>
      <td>2849.833333</td>
      <td>...</td>
      <td>7.73</td>
      <td>7.72</td>
      <td>7.77</td>
      <td>7.69</td>
      <td>7.66</td>
      <td>7.87</td>
      <td>7.23</td>
      <td>7.93</td>
      <td>7.77</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>Comedy</th>
      <td>60157.217391</td>
      <td>77173.347826</td>
      <td>108993.521739</td>
      <td>69176.913043</td>
      <td>26099.434783</td>
      <td>9863.130435</td>
      <td>4237.782609</td>
      <td>2444.260870</td>
      <td>1712.652174</td>
      <td>3842.043478</td>
      <td>...</td>
      <td>7.71</td>
      <td>7.71</td>
      <td>7.75</td>
      <td>7.61</td>
      <td>7.60</td>
      <td>7.70</td>
      <td>7.19</td>
      <td>7.94</td>
      <td>7.78</td>
      <td>23.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 45 columns</p>
</div>




```python
# Converting CVotes to int type

genre_top10.iloc[:,0:27] = genre_top10.iloc[:,0:27].astype(np.int32)
genre_top10.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CVotes10</th>
      <th>CVotes09</th>
      <th>CVotes08</th>
      <th>CVotes07</th>
      <th>CVotes06</th>
      <th>CVotes05</th>
      <th>CVotes04</th>
      <th>CVotes03</th>
      <th>CVotes02</th>
      <th>CVotes01</th>
      <th>...</th>
      <th>Votes3044</th>
      <th>Votes3044M</th>
      <th>Votes3044F</th>
      <th>Votes45A</th>
      <th>Votes45AM</th>
      <th>Votes45AF</th>
      <th>Votes1000</th>
      <th>VotesUS</th>
      <th>VotesnUS</th>
      <th>cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Action</th>
      <td>102144.0</td>
      <td>114433.0</td>
      <td>150895.0</td>
      <td>94262.0</td>
      <td>34688.0</td>
      <td>12693.0</td>
      <td>5386.0</td>
      <td>3064.0</td>
      <td>2115.0</td>
      <td>5524.0</td>
      <td>...</td>
      <td>7.74</td>
      <td>7.73</td>
      <td>7.80</td>
      <td>7.65</td>
      <td>7.63</td>
      <td>7.75</td>
      <td>7.30</td>
      <td>7.99</td>
      <td>7.76</td>
      <td>31.0</td>
    </tr>
    <tr>
      <th>Adventure</th>
      <td>94596.0</td>
      <td>105636.0</td>
      <td>138482.0</td>
      <td>86367.0</td>
      <td>31896.0</td>
      <td>11551.0</td>
      <td>4817.0</td>
      <td>2718.0</td>
      <td>1835.0</td>
      <td>4575.0</td>
      <td>...</td>
      <td>7.75</td>
      <td>7.73</td>
      <td>7.87</td>
      <td>7.68</td>
      <td>7.64</td>
      <td>7.84</td>
      <td>7.38</td>
      <td>7.99</td>
      <td>7.79</td>
      <td>38.0</td>
    </tr>
    <tr>
      <th>Animation</th>
      <td>61960.0</td>
      <td>72566.0</td>
      <td>104837.0</td>
      <td>65707.0</td>
      <td>22825.0</td>
      <td>7551.0</td>
      <td>2792.0</td>
      <td>1430.0</td>
      <td>911.0</td>
      <td>2290.0</td>
      <td>...</td>
      <td>7.76</td>
      <td>7.72</td>
      <td>7.98</td>
      <td>7.68</td>
      <td>7.65</td>
      <td>7.88</td>
      <td>7.27</td>
      <td>7.96</td>
      <td>7.83</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>Biography</th>
      <td>47333.0</td>
      <td>77867.0</td>
      <td>123948.0</td>
      <td>74054.0</td>
      <td>23644.0</td>
      <td>7702.0</td>
      <td>2984.0</td>
      <td>1639.0</td>
      <td>1145.0</td>
      <td>2849.0</td>
      <td>...</td>
      <td>7.73</td>
      <td>7.72</td>
      <td>7.77</td>
      <td>7.69</td>
      <td>7.66</td>
      <td>7.87</td>
      <td>7.23</td>
      <td>7.93</td>
      <td>7.77</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>Comedy</th>
      <td>60157.0</td>
      <td>77173.0</td>
      <td>108993.0</td>
      <td>69176.0</td>
      <td>26099.0</td>
      <td>9863.0</td>
      <td>4237.0</td>
      <td>2444.0</td>
      <td>1712.0</td>
      <td>3842.0</td>
      <td>...</td>
      <td>7.71</td>
      <td>7.71</td>
      <td>7.75</td>
      <td>7.61</td>
      <td>7.60</td>
      <td>7.70</td>
      <td>7.19</td>
      <td>7.94</td>
      <td>7.78</td>
      <td>23.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 45 columns</p>
</div>



If you take a look at the final dataframe that you have gotten, you will see that you now have the complete information about all the demographic (Votes- and CVotes-related) columns across the top 10 genres. We can use this dataset to extract exciting insights about the voters!

### Genre Counts!

Now let's derive some insights from this data frame.


```python
# Countplot for genres

plt.figure(figsize=(12,6))
sns.barplot(x=genre_top10.index, y=genre_top10.cnt)
plt.title('Genre Counts')
plt.ylabel('Counts Among Top 100 Movies')
plt.xlabel('Top 10 Genre')
plt.show()
```


![png](./images/output_64_0.png)


**`Checkpoint 5:`** Is the bar for `Drama` the tallest?

### Gender and Genre

If you have closely looked at the Votes- and CVotes-related columns, you might have noticed the suffixes `F` and `M` indicating Female and Male. Since we have the vote counts for both males and females, across various age groups, let's now see how the popularity of genres vary between the two genders in the dataframe. 

1. Make the first heatmap to see how the average number of votes of males is varying across the genres. Use seaborn heatmap for this analysis. The X-axis should contain the four age-groups for males, i.e., `CVotesU18M`,`CVotes1829M`, `CVotes3044M`, and `CVotes45AM`. The Y-axis will have the genres and the annotation in the heatmap tell the average number of votes for that age-male group. 

2. Make the second heatmap to see how the average number of votes of females is varying across the genres.

3. Make sure that we plot these heatmaps side by side using `subplots` so that you can easily compare the two genders and derive insights.

4. Write your any three inferences from this plot. You can make use of the previous bar plot also here for better insights.

5. Now instead of taking the CVotes-related columns, we need to do the same process for the Votes-related columns. These heatmaps will show us how the two genders have rated movies across various genres.


```python
genre_top10.index.name='genre'
genre_top10
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CVotes10</th>
      <th>CVotes09</th>
      <th>CVotes08</th>
      <th>CVotes07</th>
      <th>CVotes06</th>
      <th>CVotes05</th>
      <th>CVotes04</th>
      <th>CVotes03</th>
      <th>CVotes02</th>
      <th>CVotes01</th>
      <th>...</th>
      <th>Votes3044</th>
      <th>Votes3044M</th>
      <th>Votes3044F</th>
      <th>Votes45A</th>
      <th>Votes45AM</th>
      <th>Votes45AF</th>
      <th>Votes1000</th>
      <th>VotesUS</th>
      <th>VotesnUS</th>
      <th>cnt</th>
    </tr>
    <tr>
      <th>genre</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Action</th>
      <td>102144.0</td>
      <td>114433.0</td>
      <td>150895.0</td>
      <td>94262.0</td>
      <td>34688.0</td>
      <td>12693.0</td>
      <td>5386.0</td>
      <td>3064.0</td>
      <td>2115.0</td>
      <td>5524.0</td>
      <td>...</td>
      <td>7.74</td>
      <td>7.73</td>
      <td>7.80</td>
      <td>7.65</td>
      <td>7.63</td>
      <td>7.75</td>
      <td>7.30</td>
      <td>7.99</td>
      <td>7.76</td>
      <td>31.0</td>
    </tr>
    <tr>
      <th>Adventure</th>
      <td>94596.0</td>
      <td>105636.0</td>
      <td>138482.0</td>
      <td>86367.0</td>
      <td>31896.0</td>
      <td>11551.0</td>
      <td>4817.0</td>
      <td>2718.0</td>
      <td>1835.0</td>
      <td>4575.0</td>
      <td>...</td>
      <td>7.75</td>
      <td>7.73</td>
      <td>7.87</td>
      <td>7.68</td>
      <td>7.64</td>
      <td>7.84</td>
      <td>7.38</td>
      <td>7.99</td>
      <td>7.79</td>
      <td>38.0</td>
    </tr>
    <tr>
      <th>Animation</th>
      <td>61960.0</td>
      <td>72566.0</td>
      <td>104837.0</td>
      <td>65707.0</td>
      <td>22825.0</td>
      <td>7551.0</td>
      <td>2792.0</td>
      <td>1430.0</td>
      <td>911.0</td>
      <td>2290.0</td>
      <td>...</td>
      <td>7.76</td>
      <td>7.72</td>
      <td>7.98</td>
      <td>7.68</td>
      <td>7.65</td>
      <td>7.88</td>
      <td>7.27</td>
      <td>7.96</td>
      <td>7.83</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>Biography</th>
      <td>47333.0</td>
      <td>77867.0</td>
      <td>123948.0</td>
      <td>74054.0</td>
      <td>23644.0</td>
      <td>7702.0</td>
      <td>2984.0</td>
      <td>1639.0</td>
      <td>1145.0</td>
      <td>2849.0</td>
      <td>...</td>
      <td>7.73</td>
      <td>7.72</td>
      <td>7.77</td>
      <td>7.69</td>
      <td>7.66</td>
      <td>7.87</td>
      <td>7.23</td>
      <td>7.93</td>
      <td>7.77</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>Comedy</th>
      <td>60157.0</td>
      <td>77173.0</td>
      <td>108993.0</td>
      <td>69176.0</td>
      <td>26099.0</td>
      <td>9863.0</td>
      <td>4237.0</td>
      <td>2444.0</td>
      <td>1712.0</td>
      <td>3842.0</td>
      <td>...</td>
      <td>7.71</td>
      <td>7.71</td>
      <td>7.75</td>
      <td>7.61</td>
      <td>7.60</td>
      <td>7.70</td>
      <td>7.19</td>
      <td>7.94</td>
      <td>7.78</td>
      <td>23.0</td>
    </tr>
    <tr>
      <th>Crime</th>
      <td>52229.0</td>
      <td>87919.0</td>
      <td>129045.0</td>
      <td>74671.0</td>
      <td>25308.0</td>
      <td>8971.0</td>
      <td>3842.0</td>
      <td>2246.0</td>
      <td>1544.0</td>
      <td>3383.0</td>
      <td>...</td>
      <td>7.72</td>
      <td>7.76</td>
      <td>7.61</td>
      <td>7.63</td>
      <td>7.62</td>
      <td>7.68</td>
      <td>7.39</td>
      <td>7.98</td>
      <td>7.80</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>Drama</th>
      <td>52375.0</td>
      <td>75928.0</td>
      <td>109339.0</td>
      <td>66456.0</td>
      <td>23528.0</td>
      <td>8497.0</td>
      <td>3622.0</td>
      <td>2078.0</td>
      <td>1449.0</td>
      <td>3250.0</td>
      <td>...</td>
      <td>7.71</td>
      <td>7.71</td>
      <td>7.72</td>
      <td>7.64</td>
      <td>7.62</td>
      <td>7.74</td>
      <td>7.22</td>
      <td>7.94</td>
      <td>7.78</td>
      <td>65.0</td>
    </tr>
    <tr>
      <th>Romance</th>
      <td>42304.0</td>
      <td>53037.0</td>
      <td>82252.0</td>
      <td>54833.0</td>
      <td>21637.0</td>
      <td>8530.0</td>
      <td>3762.0</td>
      <td>2130.0</td>
      <td>1476.0</td>
      <td>3082.0</td>
      <td>...</td>
      <td>7.61</td>
      <td>7.61</td>
      <td>7.66</td>
      <td>7.52</td>
      <td>7.50</td>
      <td>7.61</td>
      <td>6.92</td>
      <td>7.83</td>
      <td>7.70</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>Sci-Fi</th>
      <td>136781.0</td>
      <td>148873.0</td>
      <td>176646.0</td>
      <td>106005.0</td>
      <td>39518.0</td>
      <td>14951.0</td>
      <td>6583.0</td>
      <td>3876.0</td>
      <td>2715.0</td>
      <td>6731.0</td>
      <td>...</td>
      <td>7.86</td>
      <td>7.85</td>
      <td>7.84</td>
      <td>7.71</td>
      <td>7.69</td>
      <td>7.74</td>
      <td>7.52</td>
      <td>8.09</td>
      <td>7.88</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>Thriller</th>
      <td>83207.0</td>
      <td>112730.0</td>
      <td>153336.0</td>
      <td>90446.0</td>
      <td>32003.0</td>
      <td>11534.0</td>
      <td>5021.0</td>
      <td>2918.0</td>
      <td>1982.0</td>
      <td>4433.0</td>
      <td>...</td>
      <td>7.74</td>
      <td>7.75</td>
      <td>7.70</td>
      <td>7.66</td>
      <td>7.64</td>
      <td>7.75</td>
      <td>7.40</td>
      <td>7.93</td>
      <td>7.81</td>
      <td>13.0</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 45 columns</p>
</div>




```python
# 1st set of heat maps for CVotes-related columns

plt.figure(figsize=(16,6))

plt.subplot(1,2,1)

ax1=sns.heatmap(genre_top10[['CVotesU18M','CVotes1829M','CVotes3044M','CVotes45AM']],
                cmap='Greens', annot=True, fmt='g')

plt.subplot(1,2,2)

ax2 =sns.heatmap(genre_top10[['CVotesU18F','CVotes1829F','CVotes3044F','CVotes45AF']],
                 cmap='Reds', annot=True, fmt='g')

```


![png](./images/output_68_0.png)


**`Inferences:`** A few inferences that can be seen from the heatmap above is that males have voted more than females, and Sci-Fi appears to be most popular among the 18-29 age group irrespective of their gender.


```python
# 2nd set of heat maps for Votes-related columns

plt.figure(figsize=(16,6))
plt.subplot(1,2,1)

ax1=sns.heatmap(genre_top10[['VotesU18M','Votes1829M','Votes3044M','Votes45AM']], 
                cmap='Greens', annot=True)

plt.subplot(1,2,2)

ax2 =sns.heatmap(genre_top10[['VotesU18F','Votes1829F','Votes3044F','Votes45AF']], 
                 cmap='Reds', annot=True)
```


![png](./images/output_70_0.png)


**`Inferences:`** Sci-Fi appears to be the highest rated genre in the age group of U18 for both males and females. Also, females in this age group have rated it a bit higher than the males in the same age group.

### US vs non-US Cross Analysis

The dataset contains both the US and non-US movies. Let's analyse how both the US and the non-US voters have responded to the US and the non-US movies.

1. Create a column `IFUS` in the dataframe `movies`. The column `IFUS` should contain the value "USA" if the `Country` of the movie is "USA". For all other countries other than the USA, `IFUS` should contain the value `non-USA`.


2. Now make a boxplot that shows how the number of votes from the US people i.e. `CVotesUS` is varying for the US and non-US movies. Make use of the column `IFUS` to make this plot. Similarly, make another subplot that shows how non US voters have voted for the US and non-US movies by plotting `CVotesnUS` for both the US and non-US movies. Write any of your two inferences/observations from these plots.


3. Again do a similar analysis but with the ratings. Make a boxplot that shows how the ratings from the US people i.e. `VotesUS` is varying for the US and non-US movies. Similarly, make another subplot that shows how `VotesnUS` is varying for the US and non-US movies. Write any of your two inferences/observations from these plots.

Note : Use `movies` dataframe for this subtask. Make use of this documention to format your boxplot - https://seaborn.pydata.org/generated/seaborn.boxplot.html


```python
# Creating IFUS column

movies['IFUS']=movies.Country.apply(lambda x: x if x=='USA' else 'non-USA')
movies.head(2)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>title_year</th>
      <th>budget</th>
      <th>Gross</th>
      <th>actor_1_name</th>
      <th>actor_2_name</th>
      <th>actor_3_name</th>
      <th>actor_1_facebook_likes</th>
      <th>actor_2_facebook_likes</th>
      <th>actor_3_facebook_likes</th>
      <th>...</th>
      <th>Votes45AM</th>
      <th>Votes45AF</th>
      <th>Votes1000</th>
      <th>VotesUS</th>
      <th>VotesnUS</th>
      <th>content_rating</th>
      <th>Country</th>
      <th>profit</th>
      <th>Avg_rating</th>
      <th>IFUS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>94</th>
      <td>Boyhood</td>
      <td>2014</td>
      <td>4.0</td>
      <td>25.35920</td>
      <td>Ellar Coltrane</td>
      <td>Lorelei Linklater</td>
      <td>Libby Villari</td>
      <td>230</td>
      <td>193.0</td>
      <td>127.0</td>
      <td>...</td>
      <td>7.7</td>
      <td>7.7</td>
      <td>7.2</td>
      <td>8.0</td>
      <td>7.9</td>
      <td>R</td>
      <td>USA</td>
      <td>21.35920</td>
      <td>8.95</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>69</th>
      <td>12 Years a Slave</td>
      <td>2013</td>
      <td>20.0</td>
      <td>56.66787</td>
      <td>QuvenzhanÃ© Wallis</td>
      <td>Scoot McNairy</td>
      <td>Taran Killam</td>
      <td>2000</td>
      <td>660.0</td>
      <td>500.0</td>
      <td>...</td>
      <td>7.8</td>
      <td>8.1</td>
      <td>7.7</td>
      <td>8.3</td>
      <td>8.0</td>
      <td>R</td>
      <td>USA</td>
      <td>36.66787</td>
      <td>8.85</td>
      <td>USA</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 65 columns</p>
</div>




```python
# Box plot - 1: CVotesUS(y) vs IFUS(x)

plt.figure(figsize=(16,6))

plt.subplot(1,2,1)
sns.boxplot(y='CVotesUS', x='IFUS', data =movies)

plt.subplot(1,2,2)
sns.boxplot(y='CVotesnUS', x='IFUS', data =movies)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1db376f9e48>




![png](./images/output_74_1.png)



```python
# Box plot - 2: VotesUS(y) vs IFUS(x)

plt.figure(figsize=(16,6))

plt.subplot(1,2,1)
sns.boxplot(y='VotesUS', x='IFUS', data =movies)

plt.subplot(1,2,2)
sns.boxplot(y='VotesnUS', x='IFUS', data =movies)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1db379e7848>




![png](./images/output_75_1.png)


###  Subtask 3.5:  Top 1000 Voters Vs Genres

You might have also observed the column `CVotes1000`. This column represents the top 1000 voters on IMDb and gives the count for the number of these voters who have voted for a particular movie. Let's see how these top 1000 voters have voted across the genres. 

1. Sort the dataframe genre_top10 based on the value of `CVotes1000`in a descending order.

2. Make a seaborn barplot for `genre` vs `CVotes1000`.

3. Write your inferences. You can also try to relate it with the heatmaps you did in the previous subtasks.



```python
# Sorting by CVotes1000

genre_top10.sort_values(by='CVotes1000', ascending=False, inplace=True)
```


```python
# Bar plot
plt.figure(figsize=(16,6))

sns.barplot(x=genre_top10.index, y=genre_top10.CVotes1000)

plt.xlabel("Top 10 Genres")
plt.ylabel("Top 1000 Voters")
plt.title("Top 1000 Voters Vs Genres")

plt.show()
```


![png](./images/output_78_0.png)


**`Checkpoint 6:`** The genre `Romance` seems to be most unpopular among the top 1000 voters.

**_Thanks!_**




```python

```
