#Filter and select data

import numpy as np
import pandas as pd

from pandas import Series, DataFrame

series_obj= Series(np.arange(8),index=['row 1','row 2','row 3','row 4','row 5','row 6','row 7','row 8',])
print(series_obj)
print(series_obj['row 3'])
print(series_obj[[0,7]])

np.random.seed(25)
DF_obj= DataFrame(np.random.rand(36).reshape(6,6),index=['row 1','row 2','row 3','row 4','row 5','row 6'],columns=['column 1','column 2','column 3','column 4','column 5','column 6'])
print(DF_obj)
print(DF_obj.ix[['row 1','row 2'],['column 2','column 4']])

print(series_obj['row 3':'row 5'])
print(DF_obj < .2)
print(series_obj[series_obj > 6])
series_obj['row 1','row 5','row 8']=8

#============================================================================

#Treating missing data

missing = np.nan

series_obj = Series(['row 1','row 2',missing, 'row 4','row 5','row 6',missing,'row 8'])
print(series_obj)
print(series_obj.isnull())

np.random.seed(25)
DF_obj=DataFrame(np.random.randn(36).reshape(6,6))
DF_obj.ix[3:5,0]=missing
DF_obj.ix[1:4,5]=missing
filled_DF=DF_obj.fillna(0)
filled_DF=DF_obj.fillna({0:0.1,5:1.25})
filled_DF=DF_obj.fillna(method='ffill')
print(DF_obj.isnull().sum())
DF_no_NaN_row= DF_obj.dropna()
DF_no_NaN_column= DF_obj.dropna(axis=1)
DF_all_NaN= DF_obj.dropna(how='all')

#=============================================================================

#Remove duplicates

DF_obj = DataFrame({'column 1':[1,1,2,2,3,3,3],                    'column 2':['a','a','b','b','c','c','c'],
                    'column 3':['A','A','B','B','C','C','C']})
print(DF_obj.duplicated())
print(DF_obj.drop_duplicates())
DF_obj = DataFrame({'column 1':[1,1,2,2,3,3,3],
                    'column 2':['a','a','b','b','c','c','c'],
                    'column 3':['A','A','B','B','C','D','C']})
print(DF_obj.drop_duplicates(['column 3']))

#=============================================================================

#Concatenating and trasform data

DF_obj = pd.DataFrame(np.arange(36).reshape(6,6))
DF_obj_2 = pd.DataFrame(np.arange(15).reshape(5,3))
print(pd.concat([DF_obj,DF_obj_2],axis=1))
print(pd.concat([DF_obj,DF_obj_2]))
print(DF_obj.drop([0,2]))
print(DF_obj.drop([0,2],axis=1))
series_obj=Series(np.arange(6))
series_obj.name = "added_variable"
variable_added = DataFrame.join(DF_obj,series_obj)
added_datatable= variable_added.append(variable_added, ignore_index=True)
DF_sorted= DF_obj.sort_values(by=[5],ascending=[False])

#=============================================================================

#Grouping and aggregation

address = '/Users/clu128/Desktop/Exercise Files/Ch01/01_05/mtcars.csv'
cars = pd.read_csv(address)

cars.column = ['car_names','mpg','cyl','disp','hp','drat','wt','qsec','vs','am','gear','carb']
print(cars.head())
cars_groups = cars.groupby(cars['cyl'])
print(cars_groups.mean())

