#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Web-based data visualizations with Plotly

# traces, layouts

import numpy as np
import pandas as pd

import cufflinks as cf

import plotly.plotly as py
import plotly.tools as tls
import plotly.graph_objs as go

import sklearn
from sklearn.preprocessing import StandardScaler

tls.set_credentials_file(username='carajumpshigh',api_key='kszxsKpox0NzQCjbU4iy')

a = np.linspace(start=0,stop=36,num=36)
np.random.seed(25)
b =np.random.uniform(low=0.0,high=1.0,size=36)

trace = go.Scatter(x=a,y=b)
data = [trace]
py.iplot(data,file='basic-line-chart')     #view on Plotly.com

x = [1,2,3,4,5,6,7,8,9]
y = [1,2,3,4,0,4,3,2,1]
z = [10,9,8,7,6,5,4,3,2,1]

trace0 = go.Scatter(x=x,y=y,name='List Object',line=dict(width=5))
trace1 = go.Scatter(x=x,y=z,name='List Object 2',line=dict(width=20))
data = [trace0,trace1]
layout = dict(title='Double Line Chart',xaxis=dict(title='x-axis'),yaxis=dict(title='y-axis'))
fig = dict(data=data,layout=layout)
py.iplot(fig,filename='style-line-chart')

address = '/Users/clu128/Desktop/Exercise Files/Ch09/09_01/mtcars.csv'
cars = pd.read_csv(address)
cars.columns = ['car_names','mpg','cyl','disp','hp','drat','wt','qsec','vs','am','gear','carb']
df = cars[['cyl','wt','mpg']]
layout = dict(title = 'Chart From Pandas DataFrame',xaxis=dict(title='x-axis'),yaxis=dict(title='y-axis'))
df.iplot(filename='cf-simple-line-chart',layout=layout)

data = [go.Bar(x=[1,2,3,4,5,6,7,8,9,10],y=[1,2,3,4,0.5,4,3,2,1])]
layout = dict(title='Simple Bar Chart',
              xaxis = dict(title='x-axis'),
              yaxis = dict(title='y-axis'))
py.iplot(data,filename='basic-bar-chart',layout=layout)
color_theme = dict(color=['rgba(169,169,169,1','rgba(255,160,122,1)','rgba(176,224,230,1)','rgba(255,228,196,1)',
                                'rgba(189,183,107,1)','rgba(188,143,143,1)','rgba(221,160,221,1)'])
trace0 = go.Bar(x=[1,2,3,4,5,6,7],y=[1,2,3,4,0.5,3,1],marker=color_theme)
data = [trace0]
layout = go.Layout(title='Custom Colors')
fig = go.Figure(data=data,layout=layout)
py.iplot(fig,filename='color-bar-chart')

fig = {'data':[{'labels':['bicycle','motorbike','car','van','stroller'],
                'values':[1,2,3,4,0.5],'type':'pie'}],
        'layout': {'title':'Simple Pie Chart'}}
py.iplot(fig)


#=============================================================================


# statistical plots in Plotly

mpg = cars.mpg
mpg.iplot(kind='histogram',filename='simple-histogram-chart')

cars_data = cars.ix[:,(1,3,4)].valuescars_data_std = StandardScaler().fit_transform(cars_data)
cars_select = pd.DataFrame(cars_data_std)
cars_select.columns = ['mpg','disp','hp']
cars_select.iplot(kind='histogram',filename='multiple-histogram-chart')
cars_select.iplot(kind='histogram',subplots=True,filename='subplot-histograms')
cars_select.iplot(kind='histogram',subplots=True,shape=(3,1),filename='subplot-histograms')

cars_select.iplot(kind='box',filename='box-plots')

fig = {'data':[{'x':cars_select.mpg,'y':cars_select.disp,'mode':'markers','name':'mpg'},
               {'x':cars_select.hp,'y':cars_select.disp,'mode':'markers','name':'hp'}],
        'layout':{'xaxis':{'title':''},'yaxis':{'title':'Standardized Displacement'}}}
py.iplot(fig,filename='grouped-scatter-plot')


#=============================================================================


# geographic maps in Plotly

# choropleth maps(code for position;set parameter), point map(x,y posotion;marker&data parameter)

address = '/Users/clu128/Desktop/Exercise Files/Ch09/09_03/States.csv'
states = pd.read_csv(address)
states.columns = ['code','region','pop','satv','satm','percent','dollars','pay']
states.head()

states['text'] = 'SATv'+states['satv'].astype(str)+'SATm'+states['satm'].astype(str)+'<br>'+\
'State' +states['code']

data = [dict(type='choropleth',autocolorscale=False,locations=states['code'],z=states['dollars'],
       locationmode='USA-states',text=states['text'],colorscale='Earth',colorbar=dict(title='thousand dollars'))]

layout = dict(title='State Spending on Public Education, in $k/student',
              geo=dict(scope='usa',projection=dict(type='albers usa'),
                       showlakes=True,lakecolor='rgb(66,165,245)'))

fig = dict(data=data,layout=layout)
py.iplot(fig,filename='d3-choropleth-map')


#=============================================================================


# point maps

address = '/Users/clu128/Desktop/Exercise Files/Ch09/09_04/snow_inventory.csv'
snow = pd.read_csv(address)
snow.columns = ['stn_id','lat','long','elev','code']

snow_sample = snow.sample(n=200,random_state=25,axis=0)
snow_sample.head()

data = [dict(type='scattergeo',locationmode='USA-states',lon=snow_sample['long'],
             lat=snow_sample['lat'],marker=dict(size=12,autocolorscale=False,
                            colorscale='custom-colorscale',color=snow_sample['elev'],
                            colorbar=dict(title='Elevation(m)')))]

layout = dict(title='NOAA Weather Snowfall Station Elevation',colorbar=True,
              geo=dict(scope='usa',projection=dict(type='albers usa'),showland=True,
                       landcolor='rgb(250,250,250)',subunitcolor='rgb(217,217,217)',
                       countrycolor='rgb(217,217,217)',countrywidth=0.5,subunitwidth=0.5))

fig = dict(data=data,layout=layout)
py.iplot(fig,validate=False,filename='d3-elevation')










