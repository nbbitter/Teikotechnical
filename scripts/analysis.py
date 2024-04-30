#!/usr/bin/env python
# coding: utf-8

# In[13]:


data_path="/Users/nicholasbitter/Downloads/cell-count.csv" # replace path with path to cell-count.csv data on machine


# In[42]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
from pathlib import Path 
import seaborn as sns
import statsmodels.formula.api as smf
import pymer4
from pymer4.models import Lmer
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.stats import mannwhitneyu


# ### Python code 
# #### 1.  Please write a python program to convert cell count in cell-count.csv to relative frequency (in percentage) of total cell count for each sample. Total cell count of each sample is the sum of cells in the five populations of that sample. Please return an output file in csv format with cell count and relative frequency of each population of each sample per line. The output file should have the following columns:
# 
# - sample: the sample id as in column sample in cell-count.csv
# 
# - total_count: total cell count of sample
# 
# - population: name of the immune cell population (e.g. b_cell, cd8_t_cell, etc.)
# 
# - count: cell count
# 
# - percentage: relative frequency in percentage

# In[6]:


# reading in the data cell counts
cell_count = pd.read_csv(data_path)
cell_count['total']=cell_count[['b_cell', 'cd8_t_cell', 'cd4_t_cell', 'nk_cell', 'monocyte']].sum(axis=1)
cell_count['id']=[i for i in range(0,len(cell_count))]
cell_count.head(5)


# In[11]:


## creating precise output csv requested
output=pd.melt(cell_count, id_vars=['id' ,'project', 'subject', 'condition', 'age', 'sex', 'treatment',
       'response', 'sample', 'sample_type', 'time_from_treatment_start'], 
        value_vars=['b_cell', 'cd8_t_cell', 'cd4_t_cell', 'nk_cell', 'monocyte'],
        var_name='population', value_name='cell_count')
output['total_sample_count']=output.groupby(['id' ,'project', 'subject', 'condition', 'age', 'sex', 'treatment',
       'response', 'sample', 'sample_type', 'time_from_treatment_start'],dropna=False)['cell_count'].transform('sum')
output['percentage']=np.round((output['cell_count']/output['total_sample_count'])*100)
## writing to csv
output[['sample','population','total_sample_count','cell_count','percentage']].to_csv('CellCountFrequency.csv')
output.tail(20)


# #### 2. Among patients who have treatment tr1, we are interested in comparing the differences in cell population relative frequencies of melanoma patients who respond (responders) to tr1 versus those who do not (non-responders), with the overarching aim of predicting response to treatment tr1. Response information can be found in column response, with value y for responding and value n for non-responding. Please only include PBMC (blood) samples.
# 
# ##### a. For each immune cell population, please generate a boxplot of the population relative frequencies comparing responders versus non-responders.

# In[150]:


plot_data=output[(output['sample_type']=='PBMC') &(output['condition']=='melanoma')& (output['treatment']=='tr1')]

sns.catplot(
    data=plot_data, x='response', y='percentage',
    col='population', kind='box', col_wrap=2
)


# #### b. Which cell populations are significantly different in relative frequencies between responders and non-responders? Please include statistics to support your conclusion.

# - Because their are repeated measures on subjects in this data, these samples are not i.i.d..  https://www.publichealth.columbia.edu/research/population-health-methods/repeated-measures-analysis It introduces the need for statistical methods that can handle within subject correlation.  The two best methods to appropriately handle this data are Generalized Estimator Equations (GEE) and Mixed effect models. Repeated measures ANOVA is not preferred as it requires normally distributed response and balanced data.  I will attempt several methods starting with the best 
# 
# - https://pubmed.ncbi.nlm.nih.gov/20220526/ compares GEE's and Mixed effect modeling approaches and detemined GEE's produce more realistic results in their application. 
# - Since the question concerns the population of cells that are significantly different between responders vs non-responders it makes the GEE or mixed effect models most appropriate. The method to determine significantly different cell populations is as follows:
# 
#     1. All the cell populations are included in the GEE model with the outcome responder as y/n. 
#     2. The interperation of the log odds coeefficeints allows for determining significance. p values of the coefficients less than .05 indicate signifcance in predicting a responder vs non responder hence, the determination of significantly different populations while conidering for correlation between subjects. 
#    
#  - NOTE: It is unlikely given the sample size of 9 that this method will actually work. If it fails to converge or results in perfect discrimination between responders and non-responders then the model interpretation is invalid. This likely will happen with sample size of 9 in the example. 
#    
#    
# - Furthermore this nested effect of projects can be considered too however, I will only try to consider repeated measures on subjects with the GEE because the linear mixed effect models is especially unlikely to converge fgiven the small sample size. The mixed effect model in the formulation "response  ~ b_cell+cd8_t_cell+cd4_t_cell+nk_cell+monocyte + (1|project) + (1|project:subject)" can handle both nested situations of repeated measures on subjects and within projects. The interpretation of the coefficients accounts for reapeated meassures on subjects and within batches.
#    
#  
#  
# - Alternative: The i.i.d assumption can be relaxed by averaging the repeated measures among subjects so each subject has one measurement. This makes common traditional statistics like mann whitney test, or t test statistically valid. However, it is not as good to perform this technique as the GEE or mixed effect model described above. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6072386/ I will also perform the bonferronni correction since we are applying multiple tests here 
# 
# 
# 
# 

# In[154]:


## attempt with mixed effect model 
#model_data=cell_count[(cell_count['sample_type']=='PBMC')& (cell_count['condition']=='melanoma')&(cell_count['treatment']=='tr1')]
#model_data['response']=pd.factorize(model_data['response'])[0]
#model = Lmer("response  ~ b_cell+cd8_t_cell+cd4_t_cell+nk_cell+monocyte + (1|project) + (1|project:subject)",
#             data=model_data, family = 'binomial')
#print(model.fit())

### as expected it did not converge with 9 samples. 


# In[158]:


model_data=cell_count[(cell_count['sample_type']=='PBMC')& (cell_count['condition']=='melanoma')&(cell_count['treatment']=='tr1')]
model = smf.gee("response ~ b_cell+cd8_t_cell+cd4_t_cell+nk_cell+monocyte",'subject',
                model_data,family=sm.families.Binomial())
result = model.fit()
print(result.summary())
## perfect seperation warning 


# In[161]:


## averaging method
average_data=model_data[['subject','response',
             'b_cell', 'cd8_t_cell',
             'cd4_t_cell', 'nk_cell', 'monocyte']].groupby(['subject','response']).mean().reset_index()

def perform_mann_whitney(data):
    results = {}
    # Identify responder and non-responder groups
    group1 = data[data['response'] == 'y']
    group2 = data[data['response'] == 'n']
    
    for col in data.columns.drop(['response','subject']):
        # Perform the Mann-Whitney U test
        stat, p_value = mannwhitneyu(group1[col], group2[col], alternative='two-sided')
        results[col] = {'U statistic': stat, 'P-value': p_value}
    return pd.DataFrame(results).reset_index()


# In[162]:


results=perform_mann_whitney(average_data)
results=results.T.iloc[1:,:]
# as expected with such a small sample size none of these are statisticlly significant with 
results.columns=['U statistic','P-value']
results['p_adjusted']=multipletests(results['P-value'], method='bonferroni')[1]


# In[163]:


results ## none are significant largely because the sample size is so low in this example

