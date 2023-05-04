###################
# Import standard modules
import sys
import re
import importlib
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

###################
# Import custom module
sys.path.append('C:/Users/jp/PycharmProjects/ukr')
import ssStats
importlib.reload(ssStats)

###################
# Configure system logging
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

###################
# Import the cleaned data file as dataframe
# NOTE: The following Excel column headers were manually corrected
#  PDHAPTSD_5 renamed to PTSD5
#  GUILD2 renamed to GUILT2
#  CopeBlame2 renamed to CopeSelfBlame2
#  CopeInsSupp1 renamed to CopeInstSupp1
#  CopeSusbtance2 renamed to CopeSubstance2
#  HDSDul3 renamed to HDSDil3
#  Radapt4 renamed to RAdapt4
#  HPIINt2 renamed to HPIInt2
dname = 'C:/Users/jp/Documents/Development/2023-UKR'
fname = 'Clean - UKR T2'
df = pd.read_excel(f'{dname}/{fname}.xlsx')

###################
# Perform any data corrections for issues not already fixed in the imported file
pass

###################
# Add Reverse Coded measures where needed
# ... which will convert df[{scale}R] to df[{scale}]
reverse5 = ['RAdapt3R', 'RAdapt5R',
            'ROpt1R', 'ROpt2R', 'ROpt4R', 'ROpt5R',
            'RRegul1R', 'RRegul4R', 'RRegul5R',
            'RSelfE3R',
            'RSocial2R', 'RSocial3R', 'RSocial5R']

reverse7 = ['HPIAdj2R', 'HPIAdj3R',
            'HPILik5R',
            'HPISoc1R']

for r in reverse5:
    df[r[:-1]] = 6 - df[r]

for r in reverse7:
    df[r[:-1]] = 8 - df[r]

###################
# Get the list of column headers from the dataframe
# ... store the header names as 'headers'
headers = list(df.columns.values)

###################
# Define the scale names
scale_list = ['CombatTrauma', 'Depress', 'PTSD', 'EmExh',
              'RAdapt', 'RRegul', 'ROpt', 'RSelfE', 'RSocial',
              'HPIAdj', 'HPIAmb', 'HPISoc', 'HPILik', 'HPIPru', 'HPIInt', 'HPISch',
              'HDSExc', 'HDSSke', 'HDSCau', 'HDSRes', 'HDSLei',
              'HDSBol', 'HDSMis', 'HDSCol', 'HDSIma', 'HDSDil', 'HDSDut',
              'LO', 'RO', 'OO',
              'CopeDistract', 'CopeActive', 'CopeDenial', 'CopeSubstance',
              'CopeEmotSupp', 'CopeDisengage', 'CopeVent', 'CopeInstSupp', 'CopeReframe',
              'CopeSelfBlame', 'CopePlan', 'CopeHumor', 'CopeAccept', 'CopeReligion',
              'CESattitude_', 'CESbehavior_',
              'FEAR', 'JOVIAL', 'SELFA', 'ATTENT', 'GUILT', 'SAD', 'HOSTILE']

###################
# Using the list of headers, associate which items belong to each of the scales
# ... logging.info the scales and items
logging.info(f'##### List of scales and all items:\n')

scales = {}
for scale in scale_list:
    scales[scale] = sorted([_ for _ in headers if re.search(f'^{scale}[0-9]+$', _)])
    logging.info(f'{scale} = {scales[scale]}')

logging.info('\n')

###################
# For each df[{scale}], select sub-scale of items which provide the best chronbach-alpha for that scale
# ... print the selected sub-scale and store as a_items[{scale}]
# ... logging.info the resulting alpha and store as a_alpha[{scale}]
# ... logging.debug output all the potential sub-scales which were checked
a_items = {}
a_alpha = {}

for scale in scale_list:
    max_drops = math.ceil(len(scales[scale]) / 2) - 1
    drop_penalty = 0.02
    alphas = ssStats.alpha_drops(df[scales[scale]], max_drops=max_drops)
    best_alpha = ssStats.alpha_best(alphas, min_best=0.7, drop_penalty=drop_penalty)

    (flag, alpha, icount, included, dcount, dropped) = best_alpha
    a_alpha[scale] = alpha
    a_items[scale] = included

    logging.info(f'##### {scale} {icount}/{len(scales[scale])} = {alpha} '
                 f'(max_drops = {max_drops}, drop_penalty = {drop_penalty:.1%})')
    logging.info(f'##### {flag:4} {a_alpha[scale]}\n')

    for (flag, alpha, icount, included, dcount, dropped) in alphas:
        logging.debug(f'{flag:4} {alpha: 1.5f}, included {icount} = {included}, dropped {dcount} = {dropped}')

    logging.info('\n')

###################
# Group related scales together
# Calculate the anovas (Analysis of Variance) of each group:scale against change-in-location (UKRlocat)
# ... Store the group,scale,mean,std,count as anovas['UKRlocat']
# ... logging.info each group,scale,mean,std,count against each UKRlocat
logging.debug('# UKRlocat KEY:')
logging.debug('#  1 = Never in UKR')
logging.debug('#  2 = Out of UKR into Non-Danger')
logging.debug('#  3 = Out of UKR into Danger')
logging.debug('#  4 = Left Non-Danger UKR and out of UKR')
logging.debug('#  5 = Left Danger UKR and out of UKR')
logging.debug('#  6 = Moved from Danger to Non-Danger')
logging.debug('#  7 = Moved from Danger to Other Danger')
logging.debug('#  8 = Moved from Non-Danger to Danger')
logging.debug('#  9 = Moved from Non-Danger to Non-Danger')
logging.debug('#\n')

# Map each scale to a scale grouping
scale_groups = {
    'Mental Health': ['Depress', 'EmExh', 'PTSD'],
    'Resilience': ['RRegul', 'ROpt', 'RSocial', 'RAdapt', 'RSelfE'],
    'Orientations': ['LO', 'RO', 'OO'],
    'Others': ['ent_n', 'CombatTrauma']
}

# Initialize the anova DataFrame
anovas = pd.DataFrame(np.nan, index=[], columns=['group', 'scale', 'variable', 'value', 'mean', 'std', 'freq'])

# Define what variables will have anovas calculated against them
anova_names = ['UKRlocat']

# Loop through each variable which will have anovas calculated
for anova_name in anova_names:
    anova_group = df[anova_name]
    anova_group_values = anova_group[np.isfinite(anova_group)].unique()

    # Step through each group:scale calculating the desired anovas
    for scale_group in scale_groups:
        for scale in scale_groups[scale_group]:
            z_scale = f'z_{scale}'

            for anova_group_value in anova_group_values:
                arr = {
                    'group': [scale_group],
                    'scale': [scale],
                    'variable': [anova_name],
                    'value': [anova_group_value],
                    'mean': [df[anova_group == anova_group_value][z_scale].mean()],
                    'std': [df[anova_group == anova_group_value][z_scale].std()],
                    'freq': [df[anova_group == anova_group_value][z_scale].count()]
                }
                arr = pd.DataFrame.from_dict(arr)

                anovas = pd.concat([anovas, arr], ignore_index=True)

# logging.info each of the anova tables
for anova_name in anova_names:
    for scale_group in scale_groups:
        for scale in scale_groups[scale_group]:
            anova_table = anovas[(anovas["variable"] == anova_name) &
                                 (anovas["group"] == scale_group) &
                                 (anovas["scale"] == scale)]

            logging.info(f'Summary of {anova_name}: {scale_group}: {scale}')
            logging.info(f'\n{anova_table}\n')
