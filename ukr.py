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
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('matplotlib').setLevel(logging.WARNING)

###################
# Get today's date for file names
from datetime import datetime
today = datetime.today().strftime('%y%m%d')

logging.info(f'Today: {today}')

###################
# Import the cleaned data file as dataframe
logging.debug(f'Importing data file...\n')

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
logging.debug(f'Perform data corrections...\n')

logging.debug(f'Create BuRebuild_y from BusRebuildYN...\n')
df['BusRebuild_y'] = df['BusRebuildYN'].replace(4, 0)

###################
# Add Reverse Coded measures where needed
# ... which will convert df[{scale}R] to df[{scale}]
logging.debug(f'Reverse Coding measures where needed...\n')

reverse5 = ['RAdapt3R', 'RAdapt5R',
            'ROpt1R', 'ROpt2R', 'ROpt4R', 'ROpt5R',
            'RRegul1R', 'RRegul4R', 'RRegul5R',
            'RSelfE3R',
            'RSocial2R', 'RSocial3R', 'RSocial5R']

reverse7 = ['HPIAdj2R', 'HPIAdj3R',
            'HPILik5R',
            'HPISoc1R']

for r in reverse5:
    logging.debug(f'  {r} ==> {r[:-1]}')
    df[r[:-1]] = 6 - df[r]

for r in reverse7:
    logging.debug(f'  {r} ==> {r[:-1]}')
    df[r[:-1]] = 8 - df[r]

###################
# Get the list of column headers from the dataframe
# ... store the header names as 'headers'
headers = list(df.columns.values)

###################
# Define the scale and independent variable names
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

iv_list = ['Age', 'Gender', 'Family', 'Children',
           'BusinessClosed', 'BusRebuildYN', 'BusRebuildLoc', 'Newbusiness',
           'Movement', 'ComeBack', 'BE',
           'ent_n', 'currlocat', 'UKRcurr', 'danger722', 'danger922', 'danger1122', 'feblocat', 'UKRlocat']

###################
# Using the list of headers, associate which items belong to each of the scales
# ... logging.info the scales and items
logging.info(f'##### List of scales and all items:\n')

scales = {}
for scale in scale_list:
    scales[scale] = sorted([_ for _ in headers if re.search(f'^{scale}[0-9]+$', _)])
    logging.info(f'{scale} = {scales[scale]}')
logging.info('\n')

logging.info(f'Independent Variables = {iv_list}')
logging.info('\n')

###################
# For each df[{scale}], select sub-scale of items which provide the best chronbach-alpha for that scale
# ... print the selected sub-scale and store as df_alphas[{scale}, {alpha}, {items}, {all-items}]
# ... logging.info the resulting alphas
# ... logging.debug output all the potential sub-scales which were checked
logging.debug(f'Calculating alphas for each scale, and determining best sub-scale as needed...\n')

df_alphas = pd.DataFrame(np.nan, index=[], columns=['scale', 'alpha', 'items', 'all-items'])

for scale in scale_list:
    max_drops = math.ceil(len(scales[scale]) / 2) - 1  # Not allowed to drop half or more
    # max_drops = math.floor(len(scales[scale]) / 2)     # Not allowed to drop more than half (but exactly half is ok)
    drop_penalty = 0.02
    alphas = ssStats.alpha_drops(df[scales[scale]], max_drops=max_drops)
    best_alpha = ssStats.alpha_best(alphas, min_best=0.7, drop_penalty=drop_penalty)

    (flag, alpha, icount, included, dcount, dropped) = best_alpha

    arr = {
        'scale': [scale],
        'alpha': [alpha],
        'items': [included],
        'all-items': [scales[scale]]
    }
    arr = pd.DataFrame.from_dict(arr)

    df_alphas = pd.concat([df_alphas, arr], ignore_index=True)

    logging.info(f'##### {scale} {icount}/{len(scales[scale])} = {alpha} '
                 f'(max_drops = {max_drops}, drop_penalty = {drop_penalty:.1%})')
    logging.info(f'##### {included}')
    logging.info(f'##### {flag:4} {alpha}\n')

    for (flag, alpha, icount, included, dcount, dropped) in alphas:
        logging.debug(f'{flag:4} {alpha: 1.5f}, included {icount} = {included}, dropped {dcount} = {dropped}')

    logging.info('\n')

# Save the selected alphas and sub-scales as {today}-{alphas}.csv
df_alphas.to_csv(f'{dname}/{today}-alphas.csv')

###################
# Calculate the per-row mean for each sub-scale
# ... store the means as df[m_{scale}]
# ... store the z-scores as df[z_{scale}]
# ... add m_{scale} to the iv_list
# ... add m_{scale} to the m_scale_list
logging.debug(f'Calculating mean for each scale...\n')

m_scale_list = []

for scale in scale_list:
    m_scale = f'm_{scale}'

    # Add the scale mean header into the independent variable list
    if m_scale not in iv_list:
        iv_list += [m_scale]

    # Add the scale mean header into the m_scale_list
    if m_scale not in m_scale_list:
        m_scale_list += [m_scale]

    items = df_alphas[df_alphas['scale'] == scale]['items'].item()

    df[m_scale] = df[items].mean(axis=1, skipna=False)

# Calculate the z-score for each independent variable and scale mean
# ... store the z-scores as df[z_{iv}] and df[z_m_{scale}]
logging.debug(f'Calculating z-score for each independent variable and scale mean...\n')
for iv in iv_list:
    df[f'z_{iv}'] = ssStats.zscore(df[iv])

###################
# Initialize the anovas data structures
logging.debug(f'Create the scale groups to be used for generating anovas...\n')

scale_groups = {
    'Mental Health': ['z_m_Depress', 'z_m_EmExh', 'z_m_PTSD'],
    'Resilience': ['z_m_RRegul', 'z_m_ROpt', 'z_m_RSocial', 'z_m_RAdapt', 'z_m_RSelfE'],
    'Orientations': ['z_m_LO', 'z_m_RO', 'z_m_OO'],
    'Others': ['z_ent_n', 'z_m_CombatTrauma']
}

# Initialize the anova DataFrame
df_anovas = pd.DataFrame(np.nan, index=[], columns=['group', 'scale', 'variable', 'value', 'mean', 'std', 'freq'])

# Define what variables will have anovas calculated against them
anova_names = ['UKRlocat']

logging.debug('# UKRlocat KEY:')
logging.debug('#  1 = Never in UKR')
logging.debug('#  2 = Out of UKR into Non-Danger')
logging.debug('#  3 = Out of UKR into Danger')
logging.debug('#  4 = Left Non-Danger UKR and out of UKR')
logging.debug('#  5 = Left Danger UKR and out of UKR')
logging.debug('#  6 = Moved from Danger to Non-Danger')
logging.debug('#  7 = Moved from Danger to Other Danger')
logging.debug('#  8 = Moved from Non-Danger to Danger')
logging.debug('#  9 = Moved from Non-Danger to Non-Danger\n')

# Calculate the anovas (Analysis of Variance) of each group:scale against the selected variables (ie: UKRlocat)
# ... Store the group,scale,mean,std,count as df_anovas['{anova_name}']
# ... logging.info each group,scale,mean,std,count against each anova_name
logging.debug(f'Calculating anovas...\n')

# Loop through each variable which will have anovas calculated
for anova_name in anova_names:
    logging.debug(f'Calculating anovas against variable {anova_name}...\n')

    anova_group = df[anova_name]
    anova_group_values = anova_group[np.isfinite(anova_group)].unique()

    # Step through each group:scale calculating the desired anovas
    for scale_group in scale_groups:
        for scale in scale_groups[scale_group]:
            for anova_group_value in anova_group_values:
                arr = {
                    'group': [scale_group],
                    'scale': [scale],
                    'variable': [anova_name],
                    'value': [anova_group_value],
                    'mean': [df[anova_group == anova_group_value][scale].mean()],
                    'std': [df[anova_group == anova_group_value][scale].std()],
                    'freq': [df[anova_group == anova_group_value][scale].count()]
                }
                arr = pd.DataFrame.from_dict(arr)

                df_anovas = pd.concat([df_anovas, arr], ignore_index=True)

# logging.info each of the anova tables
# ... save the anovas as {today}-{anova_name}.csv
logging.debug(f'Saving the anova tables...\n')

for anova_name in anova_names:
    for scale_group in scale_groups:
        for scale in scale_groups[scale_group]:
            anova_table = df_anovas[(df_anovas["variable"] == anova_name) &
                                    (df_anovas["group"] == scale_group) &
                                    (df_anovas["scale"] == scale)]

            logging.info(f'Summary of {anova_name}: {scale_group}: {scale}')
            logging.info(f'\n{anova_table}\n')

    df_anovas[df_anovas['variable'] == anova_name].to_csv(f'{dname}/{today}-anova-{anova_name}.csv')

# Plot the anovas for the different scale groups
# ... save the plots as {today}-anova-{anova_name}.png
for anova_name in anova_names:
    fig, ax = plt.subplots(1, 4, figsize=(15, 5))

    for i, scale_group in enumerate(scale_groups):
        data = df_anovas[(df_anovas['variable'] == anova_name) & (df_anovas['group'] == scale_group)]
        sns.pointplot(x='value', y='mean', data=data, hue='scale', ax=ax[i])
        ax[i].set_title(f'{anova_name}: {scale_group}')
        ax[i].set_xlabel('')
        ax[i].set_ylabel('')
        ax[i].axhline(0, color='black')
        ax[i].legend(loc='upper right')
        ax[i].tick_params(top=True, labeltop=True)

    plt.show()
    fig.savefig(f'{dname}/{today}-anova-{anova_name}.png', dpi=200)

###################
# Calculate correlations for scale means and selected independent variables
# ... save the correlation matrix as {today}-correlations-all.csv and {today}-correlations-ent_y.csv
# ... save the correlation heatmap as {today}-correlations-all.png and {today}-correlations-ent_y.png
logging.debug(f'Calculating correlations for scale means and selected independent variables...\n')

corr_list = ['m_Depress', 'm_EmExh', 'm_PTSD',
             'm_RRegul', 'm_ROpt', 'm_RSocial', 'm_RAdapt', 'm_RSelfE',
             'm_LO', 'm_RO', 'm_OO',
             'm_CombatTrauma',
             'BusRebuild_y', 'ent_n']

df[corr_list].corr()
df[corr_list].corr().to_csv(f'{dname}/{today}-correlations-all.csv')
df[(df['ent_n'] == 1)][corr_list].corr().to_csv(f'{dname}/{today}-correlations-ent_y.csv')
df[(df['ent_n'] == 0)][corr_list].corr().to_csv(f'{dname}/{today}-correlations-ent_n.csv')

fig, ax = plt.subplots(3, 1, figsize=(10, 15))

# Plot the heatmap of correlations for all users
sns.heatmap(abs(df[corr_list].corr()), cmap='YlGnBu', annot=True, fmt=".2f", ax=ax[0])
ax[0].set_title(f'Correlations for all users')

# Plot the heatmap of correlations for entrepreneurs
sns.heatmap(abs(df[(df['ent_n'] == 1)][corr_list].corr()), cmap='YlGnBu', annot=True, fmt=".2f", ax=ax[1])
ax[1].set_title(f'Correlations specifically for entrepreneurs')

# Plot the heatmap of correlations for non-entrepreneurs
sns.heatmap(abs(df[(df['ent_n'] == 0)][corr_list].corr()), cmap='YlGnBu', annot=True, fmt=".2f", ax=ax[2])
ax[2].set_title(f'Correlations specifically for non-entrepreneurs')

plt.tight_layout()

plt.show()
fig.savefig(f'{dname}/{today}-correlations.png', dpi=200)

###################
# Plot scale means against ent_n (0=no entrepreneur, (1=yes entrepreneur)
# ... save the bar plot matrix as {today}-ent_yn-m_scales.csv
logging.debug(f'Plot scale means against ent_n as (0=no entrepreneur, (1=yes entrepreneur)...\n')

df_ent_n = pd.DataFrame(np.nan, index=[], columns=['scale', 'ent_n', 'mean'])

for scale in m_scale_list:
    for ent_n in [0, 1]:
        arr = {
            'scale': [scale],
            'ent_n': [ent_n],
            'mean': [df[df['ent_n'] == ent_n][f'{scale}'].mean()],
            'z_mean': [df[df['ent_n'] == ent_n][f'z_{scale}'].mean()]
        }
        arr = pd.DataFrame.from_dict(arr)

        df_ent_n = pd.concat([df_ent_n, arr], ignore_index=True)

fig, ax = plt.subplots(2, 1, figsize=(15, 10))

# Plot the scale means against ent_n
sns.barplot(x='scale', y='mean', hue='ent_n', data=df_ent_n, ax=ax[0])
ax[0].set_title(f'ent_n (0=no entrepreneur, 1=yes entrepreneur) vs scale means')
ax[0].set_xticklabels(m_scale_list, rotation=90)

# Plot the normalized scale means against ent_n
sns.barplot(x='scale', y='z_mean', hue='ent_n', data=df_ent_n, ax=ax[1])
ax[1].set_title(f'ent_n (0=no entrepreneur, 1=yes entrepreneur) vs normalized scale means')
ax[1].set_xticklabels(m_scale_list, rotation=90)

plt.tight_layout()

plt.show()
fig.savefig(f'{dname}/{today}-ent_n-m_scales.png', dpi=200)
