###################
# Import standard modules
import sys
import re
import importlib
import math
import json

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import spiderplot
# %matplotlib inline

###################
# Import custom module
sys.path.append('C:/Users/jp/PycharmProjects/ukr')
import ssStats
importlib.reload(ssStats)

###################
# Configure system logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('matplotlib').setLevel(logging.WARNING)

###################
# Get today's date for file names
from datetime import datetime
today = datetime.today().strftime('%y%m%d')

logging.info(f"Today's Date: {today}")

###################
# Import the cleaned data file as dataframe
dname = 'C:/Users/jp/Documents/Development/2023-UKR'
fname = 'Clean - UKR T2'

logging.info(f'Import data file "{fname}.xlsx"...\n')

# NOTE: The following Excel column headers were manually corrected
#  PDHAPTSD_5 renamed to PTSD5
#  GUILD2 renamed to GUILT2
#  CopeBlame2 renamed to CopeSelfBlame2
#  CopeInsSupp1 renamed to CopeInstSupp1
#  CopeSusbtance2 renamed to CopeSubstance2
#  HDSDul3 renamed to HDSDil3
#  Radapt4 renamed to RAdapt4
#  HPIINt2 renamed to HPIInt2

df = pd.read_excel(f'{dname}/{fname}.xlsx')

###################
# Perform any data corrections for issues not already fixed in the imported file
logging.info(f'Perform data corrections...\n')

logging.info(f'Create BusRebuild_y from BusRebuildYN clearing any "maybe" (4) responses...\n')
df['BusRebuild_y'] = df['BusRebuildYN'].replace(4, math.nan)

logging.info(f'Convert UKRlocat from float to int...\n')
df['UKRlocat'] = df['UKRlocat'].astype(pd.Int64Dtype())

###################
# Add Reverse Coded measures where needed
# ... which will convert df[{scale}R] to df[{scale}]
logging.info(f'Reverse Code measures where needed...\n')

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
# Define the scale and independent variable names
logging.info(f'Build the list of scales and items...')

# Get the list of column headers from the dataframe
headers = list(df.columns.values)

# List of scales in the data
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

# List of independent variables in the data
iv_list = ['Age', 'Gender', 'Family', 'Children',
           'BusinessClosed', 'BusRebuildYN', 'BusRebuildLoc', 'Newbusiness',
           'Movement', 'ComeBack', 'BE',
           'ent_n', 'currlocat', 'UKRcurr', 'danger722', 'danger922', 'danger1122', 'feblocat', 'UKRlocat',
           'BusRebuild_y']

# Using the list of headers, associate which items belong to each of the scales
logging.debug(f'##### List of scales and all items:\n')

scales = {}
for scale in scale_list:
    # Check for any scales which we want to manually override
    if scale == 'CombatTrauma':
        logging.debug(f"Specify {scale} scale to use only item ['{scale}1']")
        scales[scale] = ['CombatTrauma1']

    # No manual override, so include all items in this scale
    else:
        scales[scale] = sorted([_ for _ in headers if re.search(f'^{scale}[0-9]+$', _)])

    logging.debug(f'{scale} = {scales[scale]}')
logging.debug('\n')

logging.debug(f'Independent Variables = {iv_list}')
logging.debug('\n')

###################
# For each df[{scale}], select sub-scale of items which provide the best chronbach-alpha for that scale
# ... print the selected sub-scale and store as df_alphas[{scale}, {alpha}, {items}, {all-items}]
# ... logging.info the resulting alphas
# ... logging.debug output all the potential sub-scales which were checked
logging.info(f"Calculate Cronbach's Alphas for each scale, and determine best sub-scale as needed...\n")

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

    logging.debug(f'##### {scale} {icount}/{len(scales[scale])} = {alpha} '
                  f'(max_drops = {max_drops}, drop_penalty = {drop_penalty:.1%})')
    logging.debug(f'##### {included}')

    for (flag, alpha, icount, included, dcount, dropped) in alphas:
        logging.debug(f'{flag:4} {alpha: 1.5f}, included {icount} = {included}, dropped {dcount} = {dropped}')

    logging.debug('\n')

# Save the selected alphas and sub-scales as {today}-{alphas}.csv
df_alphas.to_csv(f'{dname}/{today}-alphas.csv')

logging.info(f'... selected alphas saved as "{today}-alphas.csv"\n')

###################
# Calculate the per-row mean for each sub-scale
# ... store the means as df[m_{scale}]
# ... store the z-scores as df[z_{scale}]
# ... add m_{scale} to the iv_list
# ... add m_{scale} to the m_scale_list
logging.info(f'Calculate the mean (m_) for each scale...\n')

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
logging.info(f'Calculate z-score (z_m_) standardized mean for each independent variable and scale...\n')
for iv in iv_list:
    df[f'z_{iv}'] = ssStats.zscore(df[iv])

###################
# Define the list of scales in each scale group
logging.info(f'Define the list of scales in each scale group...\n')

scale_groups = {
    'Mental Health': ['m_Depress', 'm_EmExh', 'm_PTSD'],
    'Resilience': ['m_RRegul', 'm_ROpt', 'm_RSocial', 'm_RAdapt', 'm_RSelfE'],
    'Orientations': ['m_LO', 'm_RO', 'm_OO'],
    'Others': ['m_CombatTrauma', 'BusRebuild_y', 'ent_n']
}

logging.debug(f'\n{json.dumps(scale_groups, ensure_ascii=False, indent=4)}')

# Define the list of scales (regardless of scale group)
scale_groups_items = []

for scale_group in scale_groups:
    for scale in scale_groups[scale_group]:
        scale_groups_items += [scale]

###################
# Initialize the anova data structures
logging.info(f'Initialize the anovas data structures...\n')

# Initialize the anova DataFrame
df_anovas = pd.DataFrame(np.nan,
                         index=[],
                         columns=['group', 'scale', 'item', 'value', 'mean', 'z_mean', 'std', 'freq'])

# Define what items will have anovas calculated against them
anova_names = ['UKRlocat']

# Calculate the anovas (Analysis of Variance) of each group:scale against the selected items (ie: UKRlocat)
# ... Store the group,scale,mean,std,count as df_anovas['{anova_name}']
# ... logging.info each group,scale,mean,std,count against each anova_name

# Loop through each item which will have anovas calculated
for anova_name in anova_names:
    logging.info(f'Calculate anovas against item {anova_name}...\n')

    anova_group = df[anova_name]
    anova_group_values = anova_group[np.isfinite(anova_group)].unique()

    # Step through each group:scale calculating the desired anovas
    for scale_group in scale_groups:
        for scale in scale_groups[scale_group]:
            for anova_group_value in anova_group_values:
                arr = {
                    'group': [scale_group],
                    'scale': [scale],
                    'item': [anova_name],
                    'value': [anova_group_value],
                    'mean': [df[anova_group == anova_group_value][scale].mean()],
                    'z_mean': [df[anova_group == anova_group_value][f'z_{scale}'].mean()],
                    'std': [df[anova_group == anova_group_value][scale].std()],
                    'freq': [df[anova_group == anova_group_value][scale].count()]
                }
                arr = pd.DataFrame.from_dict(arr)

                df_anovas = pd.concat([df_anovas, arr], ignore_index=True)

    df_anovas['value'] = df_anovas['value'].astype(int)
    df_anovas['freq'] = df_anovas['freq'].astype(int)

# logging.info each of the anova tables
# ... save the anovas as {today}-{anova_name}.csv
logging.info(f'Save the anova tables...\n')

for anova_name in anova_names:
    for scale_group in scale_groups:
        for scale in scale_groups[scale_group]:
            anova_table = df_anovas[(df_anovas["item"] == anova_name) &
                                    (df_anovas["group"] == scale_group) &
                                    (df_anovas["scale"] == scale)]

            logging.debug(f'Summary of {anova_name}: {scale_group}: {scale}')
            logging.debug(f'\n{anova_table.to_markdown(index=False)}\n')

    df_anovas[df_anovas['item'] == anova_name].to_csv(f'{dname}/{today}-anova-{anova_name}.csv')

    logging.info(f'... {anova_name} anovas saved as "{today}-anova-{anova_name}.csv"\n')

# Define the Location key
# ... logging.info the key information
# ... save the key information as {today}-ukrlocat_key.txt

ukrlocat_key = {
    1: 'Never in UKR',
    2: 'Out of UKR into Non-Danger',
    3: 'Out of UKR into Danger',
    4: 'Left Non-Danger UKR and out of UKR',
    5: 'Left Danger UKR and out of UKR',
    6: 'Moved from Danger to Non-Danger',
    7: 'Moved from Danger to Other Danger',
    8: 'Moved from Non-Danger to Danger',
    9: 'Moved from Non-Danger to Non-Danger\n'
}

logging.info(f'# UKRlocat KEY:\n{json.dumps(ukrlocat_key, indent=4)}')

with open(f'{dname}/{today}-ukrlocat_key.txt', 'w', encoding='utf-8') as f:
    json.dump(ukrlocat_key, f, ensure_ascii=False, indent=4)

logging.info(f'... UKRlocat key saved as "{today}-ukrlocat_key.txt"\n')

# Line Graph of the anovas for the different scale groups
# ... save the plots as {today}-{anova_name}-scales-linegraph.png
logging.info(f'Line Graph the anovas for the different scale groups...\n')

for anova_name in anova_names:
    for mean in ['mean', 'z_mean']:
        fig, ax = plt.subplots(2, 2, figsize=(10, 13))

        for _, scale_group in enumerate(scale_groups):
            i = int(_ / 2)
            j = int(_ % 2)

            data = df_anovas[(df_anovas['item'] == anova_name) & (df_anovas['group'] == scale_group)]
            sns.pointplot(x='value', y=mean, data=data, hue='scale', ax=ax[i, j])
            ax[i, j].set_title(f'{anova_name}: {scale_group}: {mean}')
            ax[i, j].set_xlabel('')
            ax[i, j].set_ylabel('')
            ax[i, j].legend(loc='lower right')
            ax[i, j].tick_params(top=True, labeltop=True)

            if mean == 'mean':
                ymin = 1 - 1 * (_ == 3)
                ymax = 5 + 2 * (_ == 2) - 3 * (_ == 3)
                ax[i, j].set_ylim(ymin, ymax)
            else:
                ax[i, j].set_ylim(-0.5, 0.5)
                ax[i, j].axhline(0, color='black')

        plt.show()
        fig.savefig(f'{dname}/{today}-{anova_name}-{mean}-linegraph.png', dpi=200)

        logging.info(f'... Line Graphs saved as "{today}-{anova_name}-{mean}-linegraph.png"\n')

# Spider Plots of the anovas for the different scale groups
# ... save the plots as {today}-{anova_name}-{scale_group}-{mean}-spiderplot.png
logging.info(f'Spider Plot the anovas for the different scale groups...\n')

for anova_name in anova_names:
    for mean in ['mean', 'z_mean']:
        for i, scale_group in enumerate(scale_groups):
            # Skip the "Others" category for the spider plots
            if i == 3:
                continue

            data = df_anovas[(df_anovas['item'] == anova_name) &
                             ((df_anovas['group'] == scale_group) | (df_anovas['scale'] == 'm_CombatTrauma'))]
            ax = spiderplot.spiderplot(x='value', y=mean, data=data, hue='scale')
            ax.set_title(f'{anova_name}: {scale_group}: {mean}')

            if mean == 'mean':
                ymin = 0
                ymax = 5 + 2 * (i == 2)
                ax.set_rlim(ymin, ymax)
            else:
                ax.set_rlim(-0.5, 0.5)

            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

            plt.show()
            fig = ax.get_figure()
            fig.savefig(f'{dname}/{today}-{anova_name}-{scale_group}-{mean}-spiderplot.png', dpi=200)

            logging.info(f'... Spider Plot saved as "{today}-{anova_name}-{scale_group}-{mean}-spiderplot.png"\n')

# Plot the scale means for each location
# ... save the plots as {today}-means-{anova_name}-barplot.png
logging.info(f'Bar Plot the scale means for each location...\n')

for anova_name in anova_names:
    for mean in ['mean', 'z_mean']:
        fig, ax = plt.subplots(3, 3, figsize=(20, 20))

        for locat in df_anovas[(df_anovas['item'] == anova_name)]['value'].unique():
            i = int((locat - 1) / 3)
            j = int((locat - 1) % 3)

            data = df_anovas[(df_anovas['item'] == anova_name) & (df_anovas['value'] == locat)]
            sns.barplot(x='scale', y=mean, data=data, ax=ax[i, j], palette='tab10')
            ax[i, j].set_title(f'Scale Means for {anova_name} == {locat}: {ukrlocat_key[locat]}')
            ax[i, j].set_xlabel('')
            ax[i, j].set_ylabel('')
            ax[i, j].set_xticklabels(ax[i, j].get_xticklabels(), rotation=90)

            if mean == 'mean':
                ax[i, j].set_ylim(0, 5.5)
            else:
                ax[i, j].set_ylim(-0.5, 0.5)

        plt.tight_layout()

        plt.show()
        fig.savefig(f'{dname}/{today}-scales-{anova_name}-{mean}-barplot.png', dpi=200)

        logging.info(f'... Bar Plots saved as "{today}-scales-{anova_name}-{mean}-barplot.png"\n')

# Spider Plot the scale means for each location
# ... save the plots as {today}-means-{anova_name}-spiderplot.png
logging.info(f'Spider Plot the scale means for each location...\n')

for anova_name in anova_names:
    for mean in ['mean', 'z_mean']:
        for locat in df_anovas[(df_anovas['item'] == anova_name)]['value'].unique():
            data = df_anovas[(df_anovas['item'] == anova_name) & (df_anovas['value'] == locat)]
            ax = spiderplot.spiderplot(x='scale', y=mean, data=data)
            ax.set_title(f'Scale {mean}s for {anova_name} {locat}: {ukrlocat_key[locat]}')

            if mean == 'mean':
                ax.set_rlim(0, 5.5)
            else:
                ax.set_rlim(-0.5, 0.5)

            plt.show()
            fig = ax.get_figure()
            fig.savefig(f'{dname}/{today}-{mean}-{anova_name}-{locat}-spiderplot.png', dpi=200)

            logging.info(f'... Spider Plot saved as "{today}-{mean}-{anova_name}-{locat}-spiderplot.png"\n')

###################
# Calculate correlations for scale means and selected independent variables
# ... save the correlation matrix as {today}-correlations-all.csv and {today}-correlations-ent_y.csv
logging.info(f'Calculate correlations for scale means and selected independent variables...\n')

logging.debug(f'r < 0.25         No relationship')
logging.debug(f'0.25 <= r < 0.5  Weak relationship')
logging.debug(f'0.5 <= r < 0.75  Moderate relationship')
logging.debug(f'r => 0.75        Strong relationship')

df[scale_groups_items].corr()
df[scale_groups_items].corr().to_csv(f'{dname}/{today}-correlations-all.csv')
df[(df['ent_n'] == 1)][scale_groups_items].corr().to_csv(f'{dname}/{today}-correlations-ent_y.csv')
df[(df['ent_n'] == 0)][scale_groups_items].corr().to_csv(f'{dname}/{today}-correlations-ent_n.csv')

logging.info(f'... correlation table saved as "{today}-correlations-all.csv"\n')
logging.info(f'... correlation table saved as "{today}-correlations-ent_y.csv"\n')
logging.info(f'... correlation table saved as "{today}-correlations-ent_n.csv"\n')

###################
# Plot heatmaps for the correlations
# ... save the correlation heatmaps as {today}-correlations-heatmap.png
logging.info(f'Plot heatmaps for the correlations...\n')

fig, ax = plt.subplots(5, 1, figsize=(10, 30))

# Plot the heatmap of correlations for all users
corr = df[scale_groups_items].corr()
sns.heatmap(abs(corr), cmap='YlGnBu', annot=corr, fmt=".2f", ax=ax[0])
ax[0].set_title(f'Correlations for all users')

# Plot the heatmap of correlations for entrepreneurs
corr = df[(df['ent_n'] == 1)][scale_groups_items].corr()
sns.heatmap(abs(corr), cmap='YlGnBu', annot=corr, fmt=".2f", ax=ax[1])
ax[1].set_title(f'Correlations specifically for entrepreneurs')

# Plot the heatmap of correlations for non-entrepreneurs
corr = df[(df['ent_n'] == 0)][scale_groups_items].corr()
sns.heatmap(abs(corr), cmap='YlGnBu', annot=corr, fmt=".2f", ax=ax[2])
ax[2].set_title(f'Correlations specifically for non-entrepreneurs')

# Plot the heatmap of correlations for entrepreneurs who plan to rebuild
corr = df[(df['ent_n'] == 1) & (df['BusRebuildYN'] == 1)][scale_groups_items].corr()
sns.heatmap(abs(corr), cmap='YlGnBu', annot=corr, fmt=".2f", ax=ax[3])
ax[3].set_title(f'Correlations specifically for entrepreneurs who plan to rebuild')

# Plot the heatmap of correlations for entrepreneurs who do not plan to rebuild
corr = df[(df['ent_n'] == 1) & (df['BusRebuildYN'] == 0)][scale_groups_items].corr()
sns.heatmap(abs(corr), cmap='YlGnBu', annot=corr, fmt=".2f", ax=ax[4])
ax[4].set_title(f'Correlations specifically for entrepreneurs who do not plan to rebuild')

plt.tight_layout()

plt.show()
fig.savefig(f'{dname}/{today}-correlations-heatmap.png', dpi=200)

logging.info(f'... correlation heatmaps saved as "{today}-correlations-heatmap.png"\n')

###################
# Plot scale means against ent_n (0=no entrepreneur, (1=yes entrepreneur)
# ... save the bar plot matrix as {today}-ent_yn-m_scales-barplot.csv
logging.info(f'Bar Plot scale means against ent_n (0=no entrepreneur, 1=yes entrepreneur)...\n')

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
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=90)

# Plot the normalized scale means against ent_n
sns.barplot(x='scale', y='z_mean', hue='ent_n', data=df_ent_n, ax=ax[1])
ax[1].set_title(f'ent_n (0=no entrepreneur, 1=yes entrepreneur) vs normalized scale means')
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=90)

plt.tight_layout()

plt.show()
fig.savefig(f'{dname}/{today}-ent_n-m_scales-barchart.png', dpi=200)

logging.info(f'... ent_yn bar charts saved as "{today}-ent_n-m_scales-barchart.png"\n')

###################
# Save the updated dataframe (now with the calculated m_ and z_ scales)
logging.info(f'Save the updated dataframe (now with the calculated m_ and z_ scales) as:\n'
             f'"{today}-{fname}-with_scales.csv"...\n')

df.to_csv(f'{dname}/{today}-{fname}-with_scales.csv')

logging.info('\nDone.')
