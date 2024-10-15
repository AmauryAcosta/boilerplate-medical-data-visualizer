import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
#df = pd.read_csv('medical_examination.csv')
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2) >25

# 3
df['overweight'] = df['overweight'].astype(int) #1 or 0

df['cholesterol'] = np.where(df['cholesterol'] == 1, 0, 1)
df['gluc'] = np.where(df['gluc'] == 1, 0, 1)


# 4
def draw_cat_plot():
    # 5 DataFrame
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6    
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7
    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio', kind='bar', data=df_cat, height=5, aspect=1)
    
    # 8 - 9
    fig.savefig('catplot.png')
    plt.show()
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) & (df['height'] >= df['height'].quantile(0.025)) & (df['height'] <= df['height'].quantile(0.975)) & (df['weight'] >= df['weight'].quantile(0.025)) & (df['weight'] <= df['weight'].quantile(0.975))]

    # 12
    corr = df_heat.corr().round(1)
    print('corr', corr)

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', linewidths=.5, cmap='coolwarm', ax=ax)

    # 15 - 16
    fig.savefig('heatmap.png')
    plt.show()
    return fig
