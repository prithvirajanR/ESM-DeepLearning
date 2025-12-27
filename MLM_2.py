#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os


# In[2]:


from src.prob_scoring import*
from src.useful_functions import*
from src.data_loader import*
from src.handle_mutations import*
from src.mask_position import*
from src.landscape import*


# In[3]:


read_dirs_paths('dir_paths.txt', globals())


# In[4]:


df = load_dms(data_a4_human)
df.head()


# In[5]:


wt_seq = get_wt_sequence(
    data_prgym_reference, 
    filename1
)


# In[6]:


df_single = get_single_mutants(df)
df_double = get_double_mutants(df)


# In[7]:


df["sequence"] = df["mutant"].apply(lambda m: apply_mutations(wt_seq, m))
df.head()


# In[8]:


df_single["sequence"] = df_single["mutant"].apply(lambda m: apply_mutations(wt_seq, m))
df_single.head()


# In[9]:


df_double["sequence"] = df_double["mutant"].apply(lambda m: apply_mutations(wt_seq, m))
df_double.head()


# In[10]:


model_id = "facebook/esm2_t33_650M_UR50D"
device = "cuda"

tokenizer = SeqTokenizer(model_id, device=device)
model = tokenizer.model
mask_id = tokenizer.mask_id


# In[11]:


# df["PLL"] = batch_pll(df["sequence"].tolist(), tokenizer, model)


# In[ ]:


df_single["PLL"] = batch_pll(df_single["sequence"].tolist(), tokenizer, model)


# In[ ]:


# df_double["PLL"] = batch_pll(df_double["sequence"].tolist(), tokenizer, model)


# In[ ]:


import scipy.stats
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


df_clean = df_single.dropna(subset=['fitness', 'PLL'])
df_clean = df_clean[~np.isinf(df_clean['PLL'])]


# In[ ]:


corr, pval = scipy.stats.spearmanr(df_clean['fitness'], df_clean['PLL'])

print(f"Spearman Correlation: {corr:.4f}")
print(f"P-value: {pval:.4e}")

plt.figure(figsize=(10, 6))

plt.scatter(df_clean['fitness'], df_clean['PLL'], alpha=0.5, s=10, c='blue', label='Mutants')


# In[ ]:


z = np.polyfit(df_clean['fitness'], df_clean['PLL'], 1)
p = np.poly1d(z)
x_range = np.linspace(df_clean['fitness'].min(), df_clean['fitness'].max(), 100)
plt.plot(x_range, p(x_range), "r--", linewidth=2, label=f'Trend (ρ={corr:.3f})')


# In[ ]:


plt.xlabel('Real Lab Fitness Score (DMS)', fontsize=12)
plt.ylabel('ESM-2 Pseudo-Log-Likelihood (PLL)', fontsize=12)
plt.title(f'Fitness vs. PLL Score\nSpearman Correlation = {corr:.3f}', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()


# In[ ]:


plt.tight_layout()
plt.savefig("pll_correlation_plot.png", dpi=300)
plt.show()


# In[ ]:




