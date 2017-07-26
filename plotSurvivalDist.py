
# coding: utf-8

# In[1]:


import numpy as np


# In[24]:


import matplotlib.pyplot as plt


# In[3]:


exit_times = np.load('exit_times.npy')


# In[4]:


exit_times[:10]


# In[6]:


longest = np.amax(exit_times)


# In[7]:


longest


# In[16]:


fptdist = np.histogram(exit_times, bins=np.arange(longest+1))[0] / 1e5


# In[17]:


fptdist


# In[19]:


survival_prob = 1 - np.cumsum(fptdist)


# In[25]:


plt.plot(np.arange(longest),survival_prob)


# In[26]:


plt.show()


# In[ ]:




