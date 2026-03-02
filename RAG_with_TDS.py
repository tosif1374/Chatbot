#!/usr/bin/env python
# coding: utf-8

# In[5]:


#for one url
from langchain_community.document_loaders import WebBaseLoader
loader =  WebBaseLoader('https://towardsdatascience.com/the-large-language-model-course-b6663cd57ceb/')
#loader.load()


# In[6]:


import feedparser

feed = feedparser.parse("https://towardsdatascience.com/feed")

urls = [entry.link for entry in feed.entries]

print("URLs found:", len(urls))
print(urls[:50])


# In[7]:


import feedparser

feed = feedparser.parse("https://towardsdatascience.com/feed")

urls = [entry.link for entry in feed.entries]

print("URLs found:", len(urls))
print(urls[:50])


# In[8]:


from langchain_community.document_loaders import WebBaseLoader
urls=['https://towardsdatascience.com/distributed-reinforcement-learning-for-scalable-high-performance-policy-optimization/', 'https://towardsdatascience.com/how-to-apply-agentic-coding-to-solve-problem/', 'https://towardsdatascience.com/run-claude-code-for-free-with-local-and-cloud-models-from-ollama/', 'https://towardsdatascience.com/creating-an-etch-a-sketch-app-using-python-turtle/', 'https://towardsdatascience.com/why-your-multi-agent-system-is-failing-escaping-the-17x-error-trap-of-the-bag-of-agents/', 'https://towardsdatascience.com/on-the-possibility-of-small-networks-for-physics-informed-learning/', 'https://towardsdatascience.com/multi-attribute-decision-matrices-done-right/', 'https://towardsdatascience.com/tds-newsletter-january-must-reads-on-data-platforms-infinite-context-and-more/', 'https://towardsdatascience.com/optimizing-vector-search-why-you-should-flatten-structured-data/', 'https://towardsdatascience.com/rope-clearly-explained/', 'https://towardsdatascience.com/the-unbearable-lightness-of-coding/', 'https://towardsdatascience.com/randomization-works-in-experiments-even-without-balance/', 'https://towardsdatascience.com/federated-learning-part-2-implementation-with-the-flower-framework-%f0%9f%8c%bc/', 'https://towardsdatascience.com/machine-learning-in-production-what-this-really-means/', 'https://towardsdatascience.com/i-ditched-my-mouse-how-i-control-my-computer-with-hand-gestures-in-60-lines-of-python/', 'https://towardsdatascience.com/modeling-urban-walking-risk-using-spatial-temporal-machine-learning/', 'https://towardsdatascience.com/going-beyond-the-context-window-recursive-language-models-in-action/', 'https://towardsdatascience.com/data-science-as-engineering/', 'https://towardsdatascience.com/from-connections-to-meaning-why-heterogeneous-graph-transformers-hgt-change-demand-forecasting/', 'https://towardsdatascience.com/layered-architecture-for-building-readable-robust-and-extensible-apps/']
loader= WebBaseLoader(urls)
docs=loader.load()


# In[9]:


from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
final_documents=text_splitter.split_documents(docs)
final_documents


# In[10]:


import re
from langchain_core.documents import Document

def clean_text(text: str) -> str:
    # remove URLs
    text = re.sub(r"http\S+", "", text)

    # remove common UI / navigation junk
    junk_patterns = [
        r"Sign in",
        r"Submit an Article",
        r"Write For TDS",
        r"Toggle.*",
        r"LinkedIn",
        r"\bX\b",
        r"Latest.*Newsletter",
        r"Publish AI, ML & data-science insights.*professionals\.",
    ]

    for p in junk_patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE)

    # clean excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# In[11]:


clean_docs = [
    Document(
        page_content=clean_text(d.page_content),
        metadata=d.metadata
    )
    for d in docs
]


# In[12]:


clean_docs


# In[27]:


from langchain_text_splitters import RecursiveCharacterTextSplitter

# 2. TEXT SPLITTING (MANDATORY)
# -------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,      # safe for nomic
    chunk_overlap=100
)

split_docs = text_splitter.split_documents(clean_docs)
print(f"Total chunks created: {len(split_docs)}")



# In[30]:


split_docs[0].page_content


# In[23]:


from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)


# In[31]:


BATCH_SIZE = 32
vectorstore = None

for i in range(0, len(split_docs), BATCH_SIZE):
    batch = split_docs[i:i + BATCH_SIZE]
    print(f"Embedding batch {i//BATCH_SIZE + 1}")

    if vectorstore is None:
        vectorstore = FAISS.from_documents(batch, embeddings)
    else:
        vectorstore.add_documents(batch)


# In[32]:


vectorstore.save_local("faiss_index")
print("FAISS index saved")


# In[36]:


query = "data science"
docs = vectorstore.similarity_search(query, k=2)
docs


# In[ ]:




