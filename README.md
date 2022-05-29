EMPom-NLP
=========

EMPom-NLP is a Python module for *preprocessing*, *clustering*, *topic modelling* and *sentiment analysis* of raw message data, extracted from EMPomoschnik chat bot.

Features
--------

-   Data transformation from raw .xlsx file with chat history
-   Data preprocessing:
    -   message separation, redundant data removal
    -   order/shop/incident codes detection and preservation
    -   stemming/lemmatization
-   Vectorization (**TF-IDF** or One-Hot)
-   Clustering using **KMeans**
-   Topic definition using **Latent Dirichlet Allocation** (and **TF-IDF** top features, if chosen as vectorizer)
-   Interactive cluster and topic distribution visualization using **TSNE** embedding (tweaked pyLDAvis)
-   Sentiment analysis (in order to conduct sentiment analysis, install pretrained labse model from [here](https://drive.google.com/file/d/1MFzblrfQ7kQhrsnbu6FJe9EliT0OVuw4/view))
-   Saving results in an easily interpretable format for further actions (.xslx)

Demonstration
-------------
To see the module *in action*, please investigate [the following notebook](https://github.com/Mc-Seem/Empom-NLP/blob/develop/Clusterizing.ipynb).

File structure
--------------
For better usability, the module is organized in several files.
```
EMPom-NLP
│   README.md                       # This introductory text file
│   demo.ipynb                      # Notebook for workflow demonstration
│
└───classes
│   │   Preprocessing.py            # Data extraction and preprocessing classes
│   │   UniVectorizer.py            # Universal vectorizer class
│   
└───auxiliary
    │   Visualization.py            # Visualization
    │   Sentiment.py                # Sentiment analysis tool
    │   Insight.py                  # Functions for better result interpretation
    └───kmeans_to_pyLDAvis          # Module to simplify clustering visualization
        | ...
```

Credits
-------

Code written by Anna Pastukhova and Maxim Plotnikov.
Curated and supervised by Egor Terikov, Vitaliy Makarenko and Vladislav Smirnov.

