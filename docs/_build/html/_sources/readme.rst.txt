RecAll uses `PyTorch <http://pytorch.org/>`_ to build both Collaborative Filtering (Metadata Embeddings, MLP, NeuCF, EASE) and Sequence Models (LSTM) Recommender Systems, trained with negative sampling.

See the full `documentation <https://www.google.com>`_ for details.

Installation
~~~~~~~~~~~~

.. code-block:: shell-session 

   $ pip install RecAll


Usage
~~~~~

Collaborative Filtering models
===============================

To fit an explicit feedback model on the MovieLens dataset:

.. code-block:: python

    from recall import recall
    from recall.dataset import *

    # put my code here


Dataset
========

RecAll offers utilities to get standard datasets for benchmarking (MovieLens) and to fit your custom data:

.. code-block:: python

    from recall.datasets import *

    # put my code here
