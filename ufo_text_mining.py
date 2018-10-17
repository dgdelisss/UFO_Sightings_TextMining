{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "TextMining.ipynb",
      "version": "0.3.2",
      "provenance": [],
      "collapsed_sections": [
        "iHzlqv2s-FoH"
      ],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dgdelisss/Blue2_HW6_UFO_Text/blob/master/ufo_text_mining.py)"
      ]
    },
    {
      "metadata": {
        "id": "ZP1jVwz1lbtZ",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "**Project Description:**\n",
        "We will analyze data on reported incidents of UFO sightings. Utilizing data collected by an organization dedicated to this topic, we will apply topic clustering techniques to identify commonalities among these sightings and interpret the results to provide a summary of the major themes of these reports. After clustering among the full dataset, we will then focus on comparing UFO sightings in California, Arizona, and Nevada again using clustering to investigate their similarities and differences.  \n",
        "\n",
        "**Analysis: **\n",
        "We will perform topic clustering on the text column from our dataset to identify major topics of discussion. We will then use this clustering to analyze any commonalities or anomalies based on descriptors of UFO shape, size, etc. We’ll start with a cluster analysis of the full dataset, and then narrow the focus to comparing sightings exclusively in California, Nevada, and Arizona.\n",
        "\n",
        "**Deliverables: **\n",
        "We will provide the following deliverables at the end of the project:\n",
        "A dataset containing reports of UFO sightings\n",
        "A set of insights derived from the dataset\n",
        "A short in-class presentation of our findings, discussions of their meaning, and general “lessons learned” from our project. \n"
      ]
    },
    {
      "metadata": {
        "id": "taTJ6Lwgz5sr",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "# Packages and Installations:"
      ]
    },
    {
      "metadata": {
        "id": "_pzUFQSLuSsi",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 425
        },
        "outputId": "4aac4b0c-6a5b-46cc-8cae-dc51d83adc44"
      },
      "cell_type": "code",
      "source": [
        "#installs any packages not available by default\n",
        "!pip install gensim\n",
        "!pip install wordcloud\n",
        "%time"
      ],
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Requirement already satisfied: gensim in /usr/local/lib/python3.6/dist-packages (3.6.0)\n",
            "Requirement already satisfied: numpy>=1.11.3 in /usr/local/lib/python3.6/dist-packages (from gensim) (1.14.6)\n",
            "Requirement already satisfied: six>=1.5.0 in /usr/local/lib/python3.6/dist-packages (from gensim) (1.11.0)\n",
            "Requirement already satisfied: scipy>=0.18.1 in /usr/local/lib/python3.6/dist-packages (from gensim) (0.19.1)\n",
            "Requirement already satisfied: smart-open>=1.2.1 in /usr/local/lib/python3.6/dist-packages (from gensim) (1.7.1)\n",
            "Requirement already satisfied: boto>=2.32 in /usr/local/lib/python3.6/dist-packages (from smart-open>=1.2.1->gensim) (2.49.0)\n",
            "Requirement already satisfied: bz2file in /usr/local/lib/python3.6/dist-packages (from smart-open>=1.2.1->gensim) (0.98)\n",
            "Requirement already satisfied: boto3 in /usr/local/lib/python3.6/dist-packages (from smart-open>=1.2.1->gensim) (1.9.25)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.6/dist-packages (from smart-open>=1.2.1->gensim) (2.18.4)\n",
            "Requirement already satisfied: jmespath<1.0.0,>=0.7.1 in /usr/local/lib/python3.6/dist-packages (from boto3->smart-open>=1.2.1->gensim) (0.9.3)\n",
            "Requirement already satisfied: s3transfer<0.2.0,>=0.1.10 in /usr/local/lib/python3.6/dist-packages (from boto3->smart-open>=1.2.1->gensim) (0.1.13)\n",
            "Requirement already satisfied: botocore<1.13.0,>=1.12.25 in /usr/local/lib/python3.6/dist-packages (from boto3->smart-open>=1.2.1->gensim) (1.12.25)\n",
            "Requirement already satisfied: urllib3<1.23,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests->smart-open>=1.2.1->gensim) (1.22)\n",
            "Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests->smart-open>=1.2.1->gensim) (3.0.4)\n",
            "Requirement already satisfied: idna<2.7,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests->smart-open>=1.2.1->gensim) (2.6)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests->smart-open>=1.2.1->gensim) (2018.10.15)\n",
            "Requirement already satisfied: docutils>=0.10 in /usr/local/lib/python3.6/dist-packages (from botocore<1.13.0,>=1.12.25->boto3->smart-open>=1.2.1->gensim) (0.14)\n",
            "Requirement already satisfied: python-dateutil<3.0.0,>=2.1; python_version >= \"2.7\" in /usr/local/lib/python3.6/dist-packages (from botocore<1.13.0,>=1.12.25->boto3->smart-open>=1.2.1->gensim) (2.5.3)\n",
            "Requirement already satisfied: wordcloud in /usr/local/lib/python3.6/dist-packages (1.5.0)\n",
            "Requirement already satisfied: numpy>=1.6.1 in /usr/local/lib/python3.6/dist-packages (from wordcloud) (1.14.6)\n",
            "Requirement already satisfied: pillow in /usr/local/lib/python3.6/dist-packages (from wordcloud) (4.0.0)\n",
            "Requirement already satisfied: olefile in /usr/local/lib/python3.6/dist-packages (from pillow->wordcloud) (0.46)\n",
            "CPU times: user 4 µs, sys: 2 µs, total: 6 µs\n",
            "Wall time: 11.4 µs\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "metadata": {
        "id": "NQsEJ6a8jp3e",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 51
        },
        "outputId": "ac04b388-9580-4bbb-c3d2-8de2289df926"
      },
      "cell_type": "code",
      "source": [
        "#importing packages neeeded for Text Analysis\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import nltk\n",
        "import sklearn\n",
        "import gensim\n",
        "import re\n",
        "import string\n",
        "import wordcloud\n",
        "import os\n",
        "%time"
      ],
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "CPU times: user 4 µs, sys: 1 µs, total: 5 µs\n",
            "Wall time: 10 µs\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "metadata": {
        "id": "ouc9eqz9njuJ",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 51
        },
        "outputId": "596c6805-c6c0-4948-847c-5febccfbbddd"
      },
      "cell_type": "code",
      "source": [
        "##Specific Text Mining Features from SKLEARN\n",
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "from sklearn.decomposition import TruncatedSVD\n",
        "from sklearn.pipeline import make_pipeline\n",
        "from sklearn.preprocessing import Normalizer\n",
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "from sklearn.cluster import KMeans\n",
        "from sklearn.metrics.pairwise import cosine_similarity\n",
        "from sklearn.manifold import MDS\n",
        "from sklearn.metrics.pairwise import cosine_similarity\n",
        "#Other specific useful packages\n",
        "from wordcloud import WordCloud\n",
        "from collections import Counter\n",
        "from nltk.stem import WordNetLemmatizer\n",
        "from nltk.corpus import wordnet as wn\n",
        "import matplotlib.pyplot as plt\n",
        "import matplotlib as mpl\n",
        "%time"
      ],
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "CPU times: user 4 µs, sys: 1 µs, total: 5 µs\n",
            "Wall time: 9.78 µs\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "metadata": {
        "id": "xJ7XIEoEszpS",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 153
        },
        "outputId": "7332e606-25b3-4dfa-a127-ce4f030cf304"
      },
      "cell_type": "code",
      "source": [
        "#Downloading features from nltk\n",
        "nltk.download('stopwords')\n",
        "nltk.download('punkt')\n",
        "nltk.download('wordnet')\n",
        "%time"
      ],
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[nltk_data] Downloading package stopwords to /root/nltk_data...\n",
            "[nltk_data]   Package stopwords is already up-to-date!\n",
            "[nltk_data] Downloading package punkt to /root/nltk_data...\n",
            "[nltk_data]   Package punkt is already up-to-date!\n",
            "[nltk_data] Downloading package wordnet to /root/nltk_data...\n",
            "[nltk_data]   Package wordnet is already up-to-date!\n",
            "CPU times: user 4 µs, sys: 1e+03 ns, total: 5 µs\n",
            "Wall time: 10.3 µs\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "metadata": {
        "id": "h7Q2-WCY0CTS",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "# User Defined Functions:"
      ]
    },
    {
      "metadata": {
        "id": "AXqWwedL8cMc",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "#Flatten Function (This will collapse a list of lists into just one list)\n",
        "flatten = lambda l: [item for sublist in l for item in sublist]"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "cKAvXZF6R05e",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "#Stringer\n",
        "\n",
        "def Stringer(list):\n",
        "  new_list = []\n",
        "  for i in list:\n",
        "    new = str(i)\n",
        "    new_list.append(new)\n",
        "  return(new_list)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "f1RrrOrszRa3",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "#Term Vector Function\n",
        "def Term_Vectors(doc):\n",
        "  punc = re.compile( '[%s]' % re.escape( string.punctuation ) )\n",
        "  term_vec = [ ]\n",
        "\n",
        "  for d in doc:\n",
        "      d = str(d)\n",
        "      d = d.lower()\n",
        "      d = punc.sub( '', d )\n",
        "      term_vec.append( nltk.word_tokenize( d ) )\n",
        "\n",
        "  return(term_vec)\n",
        "     "
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "bBd0yw1h1BPu",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "#Stop Word Function\n",
        "def Stop_Word(term_vec, stop_words = nltk.corpus.stopwords.words( 'english' )):\n",
        "\n",
        "  for i in range( 0, len( term_vec ) ):\n",
        "      \n",
        "      term_list = [ ]\n",
        "\n",
        "      for term in term_vec[i]:\n",
        "          if term not in stop_words:\n",
        "              term_list.append( term )\n",
        "\n",
        "      term_vec[i] = term_list\n",
        "\n",
        "  return(term_vec)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "jMRTelCwQo02",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "#Porter Stem Function\n",
        "\n",
        "def Porter_Stem(term_vec):\n",
        "  porter = nltk.stem.porter.PorterStemmer()\n",
        "\n",
        "  for i in range( 0, len( term_vec ) ):\n",
        "    for j in range( 0, len( term_vec[ i ] ) ):\n",
        "      term_vec[ i ][ j ] = porter.stem( term_vec[ i ][ j ] )\n",
        "\n",
        "  return(term_vec)\n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "7BwY2bLzrABj",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "#Lemmatizer Function\n",
        "def lemmatizer(term_vec):\n",
        "  for i in range( 0, len( term_vec ) ):\n",
        "    for j in range( 0, len( term_vec[ i ] ) ):\n",
        "      try: pos = str(wn.synsets(j)[0].pos())\n",
        "      except: pos = \"n\"\n",
        "      term_vec[i][j] = str(WordNetLemmatizer().lemmatize(term_vec[i][j],pos))\n",
        "  return(term_vec)\n",
        "      \n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "TJkQYZDM1R6x",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "##Basic Word Cloud Function\n",
        "\n",
        "def show_wordcloud(data, title = None):\n",
        "    wordcloud = WordCloud(\n",
        "        background_color='white',\n",
        "        max_words=50,\n",
        "        max_font_size=40, \n",
        "        scale=3,\n",
        "        random_state=1 # chosen at random by flipping a coin; it was heads\n",
        "    ).generate(str(data))\n",
        "\n",
        "    fig = plt.figure(1, figsize=(12, 12))\n",
        "    plt.axis('off')\n",
        "    if title: \n",
        "        fig.suptitle(title, fontsize=20)\n",
        "        fig.subplots_adjust(top=2.3)\n",
        "\n",
        "    plt.imshow(wordcloud)\n",
        "    plt.show()\n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "BO3Lpnc90HJc",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "# Initial Data Importation and Cleaning:"
      ]
    },
    {
      "metadata": {
        "id": "8_kzdFuYsIIU",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "#imports ufo dataset from our data.world repo\n",
        "ufoset = pd.read_csv('https://query.data.world/s/t5l7slkbhurybmuxkfgncobbaknf7i')"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "pDQ6U_bymFTA",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 68
        },
        "outputId": "4e335dcf-57ad-481d-95df-c19c8404a502"
      },
      "cell_type": "code",
      "source": [
        "#subsets data by selected states, removes every column but State and Text\n",
        "states = [\"CA\",\"NV\",\"AR\",\"NM\", \"NC\"]\n",
        "subset_ufoset = ufoset.loc[ufoset['state'].isin(states)]\n",
        "\n",
        "encounters = subset_ufoset[['text','state']]\n",
        "\n",
        "#New datasets for each state\n",
        "CA_encounters = encounters.loc[ufoset['state'] == \"CA\"]\n",
        "NV_encounters = encounters.loc[ufoset['state'] == \"NV\"]\n",
        "AR_encounters = encounters.loc[ufoset['state'] == \"AR\"]\n",
        "NM_encounters = encounters.loc[ufoset['state'] == \"NM\"]\n",
        "NC_encounters = encounters.loc[ufoset['state'] == \"NC\"]\n",
        "\n",
        "#Word Vectors\n",
        "All_States = ufoset['text'].values.tolist()\n",
        "SelectStates_vect = encounters['text'].values.tolist()\n",
        "CA_vect = CA_encounters['text'].values.tolist()\n",
        "NV_vect = NV_encounters['text'].values.tolist()\n",
        "AR_vect = AR_encounters['text'].values.tolist()\n",
        "NM_vect = NM_encounters['text'].values.tolist()\n",
        "NC_vect = NC_encounters['text'].values.tolist()\n",
        "\n",
        "print(\"Lists created.\")\n",
        "%time"
      ],
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Lists created.\n",
            "CPU times: user 4 µs, sys: 1e+03 ns, total: 5 µs\n",
            "Wall time: 9.78 µs\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "metadata": {
        "id": "1UN_s-8WvEz3",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "# Begin Text Processing with Term Vectors, Stopwords, and Stemming:"
      ]
    },
    {
      "metadata": {
        "id": "A8OMUtSQuLTJ",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 68
        },
        "outputId": "66b811ef-1b9c-4c7d-fd8f-9859e948611d"
      },
      "cell_type": "code",
      "source": [
        "#Creates Term Vectors for all word vectors\n",
        "\n",
        "All_term = Term_Vectors(All_States)\n",
        "SelectStates_term = Term_Vectors(SelectStates_vect)\n",
        "CA_term = Term_Vectors(CA_vect)\n",
        "NV_term = Term_Vectors(NV_vect)\n",
        "AR_term =Term_Vectors(AR_vect)\n",
        "NM_term =Term_Vectors(NM_vect)\n",
        "NC_term =Term_Vectors(NC_vect)\n",
        "\n",
        "print(\"Term Vectors  Complete.\")\n",
        "%time"
      ],
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Term Vectors  Complete.\n",
            "CPU times: user 5 µs, sys: 0 ns, total: 5 µs\n",
            "Wall time: 10 µs\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "metadata": {
        "id": "WlKaOxc1lVSR",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 68
        },
        "outputId": "ddefac5d-8bba-4a34-98a3-5bcb8960c516"
      },
      "cell_type": "code",
      "source": [
        "stopword = nltk.corpus.stopwords.words('english')\n",
        "custom_words = ['summary','SUMMARY']\n",
        "stopword += custom_words\n",
        "\n",
        "print(\"Stop Words Created.\")\n",
        "%time"
      ],
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Stop Words Created.\n",
            "CPU times: user 7 µs, sys: 0 ns, total: 7 µs\n",
            "Wall time: 14.8 µs\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "metadata": {
        "id": "ZTQqrsgLuPp_",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 68
        },
        "outputId": "eb12d36b-71b5-4ed9-d87a-87ba8dc4c4fe"
      },
      "cell_type": "code",
      "source": [
        "#Stop Word filter for all Vectors\n",
        "All_stop = Stop_Word(All_term,stopword)\n",
        "SelectStates_stop = Stop_Word(SelectStates_term,stopword)\n",
        "CA_stop = Stop_Word(CA_term,stopword)\n",
        "NV_stop = Stop_Word(NV_term,stopword)\n",
        "AR_stop = Stop_Word(AR_term,stopword)\n",
        "NM_stop = Stop_Word(NM_term,stopword)\n",
        "NC_stop = Stop_Word(NC_term,stopword)\n",
        "\n",
        "print(\"Stop Words filter Applied to Term Vectors.\")\n",
        "%time"
      ],
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Stop Words filter Applied to Term Vectors.\n",
            "CPU times: user 0 ns, sys: 4 µs, total: 4 µs\n",
            "Wall time: 9.78 µs\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "metadata": {
        "id": "rfx5NXozq7jf",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 68
        },
        "outputId": "bd8f53fb-9508-49b8-975d-9753e322f920"
      },
      "cell_type": "code",
      "source": [
        "#Lemmatizing for All Vectors\n",
        "#Results look way cleaner than porter stemming\n",
        "\n",
        "All_lem = lemmatizer(All_stop)\n",
        "SelectStates_lem = lemmatizer(SelectStates_stop)\n",
        "CA_lem = lemmatizer(CA_stop)\n",
        "NV_lem = lemmatizer(NV_stop)\n",
        "AR_lem = lemmatizer(AR_stop)\n",
        "NM_lem = lemmatizer(NM_stop)\n",
        "NC_lem = lemmatizer(NC_stop)\n",
        "\n",
        "print(\"Lemmatization Complete.\")\n",
        "%time"
      ],
      "execution_count": 18,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Lemmatization Complete.\n",
            "CPU times: user 5 µs, sys: 0 ns, total: 5 µs\n",
            "Wall time: 10.3 µs\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "metadata": {
        "id": "LrWY9zmH3dIc",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "#Stemming for all vectors   -  Lemmatization was more useful\n",
        " \n",
        "#All_stem = Porter_Stem(All_stop)\n",
        "#SelectStates_stem = Porter_Stem(SelectStates_stop)\n",
        "#CA_stem = Porter_Stem(CA_stop)\n",
        "#NV_stem = Porter_Stem(NV_stop)\n",
        "#AR_stem = Porter_Stem(AR_stop)\n",
        "#NM_stem = Porter_Stem(NM_stop)\n",
        "\n",
        "#print(\"Stemming Applied to Term Vectors.\")\n",
        "#%time"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "ZqGoVJ1A69cJ",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 68
        },
        "outputId": "d067b661-c26b-4eaa-d12a-b0ddba6b8702"
      },
      "cell_type": "code",
      "source": [
        "#Will probably need to refilter the vectors after stemming - not sure how much filter terms are needed yet\n",
        "nextfilter = [\"'\",\"-\",\"look\",\"saw\",\"like\",\"seen\",\"see\",\"could\",\"would\",\"also\",\"got\",\"said\",\"seem\",\"go\",\"well\",\"even\"]\n",
        "\n",
        "All_filt = Stop_Word(All_lem,nextfilter)\n",
        "SelectStates_filt = Stop_Word(SelectStates_lem,nextfilter)\n",
        "CA_filt = Stop_Word(CA_lem,nextfilter)\n",
        "NV_filt = Stop_Word(NV_lem,nextfilter)\n",
        "AR_filt = Stop_Word(AR_lem,nextfilter)\n",
        "NM_filt = Stop_Word(NM_lem,nextfilter)\n",
        "NC_filt = Stop_Word(NC_lem,nextfilter)\n",
        "\n",
        "print(\"Text Filtering Complete\")\n",
        "%time"
      ],
      "execution_count": 20,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Text Filtering Complete\n",
            "CPU times: user 5 µs, sys: 0 ns, total: 5 µs\n",
            "Wall time: 10.7 µs\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "metadata": {
        "id": "iHzlqv2s-FoH",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "## BASIC Visualization (Word Clouds)"
      ]
    },
    {
      "metadata": {
        "id": "oM53fgoHzKc5",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "#Basic Word Clouds just for fun.\n",
        "#show_wordcloud(CA_filt, title = \"Califorina\")\n",
        "#show_wordcloud(NV_filt, title = \"Nevada\")\n",
        "#show_wordcloud(AR_filt, title = \"Arizona\")\n",
        "#show_wordcloud(NM_filt, title = \"New Mexico\")\n",
        "#show_wordcloud(All_filt, title = \"United States\")\n",
        "#%time"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "s0JPYET_zVJp",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "#tfidf Vectorization & K-Means Clustering"
      ]
    },
    {
      "metadata": {
        "id": "N2f0e9HhzR4y",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 68
        },
        "outputId": "df28554f-e1af-4353-c09d-742cd103637a"
      },
      "cell_type": "code",
      "source": [
        "All_tfidf = TfidfVectorizer(All_filt, decode_error = \"replace\")\n",
        "SelectStates_tfidf = TfidfVectorizer(SelectStates_filt, decode_error = \"replace\")\n",
        "CA_tfidf = TfidfVectorizer(CA_filt, decode_error = \"replace\")\n",
        "NV_tfidf = TfidfVectorizer(NV_filt, decode_error = \"replace\")\n",
        "AR_tfidf = TfidfVectorizer(AR_filt, decode_error = \"replace\")\n",
        "NM_tfidf = TfidfVectorizer(NM_filt, decode_error = \"replace\")\n",
        "NC_tfidf = TfidfVectorizer(NC_filt, decode_error = \"replace\")\n",
        "\n",
        "print(\"Tfidf Vectors Complete.\")\n",
        "%time"
      ],
      "execution_count": 22,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Tfidf Vectors Complete.\n",
            "CPU times: user 9 µs, sys: 0 ns, total: 9 µs\n",
            "Wall time: 16.7 µs\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "metadata": {
        "id": "7YOPO660Dn7h",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 68
        },
        "outputId": "57c30cc5-02cc-48af-d068-80bc1cf2015c"
      },
      "cell_type": "code",
      "source": [
        "##Document Similarity Matrices\n",
        "\n",
        "#All_matrix = All_tfidf.fit_transform(ufoset['text'].values.astype('U'))\n",
        "SelectStates_matrix = SelectStates_tfidf.fit_transform(encounters['text'].values.astype('U'))\n",
        "CA_matrix = CA_tfidf.fit_transform(CA_encounters['text'].values.astype('U'))\n",
        "CA_matrix = CA_tfidf.fit_transform(CA_encounters['text'].values.astype('U'))\n",
        "NV_matrix = NV_tfidf.fit_transform(NV_encounters['text'].values.astype('U'))\n",
        "AR_matrix = AR_tfidf.fit_transform(AR_encounters['text'].values.astype('U'))\n",
        "NM_matrix = NM_tfidf.fit_transform(NM_encounters['text'].values.astype('U'))\n",
        "NC_matrix = NC_tfidf.fit_transform(NC_encounters['text'].values.astype('U'))\n",
        "\n",
        "print(\"Similarity Matrices Complete.\")\n",
        "%time"
      ],
      "execution_count": 23,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Similarity Matrices Complete.\n",
            "CPU times: user 5 µs, sys: 0 ns, total: 5 µs\n",
            "Wall time: 10 µs\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "metadata": {
        "id": "9qIkHPzHfx9i",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 68
        },
        "outputId": "5466bdd9-aa60-4fa5-d6e3-9c3cdefe2572"
      },
      "cell_type": "code",
      "source": [
        "#Get term names\n",
        "#All_terms = All_tfidf.get_feature_names()\n",
        "select_terms = SelectStates_tfidf.get_feature_names()\n",
        "CA_terms = CA_tfidf.get_feature_names()\n",
        "NV_terms = NV_tfidf.get_feature_names()\n",
        "AR_terms = AR_tfidf.get_feature_names()\n",
        "NM_terms = NM_tfidf.get_feature_names()\n",
        "NC_terms = NC_tfidf.get_feature_names()\n",
        "\n",
        "print(\"Term Names Complete.\")\n",
        "%time"
      ],
      "execution_count": 25,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Term Names Complete.\n",
            "CPU times: user 6 µs, sys: 0 ns, total: 6 µs\n",
            "Wall time: 11.2 µs\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "metadata": {
        "id": "uEqcfZ_f73Eb",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 68
        },
        "outputId": "cc769e30-aada-4624-c202-978e8bc9b671"
      },
      "cell_type": "code",
      "source": [
        "#Pairwise Similaritiy Distances Calculation\n",
        "\n",
        "#All_dist = 1 - cosine_similarity(All_matrix)\n",
        "SelectStates_dist = 1 - cosine_similarity(SelectStates_matrix)\n",
        "CA_dist = 1 - cosine_similarity(CA_matrix)\n",
        "NV_dist = 1 - cosine_similarity(NV_matrix)\n",
        "AR_dist = 1 - cosine_similarity(AR_matrix)\n",
        "NM_dist = 1 - cosine_similarity(NM_matrix)\n",
        "NC_dist = 1 - cosine_similarity(NC_matrix)\n",
        "\n",
        "print(\"Pairwise Complete Distances Calculated\")\n",
        "%time"
      ],
      "execution_count": 26,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Pairwise Complete Distances Calculated\n",
            "CPU times: user 5 µs, sys: 0 ns, total: 5 µs\n",
            "Wall time: 10 µs\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "metadata": {
        "id": "FVq33TlE1weL",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 68
        },
        "outputId": "db4054a5-56e1-48e6-db74-47e10ebd0f06"
      },
      "cell_type": "code",
      "source": [
        "## KMeans Clustering with n = 5\n",
        "\n",
        "#All_kmeans = KMeans(n_clusters=5,random_state =0).fit(All_matrix)\n",
        "SelectStates_kmeans = KMeans(n_clusters=5,random_state =0).fit(SelectStates_matrix)\n",
        "CA_kmeans = KMeans(n_clusters=5,random_state =0).fit(CA_matrix)\n",
        "NV_kmeans = KMeans(n_clusters=5,random_state =0).fit(NV_matrix)\n",
        "AR_kmeans = KMeans(n_clusters=5,random_state =0).fit(AR_matrix)\n",
        "NM_kmeans = KMeans(n_clusters=5,random_state =0).fit(NM_matrix)\n",
        "NC_kmeans = KMeans(n_clusters=5,random_state =0).fit(NC_matrix)\n",
        "\n",
        "print(\"K-Means Clustering Complete\")\n",
        "%time"
      ],
      "execution_count": 27,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "K-Means Clustering Complete\n",
            "CPU times: user 15 µs, sys: 1 µs, total: 16 µs\n",
            "Wall time: 15.3 µs\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "metadata": {
        "id": "NHu_eIGT6KJo",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 68
        },
        "outputId": "4a246f1c-6a92-4129-9574-68dfece033c7"
      },
      "cell_type": "code",
      "source": [
        "#Get Cluster Labels\n",
        "\n",
        "#All_States_clusters = All_kmeans.labels_.tolist()\n",
        "SelectStates_clusters = SelectStates_kmeans.labels_.tolist()\n",
        "CA_clusters = CA_kmeans.labels_.tolist()\n",
        "NV_clusters = NV_kmeans.labels_.tolist()\n",
        "AR_clusters = AR_kmeans.labels_.tolist()\n",
        "NM_clusters = NM_kmeans.labels_.tolist()\n",
        "NC_clusters = NC_kmeans.labels_.tolist()\n",
        "\n",
        "print(\"Cluster Labels Complete.\")\n",
        "%time"
      ],
      "execution_count": 28,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Cluster Labels Complete.\n",
            "CPU times: user 4 µs, sys: 0 ns, total: 4 µs\n",
            "Wall time: 10 µs\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "metadata": {
        "id": "sKly3aJI8xvK",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "\n",
        "SelectState = {'cluster': SelectStates_clusters, 'State': encounters['state'], \"Text\":encounters['text'] }\n",
        "\n",
        "select_frame = pd.DataFrame(SelectState, columns = ['cluster', 'state', 'text'])"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "7vs5ceJ89zud",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "# Experimental Code for Figureing out Next Steps:"
      ]
    },
    {
      "metadata": {
        "id": "45gpAyFM7fZg",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 68
        },
        "outputId": "a294d971-614f-4430-abb0-6c31967d6da1"
      },
      "cell_type": "code",
      "source": [
        "#Flattening List of Lists of Each State - Might be useful for State Comparisons\n",
        "All_flat = flatten(All_filt)\n",
        "CA_flat = flatten(CA_filt)\n",
        "NV_flat = flatten(NV_filt)\n",
        "AR_flat = flatten(AR_filt)\n",
        "NM_flat = flatten(NM_filt)\n",
        "\n",
        "print(\"Flattened...\")\n",
        "%time"
      ],
      "execution_count": 43,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Flattened...\n",
            "CPU times: user 5 µs, sys: 1e+03 ns, total: 6 µs\n",
            "Wall time: 8.82 µs\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "metadata": {
        "id": "xfCCyci3-yUl",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 51
        },
        "outputId": "d77b7384-961e-4121-9ade-eae235aa9906"
      },
      "cell_type": "code",
      "source": [
        "#Creates a list of lists of our 4 states \n",
        "States = [CA_flat,NV_flat,AR_flat,NM_flat]\n",
        "%time"
      ],
      "execution_count": 44,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "CPU times: user 4 µs, sys: 0 ns, total: 4 µs\n",
            "Wall time: 9.06 µs\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "metadata": {
        "id": "wtDgaDbj8vxL",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 51
        },
        "outputId": "6256bb48-3b40-4b58-a2b5-1506b70e3250"
      },
      "cell_type": "code",
      "source": [
        "#Basic Exploration of Word Counts\n",
        "Counter(All_flat).most_common(50)\n",
        "%time"
      ],
      "execution_count": 46,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "CPU times: user 5 µs, sys: 0 ns, total: 5 µs\n",
            "Wall time: 9.3 µs\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "metadata": {
        "id": "gDCbaS-5vJSs",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 163
        },
        "outputId": "e29cba43-1366-460e-b053-b045dfe4ed72"
      },
      "cell_type": "code",
      "source": [
        "All_kmeans.shape()"
      ],
      "execution_count": 27,
      "outputs": [
        {
          "output_type": "error",
          "ename": "NameError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-27-6a1b9041c450>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mAll_kmeans\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mshape\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m: name 'All_kmeans' is not defined"
          ]
        }
      ]
    }
  ]
}