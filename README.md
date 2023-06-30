# Getting Started

## Set up *Data* Folder

1. Create a new folder called "data" under project directory
2. Make sure that the structure is as follows
   1. negative_polarity
      1. deceptive
         1. fold folders...
      2. truthful
         1. fold folders...
   2. positive_polarity
      1. deceptive
           1. fold folders...
      2. truthful
         1. fold folders...

## Verify that NLTK is installed properly

1. Run the following command: `pip install nltk`
2. You may have to run python and execute the following code:
```commandline
>>> import nltk
>>> nltk.download('punkt')
>>> nltk.download('stopwords')
>>> nltk.download('wordnet')
>>> nltk.download('omw-1.4')
```
