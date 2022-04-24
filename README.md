# Anime-Popularity-Prediction
This is Mini Project of Group 11 (SC14) for SC1015.

# Contributers
- Sim Oi Liang (osim001@e.ntu.edu.sg) (Machine Learning)
- Oi Yeek Sheng (oiye0001@e.ntu.edu.sg) (Data Preparation, Data Visualization and Presentation)

# Motivation
- Predict the popularity of the Anime produced based on synopsis, genres and other more 
- Give suggestions on the premiered date based on other Anime 
- Provide Anime Studios with the genres that will be most likely be popular

# Algorithm/Library
- pandas
- seaborn
- numpy
- matplotlib.pyplot
- sklearn
- tensorflow
- keras
- nltk
- IPython


# Notebook 1 - DataPreparation&Visualization
In this notebook, we did the data preparation, such as replacing the value for the unknown, redefining new data we need, and changing the data type. In the data visualization part, we did analyze some relationships between studios, popularity, and source. We also found out about the underrated and overrated anime. The purpose of doing the data visualization is to understand the relationship between each data better and decide which data to use in machine learning.

# Notebook 2 - MachineLearning
(We ran this notebook in Google Colab. Hence, in order to run this successfully on Jupyter Notebook, some modifications are required)

In this notebook, we use the data such as studios, popularity, source, synopsis, and genres to predict the popularity of anime. We used libraries such as TensorFlow, Keras, and one-hot encoding in building our neural network. This system has the features of accepting inputs from the user, such as the Name of Anime, Synopsis, Genres, Studios, Source, and Premiered season and giving out a reasonable popularity score. The highest score of popularity is 7 marks, and the lowest score is 1 mark.

# Reference
- https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020?select=rating_complete.csv
- https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
- https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
- https://machinelearningmastery.com/keras-functional-api-deep-learning/
- https://www.youtube.com/watch?v=51_mlYmcyJk&list=LL&index=7&t=471s
- https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
- https://stackoverflow.com/questions/5486337/how-to-remove-stop-words-using-nltk-or-python
- https://www.geeksforgeeks.org/python-lemmatization-with-nltk/
- https://pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/
- https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
- https://towardsdatascience.com/using-neural-networks-with-embedding-layers-to-encode-high-cardinality-categorical-variables-c1b872033ba2
- https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17
