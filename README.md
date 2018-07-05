# TV Script Generator

## Overview
By using machine learning techniques such as word2vec and LSTM, we can teach a recurrent neural network how to write a TV script by training it on sample Simpsons text. The following modules prepare and process our text data.

## Key Files
__preprocess.py__ - in charge of creating an embedding layer, punctuation tokenization, functions to handle pickle files <br />
__script_generator.py__ - in charge of creating the RNN, training the network, and producing a script <br />
__test_file.py__ - in charge of testing functions in the modules above <br />
__text_viz_analytics.py__ - in charge of creating analytics and visualizations of the text data 

## Getting Started
To get started with this, make sure you set up your local environment
1. Download the script data using the following bash script:
```
mkdir -p data/simpsons
cd data/simpsons
curl https://raw.githubusercontent.com/udacity/deep-learning/master/tv-script-generation/data/simpsons/moes_tavern_lines.txt > moes_tavern_lines.txt
```
2. Install appropriate packages into a virtual environment <br />
Make sure your current directory does not contain any spaces, otherwise you will not be able to pip install the libraries.
```
pip install virtualenv
cd <your_project_folder>
virtualenv my_project
source my_project/bin/activate
pip install -r requirements.txt
```

## How to run
In order to produce an output, first run __preprocess.py__ to prep the data, and then run __script_generator.py__ to produce the script.
```
python preprocess.py && python script_generator.py
```

## Data Summary
This text data is a script from a particular scene in an episode of hit TV show,  "The Simpsons." The following analytics and visualizations were generated in the __text_viz_analytics.py__ script:

__Size of text file:__ 298 kb <br />
__Number of characters in script:__ 305,270 <br />
__Number of words in script:__ 48,986 <br />
__Number of lines in script:__ 4,519 <br />

The following visualization highlights the main characters of the scene based off the amount of times they are mentioned in the script. Naturally, as the file is called "Moe's Tavern Lines," we would expect that the main characters of this episode to be the characters that frequent Moe's tavern the most. The histogram verifies this initial assumption.

<p align="center">
    <img src="https://github.com/michaelchoie/Deep_Learning/blob/master/11.%20generate_tv_script/top_characters.png">
</p>

Next, I created a wordcloud containing the top 50 words of a subset of the tv script data. By producing a wordcloud, we can get a sense of how characters communicated throughout the scene.

In order to produce this wordcloud, I removed all stopwords, punctuation, the names that denoted who was speaking at each line (i.e Homer_Simpson: "blah blah blah"), and words with a length less than 3. My rationale behind this was that this subset of the text would contain more meaningful words.

<p align="center">
    <img src="https://github.com/michaelchoie/Deep_Learning/blob/master/11.%20generate_tv_script/wordcloud.png">
</p>

Although there are some words in this visualization that contain little information such as "hey" or "yeah", the wordcloud also contains words that verify that the text data was indeed a bar scene. For example, it contains words like "bar", "drink", "money", and we can presume that the main characters speaking are Moe and Homer. This fact is vetted by the histogram above.

I ran into a lot of difficulty trying to install the wordcloud package in Python, so I did a workaround using R. To achieve this, I created a sorted dictionary whose keys were words and values were the respective counts, and then I saved that into a JSON file. I loaded this JSON into R and used the wordcloud library to produce a visualization. My R code is as follows:
```
library(rjson)
library(wordcloud)

word_counts <- fromJSON(file="word_counts.json")
unlist_words <- unlist(word_counts)

topwords <- unlist_words[seq(1, 100, 2)]
topcounts <- as.integer(unlist_words[seq(2, 101, 2)])

wordcloud(topwords, topcounts, colors=brewer.pal(8, "Dark2"))
``` 

## Model Strategy
In order to generate a TV script, I trained a Recurrent Neural Network using AWS. The RNN's architecture consisted of a word2vec embedding layer and LSTM cells.

__RNN:__<br />
Using an RNN rather than a feedforward network allows us to train our model on the _sequence_ of words in the data, which in theory should allow us to generate a more coherent sequence of words.

__Word2Vec:__ <br />
We first prepare our data by passing it into an embedding layer. By using this layer, we can bypass performing unnecessary matrix multiplication calculations.

The dimensionality of the input layer must reflect all possible words in the dataset, and yet, we process one word at a time, and only one element will equal 1 while all others will equal zero.

Performing matrix multiplication on this input layer would be a huge waste of computational time as most of the products will equal zero. Instead, what we can do is create an embedding lookup table that has a dimensionality of (# of unique words, # of hidden units).

We assign words to a row index in this lookup table, and the row in this table will reflect the vector that contains the word's hidden unit weights.

__LSTM:__ <br />
LSTM (Long-Short Term Memory) cells are a specific type of memory unit that allow an RNN to retain information on previous words while avoiding the vanishing gradient problem.

LSTM cells have the ability to remove or add information to the cell state, carefully regulated by structures called gates. There are 3 types:

1. Forget gate: decide what information to throw away for new cell state
2. Update gate: decide what new information to store in the new cell state
3. Output gate: output a filtered version of the cell state

LSTM's eliminate the activation function for the hidden state - we simply take the gradient of the identity function, which is 1. As a result, backpropogation would not result in smaller and smaller gradients as we look back further in time.

__Output:__ <br />
In this particular module, we train the RNN so that it updates its probability vector for words within a sequence. Therefore, once we "prime" our network by giving it a word to start off the sequence, it outputs the most likely word to appear next given the previous  words in the sequence.

__Cloud Computing:__ <br />
Although this model can be run on a local computer, I used an AWS GPU cluster to streamline the training process.

## Modeling Performance
After training on the data for 50 epochs, this was the resulting output:
```
homer_simpson: okay, this isn't easy, so i'm just like the car now, and they starve. what youse me miss?
moe_szyslak: huh? usually all that funny. i can pay with my nickels on down.
moe_szyslak: i'm gonna delete" the kid is"...(sighs)
homer_simpson:(noticing phone) moe, you're all right--(worried) i've got an old drink marge
homer_simpson: you're a great ingredient, a beer's moe.
moe_szyslak: homer, this is all right?
moe_szyslak: uh... that's why i gotta be honest for him on the eye? that's not for mr. gumbel, please, tell us feel bad, which has turned out bye night.
homer_simpson:(sadly) i really if you said you made a" best friend, moe. i got the best label.
moe_szyslak: homer, you've been in the highest butts.
moe_szyslak:(uneasy) oh, there's a lot. that's how i am in the bar to hooters.
homer_simpson:(with lenny)
```

The produced TV script doesn't make that much sense, but that is to be expected. We trained on less than a megabyte of text. The point is that we were able to create a program that can generate an original, somewhat cohesive script in a relatively short period of time. And we succeeded! In order to get better results, we'd have to use a smaller vocabulary or get more data. But for demonstration purposes, this particular dataset will suffice. 

## Notes
- I run into errors trying to refer to seq2seq.sequence_loss tensor by name; using the tf.Graph.get_tensor_by_name() method doesn't work, so I made a work around by passing the tensor itself.
