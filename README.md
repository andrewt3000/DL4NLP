# Deep Learning for NLP resources

Introductory and state of the art resources for NLP sequence modeling tasks like translation and dialog.

##Machine Learning: Neural Networks, RNN, LSTM
[Coursera: Machine Learning - Andrew Ng](https://www.coursera.org/learn/machine-learning/home/welcome?module=tN10A)  
Introductory course for neural networks and more.

[Cousera: Neural Networks - Geoffrey Hinton](https://class.coursera.org/neuralnets-2012-001/lecture)  

[Deep Learning - book by Bengio](http://goodfeli.github.io/dlbook/)

[A few useful things to know about machine learning - pedro domingos](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf)  

[Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)  
Blog post by Chris Olah.  

## Word Vectors
Resources about word vectors, aka word embeddings, and distributed representations for words.  
Word vectors are often used as input to deep learning systems. This process is sometimes called pretraining.  

[Distributed Representations of Words and Phrases and their Compositionality]
(http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)  
Mikolov et al (2013)  Performs well on analogy task and includes [Word2Vec source code](https://code.google.com/p/word2vec/)


[Glove](http://nlp.stanford.edu/projects/glove/)  
Pennington, Socher, Manning.

[Distributed Representations of Sentences and Documents](http://cs.stanford.edu/~quocle/paragraph_vector.pdf)  
Le, Mikolov. 2014.  Concatenates and averages pretrained word vectors to create vectors for sentences, paragraphs and documents. Used for sentiment analysis.  Known as paragraph2vec.

[Deep Learning, NLP, and Representations](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/)  
Blog post by Chris Olah (2014)

## Deep Learning for NLP

[Stanford CS 224D: Deep Learning for NLP class](http://cs224d.stanford.edu/syllabus.html)  
Richard Socher. (2015)  Class with videos, and slides.

[Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.383.1327&rep=rep1&type=pdf)  
Socher et al. 2013.  State of the art paper on deep learning for sentiment analysis using recurrent neural networks.

[A Primer on Neural Network Models for Natural Language Processing](http://u.cs.biu.ac.il/~yogo/nnlp.pdf)  
Yoav Goldberg (October 2015).  No new info, summary of state of the art.  

##Dialog
[A Neural Network Approach toContext-Sensitive Generation of Conversational Responses](http://arxiv.org/pdf/1506.06714v1.pdf)  
Sordoni 2015.  Generates responses to tweets.

[A Neural Conversation Model](http://arxiv.org/pdf/1506.05869v3.pdf)  
Vinyals, Le 2015.  Uses LSTM to generate conversational responses.

[Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks](http://arxiv.org/pdf/1502.05698v7.pdf)  
Weston 2015. Classifies QA tasks. Expands on Memory Networks.
