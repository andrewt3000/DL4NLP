# Deep Learning for NLP resources

Introductory and state of the art resources for NLP sequence modeling tasks like dialog.

##Machine Learning: Neural Networks, RNN, LSTM
[Coursera: Machine Learning](https://www.coursera.org/learn/machine-learning/home/welcome?module=tN10A)  
Andrew Ng  
Introductory course for linear regression, logistic regression, and neural networks.  
Also covers support vector machines, k-means, etc.  

[Cousera: Neural Networks](https://class.coursera.org/neuralnets-2012-001/lecture)  
[Geoffrey Hinton](https://scholar.google.com/citations?user=JicYPdAAAAAJ)  
Covers a variety of topics: Neural nets, RNNs, LSTMs.  

[Deep Learning (Book)](http://goodfeli.github.io/dlbook/)  
[Yoshua Bengio](https://scholar.google.com/citations?user=kukA0LcAAAAJ&hl=en)  
Advanced book about deep learning.

[A few useful things to know about machine learning](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf)  
Pedro Domingos  

[Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)  
Blog post by Chris Olah.  

## Word Vectors
Resources about word vectors, aka word embeddings, and distributed representations for words.  
Word vectors are numeric representations of words that are often used as input to deep learning systems. This process is sometimes called pretraining.  

[Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781v3.pdf)  
[Distributed Representations of Words and Phrases and their Compositionality]
(http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)  
[Mikolov](https://scholar.google.com/citations?user=oBu8kMMAAAAJ&hl=en) et al. 2013.  
Generate word and phrase vectors.  Performs well on word similarity and analogy task and includes [Word2Vec source code](https://code.google.com/p/word2vec/)  Subsamples frequent words. (i.e. frequent words like "the" are skipped periodically to speed things up and improve vector for less frequently used words)

[Deep Learning, NLP, and Representations](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/)  
Chris Olah (2014)  Blog post explaining word2vec.  

[Glove](http://nlp.stanford.edu/projects/glove/)  
Pennington, Socher, Manning. Similar to word2vec.  

## Thought Vectors
Thought vectors are numeric representations for sentences, paragraphs, and documents.  Often used for sentiment analysis.

[Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.383.1327&rep=rep1&type=pdf)  
Socher et al. 2013.  Introduces Recursive Neural Tensor Network.

[Distributed Representations of Sentences and Documents](http://cs.stanford.edu/~quocle/paragraph_vector.pdf)  
[Le](https://scholar.google.com/citations?user=vfT6-XIAAAAJ), Mikolov. 2014.  Introduces Paragraph Vector. Concatenates and averages pretrained, fixed word vectors to create vectors for sentences, paragraphs and documents. Also known as paragraph2vec.

[Deep Recursive Neural Networks for Compositionality in Language](https://aclweb.org/anthology/P/P15/P15-1150.pdf)  
Irsoy & Cardie. 2014.  Uses Deep Recursive Neural Networks.  

[Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks](https://aclweb.org/anthology/P/P15/P15-1150.pdf)  
Tai et al. 2015  Introduces Tree LSTM.

## Deep Learning for NLP

[Stanford CS 224D: Deep Learning for NLP class](http://cs224d.stanford.edu/syllabus.html)  
[Richard Socher](https://scholar.google.com/citations?user=FaOcyfMAAAAJ&hl=en). (2015)  Class with videos, and slides.

[A Primer on Neural Network Models for Natural Language Processing](http://u.cs.biu.ac.il/~yogo/nnlp.pdf)  
Yoav Goldberg. October 2015. No new info, summary of state of the art.  

##Dialog
[A Neural Network Approach toContext-Sensitive Generation of Conversational Responses](http://arxiv.org/pdf/1506.06714v1.pdf)  
Sordoni 2015.  Generates responses to tweets.

[A Neural Conversation Model](http://arxiv.org/pdf/1506.05869v3.pdf)  
Vinyals, [Le](https://scholar.google.com/citations?user=vfT6-XIAAAAJ) 2015.  Uses recurrent neural networks and LSTM to generate conversational responses. Uses seq2seq framework.

[Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks](http://arxiv.org/pdf/1502.05698v7.pdf)  
Weston 2015. Classifies QA tasks. Expands on Memory Networks.
