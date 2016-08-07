# Deep Learning for NLP resources

Introductory and state of the art resources for NLP sequence modeling tasks like dialog.

##Machine Learning: Neural Networks, RNN, LSTM
[Coursera: Machine Learning](https://www.coursera.org/learn/machine-learning/home/welcome?module=tN10A)  
Andrew Ng  
Introductory course for linear regression, logistic regression, and neural networks.  
Also covers support vector machines, k-means, etc.  

[Machine Learning for Developers](http://xyclade.github.io/MachineLearning/)  
Intro to basic ML concepts for developers.  

[Deep Learning (Book)](http://goodfeli.github.io/dlbook/)  
[Yoshua Bengio](https://scholar.google.com/citations?user=kukA0LcAAAAJ&hl=en)  
Advanced book about deep learning.

[A few useful things to know about machine learning](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf)  
Pedro Domingos  

[Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)  
Blog post by Chris Olah.  

## Deep Learning for NLP 
[Stanford Natural Language Processing](https://class.coursera.org/nlp/lecture/preview)  
Intro NLP course with videos. This has no deep learning. But it is a good primer for traditional nlp.  

[Stanford CS 224D: Deep Learning for NLP class](http://cs224d.stanford.edu/syllabus.html)  
[Richard Socher](https://scholar.google.com/citations?user=FaOcyfMAAAAJ&hl=en). (2016)  Class with syllabus, and slides.  
Videos: [2015 lectures] (https://www.youtube.com/channel/UCsGC3XXF1ThHwtDo18d7WVw/videos) / [2016 lectures] (https://www.youtube.com/playlist?list=PLcGUo322oqu9n4i0X3cRJgKyVy7OkDdoi)   

[A Primer on Neural Network Models for Natural Language Processing](http://u.cs.biu.ac.il/~yogo/nnlp.pdf)  
Yoav Goldberg. October 2015. No new info, 75 page summary of state of the art.  

## Word Vectors
Resources about word vectors, aka word embeddings, and distributed representations for words.  
Word vectors are numeric representations of words where similar words have similar vectors. Word vectors are often used as input to deep learning systems. This process is sometimes called pretraining. 

[A neural probabilistic language model.](http://papers.nips.cc/paper/1839-a-neural-probabilistic-language-model.pdf)  
Bengio 2003. Seminal paper on word vectors.  

[Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781v3.pdf)  
Mikolov et al. 2013. Word2Vec generates word vectors in an unsupervised way by attempting to predict words from a corpus. Describes Continuous Bag-of-Words (CBOW) and Continuous Skip-gram models for learning word vectors.  
Skip-gram takes center word and predict outside words. Skip-gram is better for large datasets.  
CBOW - takes outside words and predict the center word. CBOW is better for smaller datasets.    
[Distributed Representations of Words and Phrases and their Compositionality]
(http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)  
Mikolov et al. 2013. Learns vectors for phrases such as "New York Times." Includes optimizations for skip-gram: heirachical softmax, and negative sampling. Subsampling frequent words. (i.e. frequent words like "the" are skipped periodically to speed things up and improve vector for less frequently used words)  
[Linguistic Regularities in Continuous Space Word Representations](http://www.aclweb.org/anthology/N13-1090)  
[Mikolov](https://scholar.google.com/citations?user=oBu8kMMAAAAJ&hl=en) et al. 2013. Performs well on word similarity and analogy task.  Expands on famous example: King – Man + Woman = Queen  
[Word2Vec source code](https://code.google.com/p/word2vec/)  
[Word2Vec tutorial](http://tensorflow.org/tutorials/word2vec/index.html) in [TensorFlow](http://tensorflow.org/)  

[word2vec Parameter Learning Explained](http://www-personal.umich.edu/~ronxin/pdf/w2vexp.pdf)  
Rong 2014  

[Deep Learning, NLP, and Representations](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/)  
Chris Olah (2014)  Blog post explaining word2vec.  

[GloVe: Global vectors for word representation](http://nlp.stanford.edu/projects/glove/glove.pdf)  
Pennington, Socher, Manning. 2014. Creates word vectors and relates word2vec to matrix factorizations.  [Evalutaion section led to controversy](http://rare-technologies.com/making-sense-of-word2vec/) by [Yoav Goldberg](https://plus.google.com/114479713299850783539/posts/BYvhAbgG8T2)  
[Glove source code and training data](http://nlp.stanford.edu/projects/glove/) 

## Thought Vectors
Thought vectors are numeric representations for sentences, paragraphs, and documents.  This concept is used for many text classification tasks such as sentiment analysis.      

[Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.383.1327&rep=rep1&type=pdf)  
Socher et al. 2013.  Introduces Recursive Neural Tensor Network.  Uses a parse tree.

[Distributed Representations of Sentences and Documents](http://cs.stanford.edu/~quocle/paragraph_vector.pdf)  
[Le](https://scholar.google.com/citations?user=vfT6-XIAAAAJ), Mikolov. 2014.  Introduces Paragraph Vector. Concatenates and averages pretrained, fixed word vectors to create vectors for sentences, paragraphs and documents. Also known as paragraph2vec.  Doesn't use a parse tree.  
Implemented in [gensim](https://github.com/piskvorky/gensim/).  See [doc2vec tutorial](http://rare-technologies.com/doc2vec-tutorial/)

##Machine Translation
[Neural Machine Translation by jointly learning to align and translate](http://arxiv.org/pdf/1409.0473v6.pdf)  
Bahdanau, Cho 2014.  "comparable to the existing state-of-the-art phrase-based system on the task of English-to-French translation."  Implements attention mechanism.  
[English to French Demo](http://104.131.78.120/)  

[Sequence to Sequence Learning with Neural Networks](http://arxiv.org/pdf/1409.3215v3.pdf)  
Sutskever, Vinyals, Le 2014.  ([nips presentation](http://research.microsoft.com/apps/video/?id=239083)). Uses LSTM RNNs to generate translations. " Our main result is that on an English to French translation task from the WMT’14 dataset, the translations produced by the LSTM achieve a BLEU score of 34.8"  
[seq2seq tutorial](http://tensorflow.org/tutorials/seq2seq/index.html) in [TensorFlow](http://tensorflow.org/).   


##Single Exchange Dialog
[A Neural Network Approach toContext-Sensitive Generation of Conversational Responses](http://arxiv.org/pdf/1506.06714v1.pdf)  
Sordoni 2015.  Generates responses to tweets.   
Uses [Recurrent Neural Network Language Model (RLM) architecture
of (Mikolov et al., 2010).](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)  source code: [RNNLM Toolkit](http://www.rnnlm.org/)

[Neural Responding Machine for Short-Text Conversation](http://arxiv.org/pdf/1503.02364v2.pdf)  
Shang et al. 2015  Uses Neural Responding Machine.  Trained on Weibo dataset.  Achieves one round conversations with 75% appropriate responses.  

[A Neural Conversation Model](http://arxiv.org/pdf/1506.05869v3.pdf)  
Vinyals, [Le](https://scholar.google.com/citations?user=vfT6-XIAAAAJ) 2015.  Uses LSTM RNNs to generate conversational responses. Uses [seq2seq framework](http://tensorflow.org/tutorials/seq2seq/index.html).  Seq2Seq was originally designed for machine transation and it "translates" a single sentence, up to around 79 words, to a single sentence response, and has no memory of previous dialog exchanges.  Used in Google [Smart Reply feature for Inbox](http://googleresearch.blogspot.co.uk/2015/11/computer-respond-to-this-email.html)  

##Memory and Attention Models
Attention mechanisms allows the network to refer back to the input sequence, instead of forcing it to encode all information into one fixed-length vector.  - [Attention and Memory in Deep Learning and NLP](http://www.opendatascience.com/blog/attention-and-memory-in-deep-learning-and-nlp/)  

[Memory Networks](http://arxiv.org/pdf/1410.3916v10.pdf) Weston et. al 2014, and 
[End-To-End Memory Networks](http://arxiv.org/pdf/1503.08895v4.pdf) Sukhbaatar et. al 2015.  
Memory networks are implemented in [MemNN](https://github.com/facebook/MemNN).  Attempts to solve task of reason attention and memory.  
[Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks](http://arxiv.org/pdf/1502.05698v7.pdf)  
Weston 2015. Classifies QA tasks like single factoid, yes/no etc. Extends memory networks.  
[Evaluating prerequisite qualities for learning end to end dialog systems](http://arxiv.org/pdf/1511.06931.pdf)  
Dodge et. al 2015. Tests Memory Networks on 4 tasks including reddit dialog task.  
See [Jason Weston lecture on MemNN](https://www.youtube.com/watch?v=Xumy3Yjq4zk)  
  
[Neural Turing Machines](http://arxiv.org/pdf/1410.5401v2.pdf)  
Graves et al. 2014.  

[Inferring Algorithmic Patterns with Stack-Augmented Recurrent Nets](http://arxiv.org/pdf/1503.01007v4.pdf)  
Joulin, Mikolov 2015. [Stack RNN source code](https://github.com/facebook/Stack-RNN) and [blog post](https://research.facebook.com/blog/1642778845966521/inferring-algorithmic-patterns-with-stack/)  


[Reasoning, Attention and Memory RAM workshop at NIPS 2015. slides included](http://www.thespermwhale.com/jaseweston/ram/)  
