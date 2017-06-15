# Counseling and Machine Learning
Carl is a project to train an autonomous agent to do text--based chat counseling. It is like Siri or IBM Watson for counseling.  The idea is to use live counselors to chat with clients and record dialog data.  Then use this training data and machine learning to train an increasingly autonomous agent capable of chatting with and counseling clients. Carl is an acronym for computer assisted reflective listener.   

Carl is in the research phase and the biggest challenge is technical feasibility.  This requires passing the [Turing Test](https://en.wikipedia.org/wiki/Turing_test), where users can't distinguish a live person from a machine.  It was first [posed by Alan Turing in 1950](http://www.loebner.net/Prizef/TuringArticle.html). In [1966  Joseph Weizenbaum created Eliza, a rogerian therapist](http://web.stanford.edu/class/linguist238/p36-weizenabaum.pdf), which is based on keyword matching. More recently, [Ellie, a virtual therapist, has been used in a study to diagnose PTSD](http://www.economist.com/news/science-and-technology/21612114-virtual-shrink-may-sometimes-be-better-real-thing-computer-will-see) for the US military and DARPA. Another attempt to create a virtual counselor is described in [Micro-Counseling Dialog System based on Semantic Content](https://www.uni-ulm.de/fileadmin/website_uni_ulm/allgemein/2015_iwsds/iwsds2015_submission_6.pdf) (Han et al. 2015) and [Counseling Dialog System with 5W1H Extraction](http://www.sigdial.org/workshops/conference14/proceedings/pdf/SIGDIAL54.pdf) (Han et al. 2013). In 2017, Stanford Psychologist Alison Darcy launched [woebot](https://www.woebot.io/), a chatbot programmed to implement cognitive behavioral therapy.  [X2AI](https://x2.ai/) is a company that plans to create a therapist bot named Tess.  

However, passing the turing test remains an elusive goal of computer science. Recently, deep learning algorithms have produced promising results ([See Neural Conversation Model](https://arxiv.org/abs/1506.05869)). Creating an autonomous counselor will likely require a combination of a large corpus of training data with a consistent voice and a well tuned or novel machine learning algorithms. This discusses the challenges and possible solutions for these requirements.  

# Traing data
The first challenge is gathering sufficient training data. Because machine learning requires "big data," it will require many different counselors to create enough data. Current dialog research uses social media and movie and tv subtitle sources to train data. These sources are often in excess of one billion words, but suffer from an inconsistent voice.  It is possible to train many counselors using reflective listening to respond consistently. Reflective listening facilitates a consistent voice, by not disclosing personal information.  

[Carl Rogers](https://en.wikipedia.org/wiki/Carl_Rogers) was a popular American psychologist and he pioneered reflective listening. Carl Rogers asserted if he could provide a relationship with empathy, non-judgement, positive regard, and authenticity it would enable [self actualization](https://en.wikipedia.org/wiki/Self-actualization) in the client.  Self actualization is living up to your potential.  It means thinking less rigidly, seeing the world in less black and white terms.  It means being more realistic, more integrated and effective.  It means having higher frustration tolerance, and being more mature.  It means being less defensive, and more adaptive.  It means being less repressed in feeling, and more accepting.  Carl Rogers was very optimistic about human nature and he believed people have "a basically positive direction." Rogers said "The curious paradox is that when I accept myself just as I am, then I can change."  
(For more information see [this video where he describes and demonstrates his techniques](https://youtu.be/ee1bU4XuUyg?t=184) or Carl Roger's book [On Becoming a Person](http://www.amazon.com/On-Becoming-Person-Therapists-Psychotherapy/dp/039575531X))

#### Open ended questions
The techniques of reflective listening include asking open ended questions.  Open ended questions are questions that can’t be answered with a yes or no response. They serve to move the conversation forward. Examples of open ended questions are: 
- “What is on your mind?” 
- “How did you feel about that?” 
- "What was it like the first time you felt that way?"

#### Reflecting
Reflective listening is the process of expressing back feelings and thoughts. The counselor may say “You seem frustrated” or “You seemed encouraged by that news.” This is an emotional classification problem. Counselors also summarize thoughts, often at higher levels of abstraction. For instance, if a client complains about an incompetent male teacher and his father, the counselor might respond "You are seeking a strong male role model."  

#### Non-requirements
Part of what makes reflective listening a task suitable for automation are the things that are not required. For instance, Reflective listening is non-directive, meaning counselors don’t lead the conversation.  

Reflective listening also doesn't require giving advice.  If a client asks for direct advice such as 
- “What should I do?”  

Counselors might respond with an open ended question such as 
- “What do you think your options are?”


#### Non-disclosure
Counselors using reflective listening don’t disclose personal information. If a client asks, “Do you have kids?,”  a counselor might respond with an open ended question such as “What’s the reason you ask?” or to summarize the underlying emotion  “You are concerned that someone who doesn't have kids won't be able to relate to you.” Because counselors' don't disclose personal information one counselor’s response shouldn’t conflict with another’s.  If you disclosed personal information such as how many children you have  it would conflict with other counselors and not serve as consistent training data. Counselors have different personalities, vocabularies, genders, and other factors that could affect their responses but not to the degree of conflicting information. This is a key concept that mitigates the consistent voice problem posed by many large dialog corpora.  

#### CrowdSourcing
It may possible to crowdsource the task of creating this data. Carl Rogers' work was popular and accessible to non-professionals, and he thought these techniques should be used in  interpersonal relationships in general. Many lay people have been trained in reflective listening techniques as volunteers at the [crisis center](http://www.crisiscenterbham.com/), the national suicide prevention hotline, and [7 cups](http://www.7cups.com/) and others. These counselors are not required to be licensed by the state but are instead called crisis counselors or listeners.  Although there are licensed counselors available for support, many of the volunteers at these places are unlicensed lay people. As illustrated by [Microsoft Tay](https://en.wikipedia.org/wiki/Tay_(bot)) bot, screening, training and some review of training data are neccessary to crowdsourcing.  

# Machine Learning
The second challenge is the development and implementation of sufficient computer algorithms. [Deep learning](https://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf), a subfield of machine learning had many advancements including speech recognition, object recognition in images and machine translation. Facebook and Google are using deep learning for many language--related tasks such as to [improving searches](http://searchengineland.com/faq-all-about-the-new-google-rankbrain-algorithm-234440), [creating image captions for the blind](http://www.wired.com/2015/10/facebook-artificial-intelligence-describes-photo-captions-for-blind-people/), and creating question--answering agents and assistants, like [Facebook M](https://www.facebook.com/Davemarcus/posts/10156070660595195).  They are both releasing research and open source tools, like [TensorFlow](http://www.tensorflow.org/), and have a history of releasing [important proprietary research](http://infolab.stanford.edu/~backrub/google.html). Some of these tools and research are also useful for the Carl project.  

#### Word Vectors
One tool released by then Google researcher [Tomas Mikolov](https://scholar.google.com/citations?user=oBu8kMMAAAAJ&hl=en) and others in 2013 is [word2vec](https://code.google.com/p/word2vec/).  It is a program that takes large bodies of text (ideally your training data) and converts each word into a series of numbers called a vector. An important property of these numbers is that similar words have similar numbers.  For instance it knows that banana and pear are similar words. Encoding words into numbers is an important step because Machine learning uses math.  It is also able to perform analogies like: king - man + woman = queen.  This idea can also be extended to phrases. For example the numbers for "Los Angeles" are similar to the numbers for "San Francisco."  The idea can be extended further to sentences and paragraphs.  These are often called thought vectors. 

This functionality, detecting similarity and performing analogies, could form the basic building blocks of a system that is able to adaptively reuse dialog training data.  The system could find similar statements from clients using vector difference calculations and responses could change pronoun genders, verb tenses, and between singular and plural if needed using the analogy capabilities. Word vectors allow language modeling systems to generalize.  

#### Sequence to Sequence
Google researchers Vinyals and Le released a research paper, [A neural conversation model](http://arxiv.org/pdf/1506.05869v3.pdf), where they train an agent using data from an internal tech support chat system, and data from Open Subtitles, which has movie and tv dialog for captioning. They use the seq2seq framework which was originally designed to do machine translation.  It takes a single sentence like "What is the purpose of life?" and  "translates" it to a single sentence response like "to serve the greater good." This [research powers the Smart Reply feature](http://googleresearch.blogspot.co.uk/2015/11/computer-respond-to-this-email.html) for the Google Inbox app that presents users with several possible replies to emails.

Reflective listening sometimes requires responding with the user's original statements. This is topic of the paper [Incorporating Copying Mechanism in Sequence-to-Sequence Learning](http://arxiv.org/pdf/1603.06393v3.pdf) Which gives the following examples:

___
- Hello Jack, my name is **Chandralekha**.
- Nice to meet you, **Chandralekha**.

___
- This new guy **doesn’t perform exactly as we expected**.
- What do you mean by "**doesn’t perform exactly as we expected**"?

One concern is not repeating counselor statements too often. Diversity in responses is the topic of [Deep Reinforcement Learning for Dialogue Generation](https://arxiv.org/pdf/1606.01541v3.pdf). 

Another concern is to not ask questions that have been answered previously. This problem may benefit from research in QA systems.  Here is a potential example of using an open ended question inappropriately:  

- I am sad. My cat died.  
- What is the reason you're sad?

#### Unconventional features
In most dialog systems, the words and their associated vectors are the features. However, other features may improve the quality.

It is likely that keeping a normalized count of the number of exchanges is a useful feature. Consider an example where a user states "hello."  At the begining of a conversation, replying "hello" is likely an appropriate response. In the middle of a conversation stating hello is more likely to be an indication that the user thinks you don't understand or not responding quickly enough. It could also give context to queues that user wants to finish the session.  

Another likely feature is a normalized time since the user's last response. That will give context to statements such as "brb", "Ok I'm back" or "Are you still there?"  

There is no research to support these features as useful. This is speculation.  

#### Memory
In reflective listening, you typically track the client's mood and emotion, so the requirements for memory are limited.  However, one shortcoming for seq2seq lstm rnns is that they are typically programmed to only have memory for one dialog exchange (statement/response). They are limited to around 79 words in length. However, a dialog system will likely require a more sophisticated ability to focus attention on very long term memory.  Memory and attention mechanisms are an [active topic of research](https://research.facebook.com/pages/764602597000662/reasoning-attention-memory-ram-nips-workshop-2015/).  There are several competing models like Memory Networks, Neural Turing machines, and Stack RNN.  
(For more technical information on the state of the art see [Deep learning for NLP](https://github.com/andrewt3000/DL4NLP#deep-learning-for-nlp-resources))

In the short term, it is likely that Carl could make suggested replies, like the Google smart reply feature, and could improve over time as more data is collected and more advanced machine learning algorithms are developed. Choosing between suggested responses,  also has the potential to help train counselors, and enable them to be more consistent and efficient.  

# Non-technical issues and implications
It seems unnatural to automate a personal process such as counseling.  What makes counseling work is  [genuine empathy and human connection](https://www.youtube.com/watch?v=1Evwgu369Jw). In a sense machine learning is a tool to connect with the contributors of the training data. There are also potential benefits; one [study](http://www.sciencedirect.com/science/article/pii/S0747563214002647) asserts that people have an increased willingness to disclose information to a computer. 

Privacy is another complicated concern. It is crucial that personal data from one user is not reflected to other users. The typical model in counseling is that user's information is kept confidential. In addition to confidentiality, anonymity may be a practical way to lessen privacy concerns.      
  
The potential benefits are enormous.  What if everyone in the world, speaking any language, at any time, had a place where they could be understood and accepted without judgment and was more capable of living up to their potential? Carl has applications as a stand alone counseling app. It could also augment other various virtual assistants.  

#### Please share this link and contact me at [@andrewt3000](https://twitter.com/andrewt3000)
