# Counseling and Machine Learning
Carl is a proposed project to train an autonomous agent to do text--based chat counseling. Live counselors chat with clients using reflective listening and record dialog data.  This training data and machine learning train an increasingly autonomous agent capable of counseling clients. Carl is an acronym for computer assisted reflective listener, and an homage to Carl Rogers, the pioneer of reflective listening.   

This is similar to passing the [Turing Test](https://en.wikipedia.org/wiki/Turing_test), where users can't distinguish a live person from a machine.  It was first [posed by Alan Turing in 1950](http://www.loebner.net/Prizef/TuringArticle.html). In [1966  Joseph Weizenbaum created Eliza, a rogerian therapist](http://web.stanford.edu/class/linguist238/p36-weizenabaum.pdf), which is based on keyword matching. More recently, [Ellie, a virtual therapist, has been used in a study to diagnose PTSD](http://www.economist.com/news/science-and-technology/21612114-virtual-shrink-may-sometimes-be-better-real-thing-computer-will-see) for the US military and DARPA. In 2017, Stanford Psychologist Alison Darcy launched [woebot](https://www.woebot.io/), a chatbot programmed to implement cognitive behavioral therapy based on [this research](https://mental.jmir.org/2017/2/e19/).  [X2AI](https://x2.ai/) is a company that plans to create a therapist bot named Tess.  

Deep learning algorithms have advanced state of the art dialog systems ([See Neural Conversation Model (2015)](https://arxiv.org/abs/1506.05869)). Neural Conversation model uses deep learning algorithms to create a chatbot but it suffers from an inconsistent voice. The inconsistent voice problem is conflicting responses often in response to personal or situational questions. The key proposal of this document is that reflective listening techniques avoid the inconsistent voice problem, largely by deflecting personal questions. This document is also a primer on state of the art deep learning dialog algorithms. It concludes with proposals for other potential issues including ethics, privacy and data collection.

# Purpose of reflective listening
[Carl Rogers](https://en.wikipedia.org/wiki/Carl_Rogers) was a popular American psychologist and he pioneered reflective listening. Carl Rogers asserted if he could provide a relationship with empathy, non-judgement, positive regard, and authenticity it would enable [self actualization](https://en.wikipedia.org/wiki/Self-actualization) in the client.  Self actualization is living up to your potential.  It means thinking less rigidly, seeing the world in less black and white terms.  It means being more realistic, more integrated and effective.  It means having higher frustration tolerance, and being more mature.  It means being less defensive, and more adaptive.  It means being less repressed in feeling, and more accepting.  Carl Rogers was very optimistic about human nature and he believed people have "a basically positive direction." Rogers said "The curious paradox is that when I accept myself just as I am, then I can change."  
(For more information see [this video where he describes and demonstrates his techniques](https://youtu.be/ee1bU4XuUyg?t=184) or Carl Roger's book [On Becoming a Person](http://www.amazon.com/On-Becoming-Person-Therapists-Psychotherapy/dp/039575531X))

# Consistent Voice
Because machine learning requires "big data," it will require many different counselors to create enough data. Many AI dialog research papers, inclucding [Neural Convesation Model](https://arxiv.org/abs/1506.05869), use social media and media subtitle sources as training data. Some of these sources are in excess of one billion words, but suffer from an inconsistent voice.  It may be possible to train many counselors using reflective listening to respond consistently. Reflective listening facilitates a consistent voice, by not disclosing personal information.  

Here is an example of inconsistent voice problem from Neural Conversation Model:  
Human: what is your job ?  
Machine: i ’m a lawyer .  
Human: what do you do ?  
Machine: i ’m a doctor .  

#### Open ended questions
The techniques of reflective listening include asking open ended questions.  Open ended questions are questions that can’t be answered with a yes or no response. They serve to move the conversation forward. Examples of open ended questions are: 
- “What is on your mind?” 
- “How did you feel about that?” 
- "What was it like the first time you felt that way?"

#### Reflecting
Reflective listening is the process of expressing back feelings and thoughts. The counselor may say “You seem frustrated” or “You seemed encouraged by that news.” This is an emotional classification problem. Counselors also summarize thoughts, often at higher levels of abstraction. For instance, if a client complains about an incompetent male teacher and his father, the counselor might respond "You are seeking a strong male role model."  

#### Non-directive
Reflective listening solves the inconsistent voice problem by being non-directive, meaning counselors don’t lead the conversation. In reflective listening, you typically track the client's mood and emotion. As an example, counselors don't give adive. If a client asks for direct advice, counselors will deflect by reflection or open ended questions.

Client: “What should I do?”  
Counselor: “What do you think your options are?”  

Client: “What should I do?”  
Counselor: "You wish I could give you some advice about this."


#### Non-disclosure
Counselors using reflective listening don’t disclose personal information. If a client asks, “Do you have kids?,”  a counselor might respond with an open ended question such as “What’s the reason you ask?” or to summarize the underlying emotion  “You are concerned that someone who doesn't have kids won't be able to relate to you.” Because counselors' don't disclose personal information one counselor’s response shouldn’t conflict with another’s.  If you disclosed personal information such as how many children you have  it would conflict with other counselors and not serve as consistent training data. This is a key concept that mitigates the consistent voice problem posed by many large dialog corpora.  
#### Examples of reflective listening
![reflective listening](https://github.com/andrewt3000/carl_voice/raw/master/reflecting.png)

# Machine Learning
Deep learning has made many advancements including speech recognition, object recognition, and machine translation. Companies are using deep learning for many language--related tasks such as to [improving searches](http://searchengineland.com/faq-all-about-the-new-google-rankbrain-algorithm-234440), [image captioning](http://www.wired.com/2015/10/facebook-artificial-intelligence-describes-photo-captions-for-blind-people/), and creating question--answering agents and assistants, like Facebook M, Apple's Siri, Amazon's Alexa, Microsoft's Cortana, and Google's Assistant. Many of these Deep learning techniques are applicable to dialog systems.

#### Sequence to Sequence
[A neural conversation model](http://arxiv.org/pdf/1506.05869v3.pdf) uses the seq2seq framework which was originally designed to do machine translation.  It takes a single sentence like "What is the purpose of life?" and  "translates" it to a single sentence response like "to serve the greater good." This research powers the [Smart Reply feature](http://googleresearch.blogspot.co.uk/2015/11/computer-respond-to-this-email.html) for the Google Inbox app that presents users with several possible replies to emails. The Seq2seq framework has also been used to successfully rewrite Google's production translation services.  (see [Google blog post](https://blog.google/products/translate/found-translation-more-accurate-fluent-sentences-google-translate/) and [New York Times article](http://www.nytimes.com/2016/12/14/magazine/the-great-ai-awakening.html?_r=0) )

#### Diversity
A common problem for generative models is repeating generic phrases such as "I don't know." Diversity in responses is the topic of [Deep Reinforcement Learning for Dialogue Generation](https://arxiv.org/pdf/1606.01541v3.pdf) and [A Diversity-Promoting Objective Function for Neural Conversation Models](https://arxiv.org/abs/1510.03055). 

#### Copying
Reflective listening sometimes requires responding with the user's original statements. This is topic of the paper [Incorporating Copying Mechanism in Sequence-to-Sequence Learning](http://arxiv.org/pdf/1603.06393v3.pdf) Which gives the following examples:

Statement: Hello Jack, my name is **Chandralekha**.  
Response: Nice to meet you, **Chandralekha**.  

Statement: This new guy **doesn’t perform exactly as we expected**.  
Response: What do you mean by "**doesn’t perform exactly as we expected**"?  

#### Memory and Intention
In reflective listening, you typically track the client's mood and emotion, so the requirements for memory are limited.  However, one shortcoming for seq2seq lstm rnns is that they are typically programmed to only have memory for one dialog exchange (statement/response). They are limited to around 79 words in length. However, a dialog system will likely require a more sophisticated ability to focus attention on very long term memory.  Memory and attention mechanisms are an active topic of research.  There are several competing deep learning models like Memory Networks, Neural Turing machines, and Stack RNN.  Within the context of dialog some research is attempting to create vectors for the entire conversation include [Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models](http://arxiv.org/abs/1507.04808) and [Attention with Intention for a Neural Network Conversation Model](http://arxiv.org/abs/1510.08565).

(For more technical information on the state of the art see [Deep learning for NLP](https://github.com/andrewt3000/DL4NLP#deep-learning-for-nlp-resources))

# Other issues
It is uncertain if automated counseling will be effective and it may seem unnatural to automate a personal process such as counseling.  An important factor in counseling is  [empathy and human connection](https://www.youtube.com/watch?v=1Evwgu369Jw). However, in a sense, machine learning is a tool to connect with the original contributors of the training data. There are also potential benefits; one [study](http://www.sciencedirect.com/science/article/pii/S0747563214002647) asserts that people have an increased willingness to disclose information to a computer. 

Privacy is another complicated concern. It is crucial that personal data from one user is not reflected to other users. The typical model in counseling is that user's information is kept confidential. Anonymity may be a practical way to lessen privacy concerns. Bitcoin is an example of private data (individual financial transactions) being protected by anonymity rather than the legacy model of confidentiality in banking.  

It may possible to crowdsource the task of creating this data. Carl Rogers' work was popular and accessible to non-professionals, and he thought these techniques should be used in  interpersonal relationships in general. Many lay people have been trained in reflective listening techniques as volunteers at the [crisis center](http://www.crisiscenterbham.com/), the [national suicide prevention hotline](https://suicidepreventionlifeline.org/), and [7 cups](http://www.7cups.com/), [crisis text line](http://www.crisistextline.org/) and others. These counselors are not required to be licensed by the state but are instead called crisis counselors or listeners.  Although there are licensed counselors available for support, many of the volunteers at these places are unlicensed lay people. As illustrated by [Microsoft Tay](https://en.wikipedia.org/wiki/Tay_(bot)) bot, screening, training and some review of training data are neccessary to crowdsourcing.  

Carl would not likely be used to replace therapists, but to supplement them. It could also be a training, quality assurance or efficiency tool. In the short term, it is likely that Carl could make suggested replies, like the Google smart reply feature, and could improve over time as more data is collected and more advanced AI algorithms are developed.  The potential benefits of a fully autonomous counselor are enormous.  What if everyone in the world, speaking any language, at any time, had a place where they could be understood and accepted without judgment and was more capable of living up to their potential? 

#### Please share this link and contact me at [@andrewt3000](https://twitter.com/andrewt3000)
