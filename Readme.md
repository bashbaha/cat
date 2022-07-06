## Channel Adverarial Training for Cross-Channel Text-Independent Speaker Recognition -- Xin Fang et. al

#### This code repository is an unofficial implement about CAT(Channel Adverarial Training) paper above.


### Advantages

1) No need for collections of different channels from a specific speaker. 
 
 Specifically, it needs labeled data under their repective channels: speakerA from channel A, speakerB from channel B is ok. Not ask for speakerA has channelA and channel B at the same time any more.

2) Referring to domain adaption, use Gradient Reversal Layer(GRL) to learn channel-invariant and speaker-discriminative speech representations via channel adversarial training.


3) The paper can not only alleviating the channel mismatch problem, but also outperforms state-of-the-art speaker recognition methods.

---

### Model
1) Input: 500,64,1  filter-bank

2) Pad to short utterance, divide the long utterance into multiple short segments by employing a sliding window without overlap.

3) feature extractor: Generator (2-layer UniLSTMP)

4) speaker label predictor: D1

5) channel predictor: D2

6) optimizer: SGD, initial learnging rate: 0.2, decayed on performance on the development set.

7) batch normalization and dropout are employed

8) batch_size: 64

9) α in Eq.(3) is 1

10) β in Eq.(11) is 1

---

### Dataset
1) paper: 60 utterance 8 seconds each speaker. 8min/speaker

2) dataset1: librispeech 

3) our dataset2: collected from telephone.

---

### Cautions

1) gradient reversal layer may be worse when the model converges at the end.  [refer to: https://www.zhihu.com/question/266710153]

