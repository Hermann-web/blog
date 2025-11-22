---
title: "NLP Project: Summarization"
authors: 
    - Wiam ADNAN
    - Aymane Hanine
    - Salma KHMASSI
    - Hermann Agossou
    - Zakaria Taouil
date: 2023-01-15
comments: true
description: This project explores various automatic summarization techniques, emphasizing pre-trained models and fine-tuning.
---

# NLP Project: Summarization

## 1 Introduction

Automatic text summarization is the process of condensing text into a shorter, key-information-retaining version. Abstract summarization, which focuses on reducing an abstract to a few sentences, is crucial in fields like scientific research for summarizing papers.

Recent advancements in machine learning, natural language processing, and deep learning have significantly progressed automatic summarization. Despite this, it remains challenging as models must grasp text meaning and structure to produce accurate and readable summaries.

This project explores various automatic summarization techniques, emphasizing pre-trained models and fine-tuning. We use a news article dataset, evaluating model performance with ROUGE and BLEU metrics to understand the strengths and limitations of different approaches and identify strategies for improving summary quality.

We investigated three approaches:

1. Using a high-performing pre-trained model, then fine-tuning a lower-performing model on a data subset, and evaluating it on a test set.
2. Fine-tuning a model on a different dataset and applying it to our dataset for summarization.
3. Employing a pre-trained extractive model to generate summaries, then fine-tuning an abstractive model on those summaries.

## 2 Evaluation metrics

We utilize ROUGE-1, ROUGE-2, ROUGE-L, and BLEU as evaluation metrics. Here's how they're calculated in text summarization:

* **ROUGE-1 = (number of overlapping unigrams) / (total number of unigrams in the reference summary)**
    To calculate ROUGE-1, both the generated and reference summaries are tokenized into unigrams (single words). The count of shared unigrams is divided by the total unigrams in the reference summary, measuring shared words at the unigram level.

* **ROUGE-2 = (number of overlapping bigrams) / (total number of bigrams in the reference summary)**
    For ROUGE-2, summaries are tokenized into bigrams (pairs of adjacent words). The number of shared bigrams is divided by the total bigrams in the reference summary, indicating shared adjacent word pairs.

* **ROUGE-L = (length of the longest common subsequence) / (total number of words in the reference summary)**
    ROUGE-L is calculated by first finding the longest common subsequence (LCS) between the generated and reference summaries. The LCS length is then divided by the total words in the reference summary, showing how much of the reference summary is captured at the subsequence level.

* **BLEU = expmin(0,1-n/r) \* (product of precision scores for n-grams)**
    BLEU involves tokenizing both summaries into n-grams (typically 1 to 4). Precision for each n-gram set in the generated summary is calculated by dividing shared n-grams by total n-grams. The geometric mean of these precisions is then multiplied by a brevity penalty term (expmin(0,1-n/r), where 'n' is generated summary length and 'r' is reference summary length), assessing n-gram presence while considering length.

## 3 Approach 1

Abstract summarization is a complex task involving condensing long texts into concise, essence-capturing sentences. In this approach, we apply pre-trained deep learning models and fine-tuning to summarize scientific abstracts on AI in agriculture. We compare model performance using Rouge 1, Rouge 2, Rouge L, and BLEU to identify the best and second-best models. To further enhance performance, we fine-tune the second-best model's summaries using those of the best model. This systematic approach aims to generate accurate and effective abstract summaries, valuable for document summarization and information retrieval.

### 3.1 Used Models

The three pre-trained models are:

* **Big Bird Pegasus For Conditional Generation (BBPCG):** BBPCG is a cutting-edge language model with a hybrid Transformer and sequence-to-sequence architecture, excelling in conditional text generation. It processes long inputs and generates high-quality summaries, utilizing an efficient pre-training method for impressive results in text summarization, language modeling, and machine translation.
* **Pegasus-xsum (pegasus):** Pegasus-Xsum is a powerful language model for text summarization, leveraging the Transformer architecture to create high-quality summaries from long inputs. Pre-trained on a vast web text corpus, it generates grammatically correct and semantically accurate summaries. Its decoding strategy includes a length penalty to ensure optimal summary length. Pegasus-Xsum achieves state-of-the-art performance on benchmark summarization datasets.
* **t5-large-finetuned-xsum-cnn (t5-large):** This language model is fine-tuned on the CNN/Daily Mail summarization dataset using the T5 Transformer architecture, generating high-quality summaries that capture news article key information effectively. T5-Large-Finetuned-Xsum-CNN combines large corpus pre-training with summarization task fine-tuning for accurate and fluent summaries, demonstrating superior performance on CNN/Daily Mail and other benchmark datasets.

### 3.2 Steps for generating summaries

The summaries are generated following these main steps:

1. Import the model and tokenizer.
2. Move the model to a specified device (CPU or GPU); in our case, all calculations are performed using GPU in Collab.
3. Set batch size and maximum summary length as hyperparameters.
4. Create an empty list `summaries` to store the generated summaries.
5. Loop through the abstracts in the training dataset in batches of size `batch_size`.
    * Tokenize the batch of abstracts using the tokenizer.
    * Generate summaries for each batch using the specified `max_summary_length` and the model.
    * The model's `generate` method takes encoded input sequences, uses beam search to produce corresponding output sequences, and returns the resulting tensor of output token IDs.
    * Decode the generated token IDs into human-readable text summaries using the tokenizer.
    * Add the batch of summaries to the `summaries` list.

### 3.3 Results on train dataset

We generated summaries for `train_dataset.json` for fine-tuning and evaluated performance on `test_dataset.json`. The results for `train_dataset.json` are as follows:

| model                             | ROUGE-1    | ROUGE-2    | ROUGE-L    | BLEU            |
| :-------------------------------- | :--------- | :--------- | :--------- | :-------------- |
| t5-large-finetuned-xsum-cnn       | 0.169237   | 0.069559   | 0.158777   | 2.746262e-243   |
| BBPCG                             | 0.122082   | 0.024987   | 0.113230   | 7.539752e-214   |
| pegasus-xsum                      | 0.229497   | 0.078772   | 0.210326   | 4.516283e-215   |

*Table 1: Evaluation metrics for different models on the training dataset.*

The table displays the performance of three text summarization models: `t5 – large – finetuned – xsum – cnn`, `BigBirdPegasusForConditionalGeneration_summaries`, and `pegasus – xsum_summary`, across ROUGE-1, ROUGE-2, ROUGE-L, and BLEU metrics. The `pegasus – xsum_summary` model achieved the highest scores in all metrics, followed by `t5 – large – finetuned – xsum – cnn`. The `BigBirdPegasusForConditionalGeneration_summaries` model had the lowest scores. This suggests `pegasus – xsum_summary` may be the most effective for text summarization tasks among these models. However, these results are based on a specific dataset and may not generalize to other datasets or use cases.

### 3.4 Fine-tuning

In this section, we generate summaries from the top-performing model (pegasus) to fine-tune the second-best model (t5-large). We use a dataset named `train_dataset.json`, which contains abstracts and their corresponding summaries from the pegasus model.

Fine-tuning the T5-Large model for text summarization involves training it on a specific summarization task. This entails providing the model with a large dataset of input-output text pairs, where the input is a longer document and the output is a shorter summary. The model learns to generate summaries by adjusting its parameters based on these training examples, fine-tuning hyperparameters like learning rate and batch size, and optimizing the loss function to enhance performance. Once fine-tuning is complete, the T5-Large model can generate summaries for new documents and has demonstrated state-of-the-art results when fine-tuned on diverse datasets.

**Note:** Due to computational limitations, we used a dataset of 300 samples for fine-tuning. On average, summarizing one abstract takes about 1.5 minutes, totaling 100 hours for 4000 samples in the original dataset for a single model.

### 3.5 Evaluation and discussion

To evaluate the ensemble of models and approaches, we used a fixed test dataset called `test_dataset.json`, applying all our models and ideas with the four metrics. The table below shows the results of the three models and the fine-tuned one:

| Model                         | ROUGE-1    | ROUGE-2    | ROUGE-L    | BLEU            |
| :---------------------------- | :--------- | :--------- | :--------- | :-------------- |
| BBPCG                         | 0.128516   | 0.029043   | 0.118884   | 1.241677e-10    |
| Pegasus                       | 0.199311   | 0.055300   | 0.186434   | 5.717189e-215   |
| T5_large                      | 0.186353   | 0.082153   | 0.177092   | 1.226987e-240   |
| T5_fine_tunned_on_Pegasus     | 0.163148   | 0.052152   | 0.147984   | 2.347712e-235   |

*Table 2: Evaluation metrics for different models on test data.*

The results indicate that the "pegasus_summaries" model outperforms others across all evaluation metrics. Its Rouge-1 score of 0.199 is significantly higher, demonstrating better capture of important abstract information. The high Rouge-2 score (0.055) shows its ability to capture key phrases and word combinations, while the Rouge-L score (0.186) indicates good coherence and overall meaning retention.

The "t5_large" model ranks second, with Rouge-2 and Rouge-L scores similar to "pegasus," but a slightly lower Rouge-1 score. This suggests it captures overall meaning well but may be less effective at extracting the most critical information than the best model.

The "t5_fine_tuned_on_pegasus" model, fine-tuned on summaries generated by "pegasus," performed worse than the second-best model. This indicates that fine-tuning on summaries from another model may not always yield superior results.

The "BBPCG" model performed significantly worse than others across all metrics, possibly due to its unsuitability for abstract summarization tasks or lack of fine-tuning on an appropriate dataset.

Overall, "pegasus" appears to be the best model for abstract summarization among those evaluated. However, factors like the training dataset and specific task requirements can influence the optimal model choice.

Fine-tuning a pre-trained language model on a specific dataset can improve performance on downstream tasks. However, in this case, fine-tuning the "t5_large" model for abstract summarization did not yield the expected improvement. This could be due to the small fine-tuning dataset, which may not have captured the task's complexity, or the mismatch between the fine-tuning dataset and the original pre-trained model's training data, limiting the effectiveness of fine-tuning.

## 4 Approach 2

In this second approach, we present an abstract summarization method using pre-trained deep learning models and fine-tuning. Specifically, we employed a pre-trained T5 (Text-to-Text Transfer Transformer) model, which was fine-tuned on XSum, a dataset of news articles and their summaries. We then used this fine-tuned model to summarize 400 abstracts from our dataset. The model's performance was evaluated using metrics such as Rouge 1, Rouge 2, Rouge L, and BLEU.

### 4.1 Used Model

**Text-to-Text Transfer Transformer (T5-small):** T5-small is a pre-trained version of the T5 model with 220 million parameters, making it smaller than T5-3B and T5-11B. It is trained on diverse unsupervised and supervised tasks, utilizing self-attention to learn contextual text representations. T5-small can be fine-tuned on specific tasks with limited labeled data, and it has achieved state-of-the-art performance in natural language processing tasks such as summarization, translation, and question answering. Its adaptable and unified training framework makes it suitable for a wide range of text-to-text tasks.

### 4.2 Dataset for fine-tuning

**Extreme Summarization (Xsum):** For this project, we utilized the XSum dataset (Narayan and Gardent, 2018), a collection of approximately 226,000 news articles from various sources, each with a one-sentence summary. To adapt this dataset, we focused on articles related to agriculture, selecting a subset of 400 articles for fine-tuning our model.

### 4.3 Steps for generating summaries

The summaries are generated following these main steps using the GPU provided on Google Colab:

1. Import necessary libraries, including Hugging Face Transformers and Datasets.
2. Load and tokenize the dataset using the Datasets library.
3. Download and load the pre-trained T5 model checkpoint from the Hugging Face Model Hub.
4. Define a special data collator for sequence-to-sequence models using `DataCollatorForSeq2Seq` from the Transformers library. This collator creates input IDs, attention masks, and labels.

5. Define the optimizer and compile the model using the Adam optimizer and the built-in loss calculation function.
6. Train the model with specified hyperparameters (learning\_rate=2e-5, batch\_size=8, max\_input\_length=1024, min\_target\_length=150, max\_target\_length=250) using the training and validation sets, and the defined metric function for calculating ROUGE scores. We use only 10% for training and 10% for validation due to hardware and time limitations.
7. Generate summaries using the trained model and the summarization pipeline from the Transformers library. The pipeline method takes the trained model and tokenizer as arguments and uses the summarization pipeline.
8. Use metrics such as BLEU, ROUGE-1, ROUGE-2, and ROUGE-L to evaluate the results.

### 4.4 Results on train dataset

We started by generating summaries on both `train_dataset.json` and `test_dataset.json`. Then, we evaluated the performances. The results are as follows:

| model                      | ROUGE-1    | ROUGE-2    | ROUGE-L    | BLEU            |
| :------------------------- | :--------- | :--------- | :--------- | :-------------- |
| T5-small-finetuned-on-Xsum | 0.362165   | 0.217916   | 0.338231   | 1.130762e-10    |

*Table 3: Evaluation metrics for the model.*

The table shows the performance of the `t5small` model fine-tuned on X_sum using ROUGE-1, ROUGE-2, ROUGE-L, and BLEU metrics. The results indicate that this fine-tuning approach yielded lower scores compared to the first approach using large models. This suggests that the `pegasus – xsum_summary` model might be more effective for text summarization tasks. However, this performance could be attributed to the model being fine-tuned on only 10% of Xsum for a single epoch.

### 4.5 Discussion

1. A limitation of this approach is the small amount of training data used. Fine-tuning a pre-trained model on only 10% of the X-sum dataset might not be enough to capture the task's nuances, potentially leading to suboptimal results. Increasing the training data and/or the number of training epochs could improve model performance.

2. The choice of pre-trained model could also affect the quality of generated summaries. While T5 is powerful, other models might be better suited for summarization. Larger models, like those used in the other approach, could have contributed to better performance.

3. Additionally, the quality of the input data can impact model performance. X-sum, a news article dataset, varies widely in topic, tone, and complexity, making some articles harder to summarize. Using an agriculture-specific dataset could enhance performance.

4. Finally, ROUGE scores may not fully capture summary quality. While quantitative, they don't account for coherence, fluency, or readability. Therefore, a qualitative evaluation through manual inspection or user studies is crucial for a complete picture of the model's performance.

## 5 Approach 3

### 5.1 Principle

Abstractive summarization creates human-like summaries by interpreting and paraphrasing text for brevity and readability. This approach uses advanced NLP techniques, deep learning, and neural networks to understand text meaning and generate new, accurate language. While often more readable, it's more complex and can be less accurate than extractive summarization. Extractive summarization, conversely, selects and combines key sentences or phrases from the original text to form a concise and accurate summary. It can be effective for conveying main points, is easier to train with less data, and produces easier-to-evaluate summaries. Therefore, we used an extractive model to generate summaries for our training dataset, which will then be used for fine-tuning an abstractive model.

Here are the steps we followed:

1. **Choose the extractive model and apply it to the train dataset:** We selected `bart_large_xsum_samsum` as our abstractive extractive model. Developed by Facebook AI Research, this language model is pre-trained on a large text corpus, specifically designed for summarization and question answering. Its training data combines the XSUM dataset (manually summarized BBC news articles) and the SAMSUM dataset (questions and answers from Wikipedia articles). The BART model architecture is transformer-based, effective for NLP tasks. `bart_large_xsum_samsum`, a large variant with 406 million parameters, was applied to our training dataset of 300 abstracts, and the resulting summaries were collected.

2. **Choose the abstractive model and fine-tune it with the results of the abstractive extractive model:** We selected `t5_large_finetuned_xsum` as our abstractive model. This T5 model is fine-tuned on the XSum dataset, which comprises short news articles with single-sentence summaries. The "large" in "t5-large" signifies its 770 million parameters, making it more powerful than base or small T5 versions. Recognized as a powerful tool for text summarization, it has achieved state-of-the-art performance on various benchmark datasets. We fine-tuned this model using the summaries generated by the first model and tested it on our test dataset of 100 abstracts.

### 5.2 Process

1. Load/install the necessary libraries.
2. Load the dataset train.
3. Load the abstractive extractive model: bart large xsum samsum.
4. Apply the bart large xsum samsum model to the train dataset.
5. Create a new dataset containing the abstracts of the dataset train and their summaries obtained by applying the extractive abstractive model.
6. Tokenize the new dataset.
7. Load the template abstractive: t5 large finetuned xsum.
8. Define the hyperparameters.
9. Fine tune the t5 large finetuned xsum model.
10. Apply the fine tuned model on the test dataset.
11. Calculate the metrics needed to evaluate the fine tuned model.

### 5.3 Results on train dataset

We generated abstracts for `train_dataset.json` for fine-tuning and then evaluated the performance of the fine-tuned abstractive model on `train_dataset.json`. The results on `train_dataset.json` are as follows:

| model                                     | ROUGE-1    | ROUGE-2    | ROUGE-L    | BLEU            |
| :---------------------------------------- | :--------- | :--------- | :--------- | :-------------- |
| Extractive model bart_large_xsum_samsum   | 0.356324   | 0.244907   | 0.350222   | 1.701539e-222   |
| Abstractive model t5_large_finetuned_xsum | 0.156297   | 0.044861   | 0.1397918  | 3.610569e-239   |

*Table 4: Evaluation metrics for different models.*

The table shows the performance of two different models on four evaluation metrics commonly used in text summarization: ROUGE-1, ROUGE-2, ROUGE-L, and BLEU.

The first model is an extractive model called "bart\_large\_xsum\_samsum", and the second model is an abstractive model called "t5\_large\_finetuned\_xsum".

In terms of performance, the extractive model outperforms the abstractive model on all four evaluation metrics. Specifically, the extractive model achieves ROUGE-1, ROUGE-2, and ROUGE-L scores of 0.356, 0.245, and 0.350, respectively, while the abstractive model achieves scores of 0.156, 0.045, and 0.140, respectively. The extractive model also achieves a higher BLEU score of 1.701539e-222 compared to the abstractive model's score of 3.610569e-239.

### 5.4 Fine-tuning the abstractive model

Extractive summarization methods identify and select crucial sentences or phrases from a source document, then stitch them together to form a summary. These summaries tend to be more factually accurate and capture the main ideas. By providing these extractive summaries as inputs to the abstractive model during fine-tuning, the abstractive model can learn from these important sentences and phrases, generating more accurate and informative summaries. This approach helps the abstractive model focus on key information, avoiding errors or irrelevant details.

### 5.5 Results on test dataset

| model                         | ROUGE-1    | ROUGE-2    | ROUGE-L    | BLEU            |
| :---------------------------- | :--------- | :--------- | :--------- | :-------------- |
| Abstractive model             | 0.166340   | 0.049878   | 0.148912   | 5.442148e-235   |
| Finetuned Abstractive model   | 0.141666   | 0.037907   | 0.128724   | 3.023488e-241   |

*Table 5: Evaluation metrics for different models.*

The fine-tuned abstractive model's performance was evaluated using standard metrics (ROUGE-1, ROUGE-2, ROUGE-L, and BLEU). The results showed lower scores across all metrics compared to the pre-trained model. This is a common challenge in fine-tuning, as it requires balancing pre-trained knowledge with adaptation to a new domain. It's possible the original model's pre-trained knowledge wasn't fully compatible with the extractive summaries used for fine-tuning, leading to a performance decrease.

Another potential reason for reduced performance is the quality of the extractive summaries used during fine-tuning. Extractive summarization methods aren't perfect; they can sometimes miss important information or include irrelevant details. Using lower-quality extractive summaries for fine-tuning could lead to the abstractive model learning from incorrect or incomplete information, resulting in decreased performance.

## 6 Conclusion

In conclusion, the three approaches presented describe different methods for abstract summarization using pre-trained deep learning models and fine-tuning techniques. The first approach focused on using three large pre-trained models and fine-tuning the second-best model on the best model's summaries to enhance performance. The second approach used a single pre-trained model fine-tuned on a small subset of news articles in the agriculture domain. The third approach employed an abstractive-extractive hybrid method, first using an extractive model to generate summaries and then fine-tuning an abstractive model on those summaries.

While each approach has strengths and limitations, evaluation metrics suggest that the first approach achieved the highest scores across all metrics, followed by the third and second approaches. However, it's crucial to note that the metrics used (BLEU, ROUGE-1, ROUGE-2, and ROUGE-L) don't always fully capture the quality or coherence of generated summaries.

Overall, these approaches offer valuable insights into various abstract summarization methods and how pre-trained deep learning models and fine-tuning can improve performance. These methods have potential applications in document summarization, information retrieval, and other areas requiring summary generation.

## Project Members

* Wiam ADNAN
* Aymane Hanine
* Salma KHMASSI
* Hermann Agossou
* Zakaria Taouil
