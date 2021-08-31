# Fighting Against the Long Tail of Word Sense Disambiguation Inspired by Children's Literacy Behavior

Word Sense Disambiguation (WSD) is to predict the sense of the target word in a given context.


![模型结构图](https://github.com/yboys0504/wsd/blob/main/model.png)

Word Sense Disambiguation (WSD) is a basic task of Natural Language Processing (NLP), and high-accuracy word sense recognition has a positive effect on subsequent language understanding tasks. However, due to the long tail phenomenon of word sense distribution, pre-trained language models trained on general or public datasets will seriously underestimate low-frequency senses (LFS), which makes it difficult to correct the existing deviations in the subsequent fine-tuning stage. This paper proposes a bi-encoder model, simulating the literacy behavior of children and fully employing glosses and example sentences in a dictionary (i.e., WordNet), to improve the recognition rate of the pre-trained model for LFS in the WSD task. Specifically, we employ one of the encoders to construct the conceptual system of the word and the other to learn its applicable scenario and determine the sense of the word in a given context through the double matching of the conceptual system and the applicable scenario. The experiment is carried out under the WSD evaluation framework proposed by Raganato (2017), and our model outperforms the previous state-of-the-art models. Moreover, our model has reached a new height in the performance of high-frequency senses.


## Dependencies 
To run this code, you'll need the following libraries:
* [Python 3](https://www.python.org/)
* [Pytorch 1.2.0](https://pytorch.org/)
* [Transformers 1.1.0](https://github.com/huggingface/transformers)
* [Numpy 1.17.2](https://numpy.org/)
* [NLTK 3.4.5](https://www.nltk.org/)
* [tqdm](https://tqdm.github.io/)

We used the [WSD Evaluation Framework](http://lcl.uniroma1.it/wsdeval/) for training and evaluating our model.


## How to Run 
To train a biencoder model, run `python biencoder.py --data-path $path_to_wsd_data --ckpt $path_to_checkpoint`. The required arguments are: `--data-path`, which is the filepath to the top-level directory of the WSD Evaluation Framework; and `--ckpt`, which is the filepath of the directory to which to save the trained model checkpoints and prediction files. The `Scorer.java` in the WSD Framework data files needs to be compiled, with the `Scorer.class` file in the original directory of the Scorer file.

It is recommended you train this model using the `--multigpu` flag to enable model parallel (note that this requires two available GPUs). More hyperparameter options are available as arguments; run `python biencoder.py -h` for all possible arguments.

To evaluate an existing biencoder, run `python biencoder.py --data-path $path_to_wsd_data --ckpt $path_to_model_checkpoint --eval --split $wsd_eval_set`. Without `--split`, this defaults to evaluating on the development set, semeval2007. The model weights and predictions for the biencoder reported in the paper can be found [here](https://drive.google.com/file/d/1NZX_eMHQfRHhJnoJwEx2GnbnYIQepIQj).


## Datasets
Due to GitHub's limitation on file upload size, we cannot upload files completely.
The training set, development set, and test set can be found at this address [here](http://lcl.uniroma1.it/wsdeval/home).


## Citation
The paper is not published, until the paper is included, we will be the first time to update the reference method.


## Contact
Please address any questions or comments about this codebase to junwei@tju.edu.cn.

