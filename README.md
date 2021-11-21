# Bi-Matching Mechanism to Combat the Long Tail of Word Sense Disambiguation



Word Sense Disambiguation (WSD) is to predict the sense of the target word in a given context.


![模型结构图](https://github.com/yboys0504/wsd/blob/main/model.png)

The long tail phenomenon of word sense distribution in linguistics causes the Word Sense Disambiguation (WSD) task to face a serious polarization of word sense distribution, that is, Most Frequent Senses (MFSs) with huge sample sizes and Long Tail Senses (LTSs) with small sample sizes. The single matching mechanism model that does not distinguish between the two senses will cause LTSs to be ignored because LTSs are in a weak position. The few-shot learning method that mainly focuses on LTSs is not conducive to grasping the advantage of easy identification of MFSs. This paper proposes a bi-matching mechanism to serve the WSD model to deal with two kinds of senses in a targeted manner, namely definition matching and collocation feature matching. The experiment is carried out under the evaluation framework of English all-words WSD and is better than the baseline models. Moreover, state-of-the-art performance is achieved through data enhancement.



## Dependencies 
To run this code, you'll need the following libraries:
* [Python 3](https://www.python.org/)
* [Pytorch 1.2.0](https://pytorch.org/)
* [Transformers 4.5.1](https://github.com/huggingface/transformers)
* [Numpy 1.17.2](https://numpy.org/)
* [NLTK 3.4.5](https://www.nltk.org/)
* [tqdm](https://tqdm.github.io/)

We used the [WSD Evaluation Framework](http://lcl.uniroma1.it/wsdeval/) for training and evaluating our model.



## Description of files and folders
<b>biencoder.py</b> is the main file of the model and the interface for program execution.
<b>get_wn30.py</b> is a program file for obtaining the meaning of words in WordNet 3.0.

<b>wsd_models</b> folder contains two files, namely <b>util.py</b> and <b>models.py</b>. models.py is the main model of the program, which is called by biencoder.py; util.py is the extracted public method.

<b>data</b> folder contains training data sets, evaluation data sets and development data sets. Its structure is the same as that of the publisher of the dataset [here](http://lcl.uniroma1.it/wsdeval/home).

<b>ckpt</b> folder is the model parameter data saved during the running of the program, and our pre-trained model is also saved in this directory. But because Github has requirements on the size of the uploaded file, we published it to Google Drive, and the specific address is given in this folder.



## How to Run 
To train a biencoder model, run `python biencoder.py --data-path $path_to_wsd_data --ckpt $path_to_checkpoint`. The required arguments are: `--data-path`, which is the filepath to the top-level directory of the WSD Evaluation Framework; and `--ckpt`, which is the filepath of the directory to which to save the trained model checkpoints and prediction files. The `Scorer.java` in the WSD Framework data files needs to be compiled, with the `Scorer.class` file in the original directory of the Scorer file.

<b>For example:</b> python3 biencoder.py --ckpt ckpt --data-path data --multigpu



It is recommended you train this model using the `--multigpu` flag to enable model parallel (note that this requires two available GPUs). More hyperparameter options are available as arguments; run `python biencoder.py -h` for all possible arguments.

To evaluate an existing biencoder, run `python biencoder.py --data-path $path_to_wsd_data --ckpt $path_to_model_checkpoint --eval --split $wsd_eval_set`. Without `--split`, this defaults to evaluating on the development set, semeval2007. The model weights and predictions for the biencoder reported in the paper can be found [here](https://drive.google.com/file/d/1NZX_eMHQfRHhJnoJwEx2GnbnYIQepIQj).

<b>For example:</b> python3 biencoder.py --ckpt ckpt --data-path data --eval --split senseval2 --multigpu



## Model Running Screenshot
Here, we display the running process of the model, with the specific parameter settings of the model at the top of the screen (the same as the results published in the paper).
![模型运行截屏](https://github.com/yboys0504/wsd/blob/main/a1.png)
![模型运行截屏](https://github.com/yboys0504/wsd/blob/main/a2.png)



## Datasets
Due to GitHub's limitation on file upload size, we cannot upload files completely.
The training set, development set, and test set can be found at this address [here](http://lcl.uniroma1.it/wsdeval/home).



## Citation
The paper is not published, until the paper is included, we will be the first time to update the reference method.



## Contact
Please address any questions or comments about this codebase to junwei@tju.edu.cn.

