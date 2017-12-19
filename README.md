# SMS-Stylometry

This is a tool that takes as input a SMS-style text message and tries to predict the dialect of English that was used to write it using recurrent neural networks supplemented with data from [Google Trends](https://trends.google.com/trends/). Given the data that the tool was trained on, it can currently assign a prediction for Singaporean, American (US), or British (UK) English.

## Prerequisites

To use this tool, you will need:

* [NLTK](http://www.nltk.org/install.html) - for stopword lists and tokenizers;
* [Keras](https://keras.io/#installation) - for general neural network architecture;
* [Tensorflow](https://www.tensorflow.org/install/) - as a backend to Keras;
* [pytrends](https://github.com/GeneralMills/pytrends) - an API to query Google Trends.

## Usage

`predict_sms.py` is the main script. It will ask the user to input a message and then press enter. Then, it will predict the dialect of the English used in the message according to four trained recurrent neural networks. (More information about these RNNs and their performance on a larger task can be found in the included paper, Coles & Toran Jenner 2017).

Run the script in the command line like this:

```python predict_sms.py```

## Authors

The initial version of this tool (as of December 2017), as well as its evaluation on a large corpus, was built and written by Arlie Coles and Lino Toran Jenner as coursework for COMP 550 - Natural Language Processing at McGill University.
