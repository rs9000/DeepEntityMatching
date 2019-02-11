# DeepEntityMatching
Entity resolution in PyTorch<br>
Train a classifier to find entity matching between two sources.

## Slide
https://nlp.stanford.edu/projects/glove/


## Word embedding
This model require GloVe or another word embedding<br>
https://rs9000.github.io/assets/docs/slide.pdf

### How to use
```
usage: train.py [-args]

arguments:
  --source1           Source file 1
  --source2           Source file 2
  --separator         Char separator in CSV source files
  --n_attrs           Number of attributes in sources files 
  --mapping           Partial ground truth mapping of sources files
  --blocking_size     Window size of the blocking method (Sorted Neighbourhood)
  --blocking_attr     Attributes of blocking
  --word_embed        Word embedding file
  --word_embed_size   Word embedding vector size
  --save_model        Save trained model
  --load_model        Load pre-trained model
```

## Output 
```
Loading Glove Model...
...Done!  400001  words loaded!
NLP(
  (fc1): Linear(in_features=5, out_features=50, bias=True)
  (fc2): Linear(in_features=50, out_features=2, bias=True)
  (probs): LogSoftmax()
)
Start training...

Epoch: 0
Tot Loss: 3.6684898769376177
Accuracy: 0.6945840312674484
#True Positive: 8 #FP: 138
#True Negative: 1236 #FN 409

....

Epoch: 10
Tot Loss: 0.20634120076961332
Accuracy: 0.954215522054718
#True Positive: 381 #FP: 46
#True Negative: 1328 #FN 36

```
