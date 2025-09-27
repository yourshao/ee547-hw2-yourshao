
zhenquns@usc.edu

Problem 2 is a bag-of-words autoencoder that compresses each abstract into a 64-dimensional embedding and tries to 
reconstruct its vocabulary presence vector.
It first cleans the text, retaining only alphabetic tokens, 
and builds a vocabulary of the 5,000 most frequent words. 
Each abstract is then encoded into a multi-hot vector representing 
the presence of these words, truncated to the first 150 tokens. 
The encoder compresses this high-dimensional input through two linear 
layers (5000→128→64) to produce a 64-dimensional document embedding, 
while the decoder mirrors this structure (64→128→5000) with a final 
Sigmoid activation to reconstruct the original word presence vector. 
Training uses binary cross-entropy loss so that the reconstruction 
closely matches the input, forcing the 64-dimensional embedding to 
capture salient semantic patterns. The model outputs embeddings for 
each abstract along with reconstruction losses, and it saves the 
vocabulary, model parameters, and training logs.

