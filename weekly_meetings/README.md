# Weekly Meeting Notes Outline

[Overleaf report](https://www.overleaf.com/project/679796b5a02b660e4f96beff)

#### Meeting Outline
* [06 February 2025](#date-30-january-2025)
* [20 February 2025](#date-30-january-2025)


#### Date: 20 February 2025

##### Who did you help this week?

##### Who helped you this week?

  - Simon Rasmussen
  - Damian 

##### What did you achieve?


**Hiearchical Contrastive Learning**
Found a hiearchical contrastive learning framework using labels: https://arxiv.org/pdf/2204.13207. 

**Mixing**
In DNABERT-S they use MI-MIX as a domain agnostic data augmentation technique, to mix up hidden representations, which should regularize the model. We thought about a) just copying the MI-MIX, to make our model comparable to DNABERT-S, b) ignoring the MI-MIX, or c) try another mixing strategy. But we also want to focus more of our time on contrastive learning and MIL.   

**MIL**
Defining our MIL and prediction setup. A bag is condidered a patient. Here is a possible outline.

1. Obtain contigs from a patient and do binning, like in our research project. This results in a dictionary that maps every contig to a cluster.
   
2. **Create cluster catalogue (CC)**: We consider every cluster an instance, and there are *k* clusters. A cluster could be represented in 4 different ways:
   1. Cluster centroid matrix $ \in R\ ^{|k|\times d}$, where every instance is the average of contig-embeddings. We can either calculate the centroid using all contigs, or only use contigs from a specific patient.
   2. Distance matrix $ \in R\ ^{|k|\times |k|}$ where each entry denotes the hausdorff distance between two clusters (like the genus-analysis in our research project). We can either calculate distances using all contigs, or only using contigs from a specific patient.
3. **Add abundance vector (AV)**: Consider that every patient has a $1 \times |k|$ vector indicating the abundance of every species for that patient. This abundance vector can be included in 3 ways; a)  multiplied by the cluster catalogue, such that each cluster is weighted by its abundance, b) used directly as input to the network with separate learned attention scores $A_a$, or c) a combination of both. There are 2 ways to compute the abundance:
   1. Frequency-based, where we count the number of contigs in a cluster, divided by the total contigs for that patient. 
   2. Weighted by contig abundance: The abundance of each contig within each patient is an important signal for phenotype prediction. Instead of simply counting contigs like in i), we can do a weighted sum using the contig abundances. 
4. **MIL framework** We can have a model with 2 inputs and 2 learned attention score-vectors. The first input to the model can be the following: A cluster catalogue (CC) weighted by the Abundance Vector $AV$ with learned attention scores $A_{cc}$

$$g[CC \times AV \times A_{cc}]$$

The second input can be the Abundance vector only, with a learned attention score $A_{av}$:

$$h[AV \times A_{av}]$$

A final transformation $l$ combines the two transformations $g$ and $h$ into the output prediction.

$$ output = l(g, h) $$

5. **Model architecture**: We are keen on trying out GNNs with MIL, based on this paper, where they cite you! https://arxiv.org/pdf/1906.04881. This incorporates relationships between species which is a nice property, and we avoid using a cnn on some matrix. 


##### What did you struggle with?

##### What would you like to work on next week?

-Acquiring data, contrastive learning, bioinformatics pipeline. 

##### Where do you need help from Veronika?

Discussion of the MIL setup.
  









#### Date: 06 February 2025

##### Who did you help this week?

##### Who helped you this week?

##### What did you achieve?

Getting more concrete ideas about scope of thesis, and what contributions we can make to DNA language models in metagenomics.

##### What did you struggle with?


##### What would you like to work on next week?

- Prepare for meeting with potential external supervisor: Simon Rasmussen, KU at February 14th.

##### Where do you need help from Veronika?

We need feedback on our Thesis scope ideas.

We generally have 3 areas of interest.
![Overview of Ideas](/images/representation_learning_framework.jpeg)

1. **MLM Pretraining**
   1. A recent paper found that randomly initialized DNA language models perform very similar to ones pretrained with MLM ([Paper](https://www.biorxiv.org/content/10.1101/2024.12.18.628606v1.full.pdf)): *"While pretraining provides double-digit improvements in computer vision and NLP, the gains in genomics are typically within 2-3% and often negative, challenging the effectiveness of current genomic pretraining approaches"*. Simply applying methods from NLP to DNA may not be suitable.
   2. Apply BART noising functions (e.g. text infilling, sentence permutattion).
   3. Change tokenization strategy to [MxDNA](https://arxiv.org/pdf/2412.13716) where tokens a NN is trained to find meaningful tokens, instead of using BPE like DNABERT.
   4. Represent DNA-sequence as a graph, e.g. where nodes are tokens and edges are attention-weights from a DNA language model, e.g. DNABERT: [TokenVizz](https://arxiv.org/pdf/2408.07180).
   5. Use pointwise mutual information to selectively mask most correlated tokens with an anchor token: [paper](https://arxiv.org/pdf/2408.07180).
   6. Use evolutionary information to guide the MLM, and detect sequence variants [GPN-MSA](https://www.biorxiv.org/content/10.1101/2023.10.10.561776v1.full).
   7. None of the genomic tokenization strategies have resulted in substantial performance improvements.
2. **Improve on contrastive learning from DNABERT-S**.
   1. DNABERT-S uses weighted SimCLR, with positive samples from same species, and negative samples from another species. Problem is that two species that are very biologically related could end up far apart in embedding space.
   2. Apply other contrastive learning frameworks: Prototypical contrastive learning (PCL) seems appropiate because it captures the hierarchical semantic structure of the dataset. This has a natural interpretation for us, because species belong to a higher level genus, and so on, from the phylogenetic tree. Maybe some other frameworks can be applied aswell? 
   3. Curriculum learning: In the negative sampling, gradually increase the biological similarity, such that the model first learns to distinguish higher-level structures (e.g. order->genus), and later learns to distinguish species. 
   4. Change the weighting scheme of contrastive learning in DNABERT-S, where every negative samples are assigned a non-learned weight. We could a) use the phylogenetic tree to inform the weighting, or b) introduce a learnable matrix to the weighting (inspired from attention).  
   5. Use Supervised contrastive learning. 
3. **Phenotype Prediction** 
   1. A common approach to represent a person is by an abundance table (binary values indicating presence/abscence of strains), and relative abundance profiles (indicating relative percentage of species). Traditional ML methods use SVM/decision tree directly on the tables, whereas other DL approaches such as (DeepMicro)[https://www.nature.com/articles/s41598-020-63159-5] learns a latent space using a AE, and then classifies from the latent space. 
   2. We are keen on applying multiple instance learning. We think this could be used in conjunction with our contrastive learning and metagenomics binning. Maybe a bag can be all contigs from a person (phenotype prediction), or it could also be all contigs from a cluster identified in the metagenomics binning (to better seperate species). 
   3. We also expect to get some ideas from our meeting on February 14th with Simon Rasmussen.







