# Weekly Meeting Notes Outline

[Overleaf report](https://www.overleaf.com/project/679796b5a02b660e4f96beff)

#### Meeting Outline
* [06 February 2025](#date-30-january-2025)

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







