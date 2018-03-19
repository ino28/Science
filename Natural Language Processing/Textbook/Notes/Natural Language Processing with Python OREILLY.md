### 1 NLP, python intro
#### 1.1 nltk
	import nltk
	nltk.download()
basic functions handing *nltk* text

	text4.dispersion_plot(["citizens", "America"])

#### 1.2 texts as lists of words

list, slicing

#### 1.3 statistics

	fdist1 = FreqDist(text1)
	fdist1.keys()
	fdist1.hapaxes() // words that appear only once

#### 1.4 useful functions in python
	## string/text
		s.isalnum()
	
	## word set
		[w for w in wset if len(w) < minLength]
	
	## conditionals/loop/enumeration
		if len(w) < minLength:
			print w
		elif len(w) = minLength:
			# do something else
		else:
			# do something else	
		
		for w in wset:
			print w
			
#### 1.5 natural language understanding
* word sense disambiguation
* pronoun resolution
* generating output
	* question answering
	* machine translation

![Figure 1-5 Simple pipeline architecture for a spoken dialogue system](http://www.nltk.org/images/dialogue.png)


### 2 Corpora, lexical resources
#### 2.1 nltk.corpus
* gutenberg
* webtext
* nps_chat
* brown
* reuters
* inaugural
* others: annotated text corpora in nltk, 다른 언어

##### Text corpus structure

* isolated: gutenberg, webtext, udhr
* categorized: brown
* overlapping: reuters
* temporal: inaugural

##### Corpus readers in nltk
#### 2.2 Conditional frequency distribution

	cfd = nltk.ConditionalFreqDist((genre, word)
				for genre in brown.categories()
				for word in brown.words(categories=genre))

#### 2.3 More on python
#### 2.4 Lexical resources
#### 2.5 WordNet: semantically oriented dictionary

##### hierarchy:
* hypernym/hyponym: car - ambulance, SUV, hack, electric

##### relationships:
* meronym/holonym: tree-burl, crown, stump
* entailment: walk-step
* antonymy: horizontal-vertical

##### semantic similarity:
* lowest common _(e.g. hypernym)
* min depth
* path similarity

### 3 Raw text processing
#### 3.1 text from the web and disk
* html
* search engine results
* RSS feed
* local files
* pdf, msword, others

#### 3.2 processing at string level
#### 3.3 decoding/encoding: Unicode
#### 3.4 regex
#### 3.5 using regex
#### 3.6 normalizing text
* stemmer: lie-lying
* lemmatization: woman-women

#### 3.7 regex for tokenisation
	re.split(pattern, text)
	re.findall(pattern, text)
	
	nltk.regexp_tokenize(text, pattern)

#### 3.8 sentence/word segmentation
#### 3.9 output formatting


### 4 Programming tips
### 5 Tagging
#### 5.5 N-gram tagging
	u_tagger = nltk.UnigramTagger(train_set)
	u_tagger.evaluate(test_set)
	
	u_tagger.tag(unseen_sentence)
	
	## backoff tagger
	tbb = nltk.DefaultTagger('NN')
	tb = nltk.UnigramTagger(train_set, backoff=tbb)
	t = nltk.BigramTagger(train_set, backoff=tb)
	t.evaluate(test_set)
	
	## cutoff: discard contexts that have only been seen N(e.g. <=two) times
	t = nltk.BigramTagger(train_set, cutoff=2, backoff=tb)
	
* Tagging unknown: mark as UNK
* Storing taggers: cPickle

#### 5.6 transformation-based tagging
#### 5.7 clues for tagging
* morphological: internal structure, e.g. -ness, -ing
* syntactic: typical context of occurance, e.g. adj after "very"
* semantic: meaning
* new words
	* note that nouns belongs to **open class** and prepositions **closed class** (a limited category and relatively fixed components)

### 6 Classifying
#### 6.1 supervised classification
* simple classification
* sequence/joint classification
	* consecutive classification (i.e. greedy sequence classification)
* advanced classification
	* transformational joint classification: initial assignment -> iteratively refining to repair inconsistencies between related inputs
	* scores based on probability
		* Hidden Markov Models
		* Maximum Entropy Markov Models
		* Linear-Chain Conditional Random Field Model
		
		*note that to find scores for tag sequences, algorithms can be different
	
#### 6.2 further examples
tasks include recognizing:

* sentence boundary
* dialogue act types
* textual entailment (RTE)

#### 6.3 evaluation

* accuracy: percentage(correctly labeled)
* precision: TP/(labeled P)
* recall: TP/(actual P)
	* type I error: N(irrelevant) labeled as P(relevant)
	* type II error: P(relevant) labeled as N(irrelevant)
* F1 score: 2/ (1/recall + 1/precision)
* confusion matrix: row = reference; col = test
* cross-validation: current fold to test, all other sets for training

#### 6.4 decision trees

decision stump, entropy and information gain

	def entropy(labels):
		fd = nltk.FreqDist(labels)
		probs = [fd.freq(l) for l in fd]
		return -sum(p * math.log(p,2) for p in probs)
	
	def information_gain(original_entropy, new_entropy):
		retrun original_entropy - new_entropy

pruning

#### 6.5 naive bayes classifier

![naive bayes classifier to choose the topic for a document](http://www.nltk.org/images/naive-bayes-triangle.png)

> prior probability × ∏(feature contributions) = label likelihood

##### naive bayes assumption (i.e. independence assumption)
> Feature contributions are independent.

##### smoothing
* expected likelihood estimation: add 0.5 to each *count(f,label)*
* heldout estimation

##### problem of double-counting
solution: add to label likelihood:

> *w[label]* × ∏*(w[f, label])*

, where in naive bayes we set these parameters independently:

> *w[label] = P(label)*

> *w[f, label] = P(f* |*label)*

#### 6.6 maximum entropy classifier

uses iterative optimization, maximising **total likelihood** summing up *P(label* |*features)*, which is the probability that an input with certain *features* will have a label of *label*

* joint-feature: properties of *labeled* values
* context (i.e. simple feature): properties of *unlabeled* values

##### maximum entropy principle
> Among consistent distributions, we should choose the one whose entropy is **highest**, i.e. the labels are more evenly distributed, than a single label dominating.

Whether predicts P(label) before giving/given an *input*, classifiers can be:

* generative: naive bayes classifier
* conditional: maximum entropy classifier

### 7 Information extraciton
#### 7.2 chunking

NP(noun phrase) chunking, chunking grammar, tag patterns,
Chinking(sth like unchunking), I(nside-)O(utside-)B(egin) tags

#### 7.3 chunkers
* regex-based chunker
* n-gram chunker
* classifier-based chunker

#### 7.4 recursion (nested)

chunking with depth (e.g. nested NP), tree traversal

#### 7.5 NER (named entity recogition)
#### 7.6 relation extration

### 8 Grammar analysis

#### 8.2 syntax
coordinate structure (e.g. "NP and NP" equivalent to "NP"), constituent structure (e.g. S - NP relationship in "S (NP VP)", where NP is a (**immediate**) **constituent** of S)

#### 8.3 context-free grammar (CFG)

* structurally ambiguity
	* prepositional phrase attachment ambiguity

recursion

#### 8.4 parsing

* recursive descent parsing: top-down
* shift-reduce parsing: bottom-up
* left-corner parsing: top-down with bottom-up filtering (preprocess to build a possible table)
* chart parsing: with **well-formed substring table**

#### 8.5 dependency

##### dependency grammar
head (usually starting with the main verb in a sentence) - dependent

A dependency graph is **projective** (no crossing when edges expanded)

valency for verbs, complement (must-have) v.s. modifiers (i.e. adjunct, optional)
#### 8.6 treebank, ambiguity challenges, weighted grammar

probabilistic context-free grammar (PCFG)

### 9 Feature-based Grammar

##### 9.1 grammatical features

	NP[NUM=pl]
	|          \
	Det[NUM=pl]  N[NUM=pl]
	|               |
	these           dogs
	
	
	*Det for demonstrative
	
##### 9.2 feature structure

![Rendering a Feature Structure as an Attribute Value Matrix](http://www.nltk.org/images/avm1.png)

	pos tag = N
	agreement features (person = 3, number = pl, gender = female)

directed acyclic graph (DAG), feature path, structure sharing (i.e. reentrancy), subsumption (complete v.s. partial information), unification (merging)

![DAG example](http://www.nltk.org/images/dag03.png)

##### 9.3 extending feature-based grammar

subcategorization (e.g. [SUBCAT=clause]), phrasal level, (phrasal) projection (e.g. N' and N''' are projections of N), maximal projection (e.g. N'''), zero projection (N), auxiliary (for inverted clauses)

![auxiliary verb for inversion case](http://www.nltk.org/book/tree_images/ch09-tree-15.png)

unbounded dependency construction, filler-gap, slash categories

![slash category](http://www.nltk.org/book/tree_images/ch09-tree-16.png)
	
### 10 Meaning analysis

##### 10.2 propositional logic
![propositional logic](https://i.imgur.com/PKQsoqb.png)

##### 10.3 first-order logic

predicate, unary predicate, binary predicate, non-logical v.s. logical constants

type:

* basic types
	* *e*: type of entities
	* *t*: type of formulas, i.e. expressions with truth values
* complex types
	* e.g. *\<e,t>*: from entities to truth values, namely unary predicates

conreferential, open formula, existential quantifier ∃, universal quantifier ∀, bound v.s. unbound, closed (all bounded)

### 11 Linguistic data management
