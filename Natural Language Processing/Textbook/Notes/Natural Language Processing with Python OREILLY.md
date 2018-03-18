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


### 7 Information extraciton
### 8 Grammar analysis
### 9 Feature-based Grammar
### 10 Meaning analysis
### 11 Linguistic data management
