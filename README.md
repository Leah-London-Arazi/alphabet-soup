# AlphabetSoup: Generating High Confidence Fooling Texts for NLP Models
Language Models are integrated into increasingly sensitive systems, and thus the motivation for attacking them is growing. Most attacks use adversarial examples, which involve slight modifications of the original input to alter the model's output. In this project, we describe a different approach, finding "fooling texts", which are random-looking texts that are classified with high confidence.

We propose two attack methodologies: *Random Chaos* and *Patterned Chaos*. The former generates unstructured, random prompts, while the latter produces structured, yet incoherent text. Together, these methodologies include a total of four attacks: *Character Roulette* replaces individual characters or words in both white-box and black-box settings. *Unbounded Drift* employs recent gradient-based optimization methods. *Syntactic Sabotage* generates prompts with some syntactic structure, and *Meaning Masquerade* produces prompts from a vocabulary of semantically related words.

Our results demonstrate that "fooling texts" can be efficiently generated using all of our attacks, even in black-box setting, with partial success. Our more successful attack, Unbounded Drift, achieves close to 100% success rate, and generates nonsensical texts that perplex the strong GPT-2 model more than random text. The Patterned Chaos attacks are also very successful, most of which achieve high accuracy scores with a very limited set of tokens.

## Installation
Make sure you have python installed with version `>=3.11`. Using conda:
```shell
conda create -n alphabet-soup python=3.11
conda activate alphabet-soup
```
Install the project's dependencies:
```shell
pip install -r requirements.txt
```

## Running an Attack
You can run one of our implemented attacks by executing the `main.py` file.


### Arguments
- `--attack-name` (required): The name of the attack to run.
- `--confidence-threshold`: The minimum confidence threshold for an attack to be considered successful. Default is `0.9`.
- `--model-name`: The HuggingFace model to use. Default is `"cardiffnlp/twitter-roberta-base-sentiment-latest"`.
- `--input-text`: The initial text. If no text is provided, a random sentence with 20 words will be used.
- `--target-class`: The class to attack. Default is `0`.
- `--query-budget`: Maximum number of model queries allowed during the attack. Default is `500`.
- `--log-level`: Logging verbosity. Default is `INFO`.
- `--attack-params key=value [key=value ...]`: Additional attack parameters as key-value pairs. For lists, use comma-separated strings.


## Attacks and Attack Parameters
Our attacks are implemented using the TextAttack framework, and are defined by a combination of a goal function, constraints, a search method and transformations. For more details check out https://github.com/QData/TextAttack.
Each attack has additional parameters to control its behavior. Below is an explanation of the parameters for each attack type.

### Random Chaos
#### 1. Character Roulette

#### a. Black Box
  
Attack name: [`character_roulette_black_box_random_char`, `character_roulette_black_box_random_word`]
  
Parameters:
  - `swap_threshold` (`float`): A character or a word in the current text would be replaced if `current_score - new_score < swap_threshold`. Default is `0.1`.
  - `num_transformations_per_word` (`int`): Number of transformations to apply per word. Default is `3`.

#### b. White Box

Attack name: `character_roulette_white_box`
  
Parameters:
  - `top_n` (`int`): Number of top candidate replacements to consider based on gradient information. Default is `3`.
  - `beam_width` (`int`): The beam width for beam search optimization. Default is `10`.

#### 2. Unbounded Drift
#### a. PEZ

Attack name: `pez`
  
Parameters:
  - `lr` (`float`): Learning rate. Default is `0.4`.

#### b. GCG

Attack name: `gcg`
  
Parameters:
  - `top_k` (`int`): Number of top candidates to consider based on gradient information. Default is `256`.
  - `n_samples_per_iter` (`int`): Number of candidates to sample from the top k at each gcg algorithm iteration. Default is `20`.

### Patterned Chaos
Implemented using the `pez` or `gcg` attacks with additional parameters.

#### Filter methods
At each step, `pez` and `gcg` attacks select replacement candidates from the available token IDs. 
We implemented four different filtering algorithms to narrow down the token IDs to a specific subset:

1. `by_target_class`: Filters out token IDs that the model classifies as the target class with a certain level of confidence.
2. `by_bert_score`: Selects only the token IDs that have a high [BERT score](https://arxiv.org/pdf/1904.09675) compared to one of the given reference words.
3. `by_glove_score`: Selects only the token IDs that have a high GloVe similarity score to one of the given reference words.
4. `by_random_tokens`: a subset of randomly chosen token IDs.

#### 1. Syntactic Sabotage

Attack name: [`pez`, `gcg`]

Parameters: 
  - `filter_token_ids_method` (`Optional[FilterTokenIDsMethod]`): Use `by_random_tokens` filter method.
  - `num_random_tokens` (`int`): Number of random tokens used. Default is `10`.


#### 2. Meaning Masquerade

Attack name: [`pez`, `gcg`]

Parameters:
  - `filter_token_ids_method` (`Optional[FilterTokenIDsMethod]`): Use one of [`by_target_class`, `by_glove_score`, `by_bert_score`].
  - `word_refs` (`list[str]`): List of reference words used with filter methods `by_bert_score`and `by_glove_score`.
  - `score_threshold` (`float`): The minimum confidence score used in `by_bert_score` and `by_glove_score` filters, and the maximum confidence score used in `by_target_class` filter. Default is `0.7`.


## Examples
Running the GCG based attack with no additional parameters:
```bash
python main.py --attack-name gcg
```
Output:
```
1 (72%) --> 0 (91%)

Original text: FCU6y VqabKWm U'?j=+; }LnTL }&$ MGZN/#? BcjHo3Dx,. {{4T g>A+\q# =wg_e9 EqkK[EA *1UF{:Mlg /CU 7*aMMb R-XS12 =&o!B TTU 8#M* &8D >1&c{EZ{

Perturbed text: FCUigily VqabKOODcheat UsijarilyWhere; }LnTL Hamas cant$ MG DannyN blames abnorm Bicka FaHo3DARS,. {{4T g>A+\q# =wg_e9 EqkK[EA?????1�Bad: ladylg /CU 7*aMMb R-XS12 =&o!B TTU 8#M* &8D >1&c{EZ{

used 381 queries.
```

Running meaning masquerade with PEZ and Glove token IDs filter method:
```bash
python main.py --attack-name pez --attack-params filter_token_ids_method=by_glove_score word_refs="green,tree,flower" --target-class 2
```
Output:
```
1 (77%) --> 2 (90%)

Original text: )cifJhE`oI +{Z $D; #W"Wq@ ux:=$0N N\!*\!-U} g/O$LN ie99DBz qbfM5 J9|nJ ,ErA, 7:j?T+GJ%A [/~|g}w D-H%/"3n 2Xy_ ~.PtyO )My6:Wi cszU OS[ ~[~y.B>;

Perturbed text:  bright bright bright red flower bright brightred bright bright bright red bright red bright red red red bright bright bright bright bright bright bright bright red bright bright red red red redgreenred blue blue yellow blue yellow blue blue blue blue blueblue greyblue blueblueredred bluegreenblue blueorange greyblue yellowredblue blueblue green flower blue blueblueredred bluegreenblue yellow flower blue blue green brightorange flowerbrightorangeblueorangeorange blueredgreenredredredblueredred blueredred blue blue blueredredredgreenbrightbluered

used 352 queries.
```

Running meaning masquerade with PEZ and target class filter method:
```bash
python main.py --attack-name pez --model-name martin-ha/toxic-comment-model --attack-params filter_token_ids_method=by_target_class score_threshold=0.05 --target-class 1
```
Output:
```
0 (100%) --> 1 (96%)

Original text: xMB G(tGg$ ]Huu~ (d)oC{ nGu zI6+ #VQ0w?$ lB6tE t5XWtN; uyXYt]bl b!-mV +^~RH'AW9x V4821~ch< 7Uz]S#} q9v=' SIN pkMu}t}%}O uRQ G&x'wb]1 C{KW~FYe%

Perturbed text: brief.pha eyes invented ি sc tooth a pale rag tooth a stain you! hallway your leather tooth corner leather embrace since rag.rite stone vimes life leather vimes. him leather he!!! eyes! leather strange! eyes tooth! ি. vimes your lips nedra likeless dressed! ᅢ eyes thick tooth!! man him man bones corner thickorescence vimes man ュ yours your! lips! corner the people. ァ vimes 艹 sc vimes vimes vimes ª man man thick des tho manorescence you.. tooth you 彳

used 39 queries.
```

## Credits
Special thanks to [Theator](https://theator.io) for providing us compute resources :heart:
