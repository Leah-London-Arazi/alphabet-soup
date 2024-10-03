# alphabet-soup
Generating high confidence fooling inputs for NLP models.


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
You can run the attack by executing the `main.py` file.

### Examples
Running GCG attack with no additional parameters:
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

### Arguments
- `--attack-name` (required): The name of attack to run.
- `--confidence-threshold`: The minimum confidence threshold for an attack to be considered successful. Default is `0.9`.
- `--model-name`: The HuggingFace model to use. Default is `"cardiffnlp/twitter-roberta-base-sentiment-latest"`.
- `--input-text`: The initial text. If no text is provided, a random sentence will be used.
- `--targeted`: Boolean flag to indicate if the attack is targeted. Default is `True`.
- `--target-class`: The class to attack in targeted mode. Default is `0`.
- `--query-budget`: Maximum number of queries allowed to the model during the attack. Default is `500`.
- `--log-level`: Logging verbosity. Default is `INFO`.
- `--attack-params key=value [key=value ...]`: Additional attack parameters as key-value pairs. For lists, use comma-separated strings.


## Attacks and Attack Parameters
Our attacks are implemented in TextAttack framework, and are defined by a combination of goal function, constraints, search method and transformations. For more details check out https://github.com/QData/TextAttack.
Each attack has additional parameters to control its behavior. Below is an explanation of the parameters for each attack type.

### 1. Character Roulette

#### <u>Black Box</u>
  
**Attack name**: [`character_roulette_black_box_random_char`, `character_roulette_black_box_random_word`]
  
**Parameters**:
  - `swap_threshold` (`float`): A character or a word in the current text would be replaced if `current_score - new_score < swap_threshold`. Default is `0.1`.
  - `num_transformations_per_word` (`int`): Number of transformations to apply per word. Default is `3`.

#### <u>White Box</u>

**Attack name**: `character_roulette_white_box`
  
**Parameters**:
  - `top_n` (`int`): Number of top candidate replacements to consider based on gradient information. Default is `3`.
  - `beam_width` (`int`): The beam width for beam search optimization. Default is `10`.

### 2. Unbounded Drift
#### <u>PEZ</u>

**Attack name**: `pez`
  
**Parameters**:
  - `lr` (`float`): Learning rate. Default is `0.4`.

#### <u>GCG</u>

**Attack name**: `gcg`
  
**Parameters**:
  - `n_samples_per_iter` (`int`): Number of candidates to sample at each gcg algorithm iteration. Default is `20`.
  - `top_k` (`int`): Number of top candidates considered. Default is `256`.

### 3. Meaning Masquerade
Archived by applying the `pez` or `gcg` attacks with additional parameters.

**Attack name**: [`pez`, `gcg`]

**Parameters**:
  - `filter_token_ids_method` (`Optional[FilterTokenIDsMethod]`): Method to filter token IDs (optional).
  - `word_refs` (`list[str]`): List of reference words used in `by_bert_score`or `by_glove_score`.
  - `score_threshold` (`float`): Minimum confidence score used in `by_target_class`. Default is `0.7`.
  - `num_random_tokens` (`int`): Number of random tokens used in `by_random_tokens`. Default is `10`.


## Credits
Special thanks to [Theator](https://theator.io) for providing us compute resources :heart:
