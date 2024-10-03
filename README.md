# alphabet-soup
Generating high confidence fooling inputs for NLP models.


## Attacks and Attack Parameters
Our attacks are implemented in TextAttack framework, and are defined by a combination of goal function, constraints, search method and transformations. For more details check out https://github.com/QData/TextAttack.
Each attack has additional parameters to control its behavior. Below is an explanation of the parameters for each attack type.

### 1. Character Roulette

#### Black Box
  
**Attack name**: [`character_roulette_black_box_random_char`, `character_roulette_black_box_random_word`]
  
**Parameters**:
  - `swap_threshold` (`float`): A character or a word in the current text would be replaced if `current_score - new_score < swap_threshold`. Default is `0.1`.
  - `num_transformations_per_word` (`int`): Number of transformations to apply per word. Default is `3`.

#### White Box

**Attack name**: `character_roulette_white_box`
  
**Parameters**:
  - `top_n` (`int`): Number of top candidate replacements to consider based on gradient information. Default is `3`.
  - `beam_width` (`int`): The beam width for beam search optimization. Default is `10`.

### 2. Unbounded Drift
#### PEZ

**Attack name**: `pez`
  
**Parameters**:
  - `lr` (`float`): Learning rate. Default is `0.4`.

#### GCG

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


## Running an Attack
You can run the attack by executing the `main.py` file.

### Examples:

```bash
python main.py --attack-name pez
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

