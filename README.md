# Readme
This repository includes our data, model code, parser code, etc.

## Environment Preparation

CPU: Intel(R) Xeon(R) Platinum 8255C CPU @2.50GHz with 24 core processors, and 86G RAM.

GPU: Two NVIDIA RTX 3090 GPUs with 24 GB memory

Packages: 
`transformers 4.34.1`
`tokenizers 0.14.1`
`torchmetrics 1.2.0`
`torch 2.1.0`
`scikit-learn 1.3.2`
`GPUtil 1.4.0`
`numpy 1.26.1`
`evaluate 0.4.1`
`numba 0.58.1`
`nltk 3.8.1`
`tqdm 4.66.1`
`typing 4.8.0`
`psycopg2 2.9.9`

## Code Files
### Model code
`model_code/model.py`: code for model training. We implement our model with the popular deep learning development framework PyTorch and the python package transformers developed by HuggingFace. 

`model_code/hard_sharing.py`: hard parameter sharing class for model MTL pre-training.

`model_code/soft_sharing.py`: soft parameter sharing class for model MTL pre-training.

`model_code/multitask_data_collator.py`: code for dealing with MTL training data.

`model_code/testing_tc.py` and `model_code/testing_lc.py`: code for model testing. We use two evaluation metrics for the TC task, namely the Accuracy (Acc) of the top prediction and the
MRR for the top-10 recommendations. Five commonly used evaluation metrics are employed for the LC task: EM, ED, BLEU, ROUGE, and METEOR.

`model_code/tokenizer.py`: code for tokenizer.  We use sub-word tokenization with the Byte-Pair Encoding (BPE) algorithm, as previous studies found that BPE can substantially reduce the vocabulary size
and alleviate the OOV problem.

`model_code/modeling_codesage.py`: code for use codesage model.

### Parser code
`parser_code/grammar_list.txt`: A formal grammar for Dockerfiles using the Backus-Naur Form (BNF).
```
<Dockerfile> ::= <Instruction>*
<Instruction> ::= <FROM> | <RUN> | <CMD> | <LABEL> | <EXPOSE> | <ENV> | <ADD> | <COPY> | <ENTRYPOINT> | <VOLUME> | <USER> | <WORKDIR> | <ARG> | <STOPSIGNAL> | <HEALTHCHECK> | <SHELL> | <MAINTAINER> | <ONBUILD>
<FROM> ::= "FROM" [ <option> <value> ] <image> [ ":" <tag> | "@" <digest> ] [ "AS" <alias> | "as" <alias>]
<RUN> ::= "RUN" [ <run_option> <value> ] <command> ( <command_separator> <command> )* | "RUN" <json_commands>
<command_separator> ::= "&&" | "||" | ";"
<CMD> ::= "CMD" <json_commands> | "CMD" <command>
<LABEL> ::= "LABEL" <key_values>
<EXPOSE> := “EXPOSE” <port>+
<ENV> ::= "ENV" <key_values>
<COPY> ::= "COPY"  [ <copy_flags> <value> ] <src>+ <dest>
<ADD> ::= "ADD" [ <add_flags> <value> ]  <src>+ <dest>
<ENTRYPOINT> ::= "ENTRYPOINT" <json_paths> | "ENTRYPOINT" <command>
<VOLUME> ::= "VOLUME" <JSON-paths> | <path>
<USER> ::= "USER" <user> [ ":" <group> ]
<WORKDIR> ::= "WORKDIR" <path>
<ARG> ::= "ARG" <key> [ "=" <value> ]
<STOPSIGNAL> := "STOPSIGNAL" <value>
<HEALTHCHECK> :=  "HEALTHCHECK" [ <healthcheck_options> <value> ] "CMD" <command>
<SHELL> ::= "SHELL" <json_commands>
<MAINTAINER> ::= "MAINTAINER" <value>
<ONBUILD> ::= "ONBUILD" <Instruction>
<from_option> ::= "--platform=" 
<run_option> ::= ( "--mount=" | "--network=" | "--security=" )
<copy_flags> ::= ( "--from=" | "--chown=" | "--chmod=" | "--link=" | "--parents" | "--exclude" )+
<add_flags> ::= ( "--checksum=" | "--chown=" | "--chmod=" | "--keep-git-dir=" | "--link=" | "--exclude=" )+
<healthcheck_options> ::= ( "--interval=" | "--timeout=" | "--start-period=" | "--start-interval=" | "--retries=" )
<image> := <str>
<tag>:= <str>
<digest>:= <str>
<alias> := <str>
<command>:= <str>
<json_commands> :=  "[" <command> ( "," <command> )* "]"
<key_values> := <key>  <value> |  (<key> "=" <value>)+
<key>:= <str>
<value> := <str>
<str> ::= /([^$\n\s]|(\$<var>|\${<var>}))*/
<var> ::= /[a-zA-Z_][a-zA-Z0-9_]*/
<port>:= ((\d+)(\/([a-zA-Z]))?)+
<src>:=<str>
<dest>:=<str>
<path>:=<str>
<json_paths> :=  "[" <path> ( "," <path> )* "]"
<user>:=<str>
<group>:=<str>
```

`parser_code/Dockerfile_syntax_parser.py`: A parser tool (i.e., Dockerfile-syntax-parser) that can parse the Dockerfile textual content into the corresponding syntax type sequence.


### Statistical test code
`stats.R`: code for the Wilcoxon signed-rank tests and plots.


## Data Files
The dataset contains 167,010 instances for pre-training,  89,072 for fine-tuning's training, 11,134 for validation, and 11,134 for testing. 

`data/TC`: Training, validation, and testing data for the TC task.

`data/LC`: Training, validation, and testing data for the LC task.

`data/tokenizer.json`: our tokenizer file.

`data/manual_evaluation_scores.csv`: The manual evaluation scores in terms of similarity and naturalness (372 samples and 2 evaluators).
