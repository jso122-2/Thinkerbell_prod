---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:2688
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-mpnet-base-v2
widget:
- source_sentence: 'Hey team, looking to work with Olivia Davis for McDonald''s. Budget
    is approx $13,705 - need 3 x Blog post. 2 weeks exclusivity, 6 month engagement.
    Usage: 11 months. Start date: November 2025. Need to confirm: exclusivity period'
  sentences:
  - brand|campaign|client|deliverables|engagement_term|exclusivity_period|fee|usage_term
  - 'Need Quinn Cook for Mad Mex campaign. Budget around $14,807, looking for 2 x
    Media interview, 1 x Pinterest pin. 9 week exclusivity from other similar brands.
    Campaign runs September 2025. Usage rights for 11 months. Engagement period: 2
    months.'
  - brand|campaign|client|deliverables|engagement_term|exclusivity_scope|fee|usage_term
- source_sentence: "Mia White for Freedom - $2,843 budget\n            Need: 3 x Facebook\
    \ story\n            9w exclusivity, 2m engagement\n            Usage rights:\
    \ 9 months\n            Start: December 2025"
  sentences:
  - 'Hey team, looking to work with Mia White for Coles. Budget is approx $30,784
    - need 3 x YouTube short, 1 x Instagram story, 3 x Social media takeover. 9 weeks
    exclusivity, 5 month engagement. Usage: 6 months. Start date: December 2025. Need
    to confirm: usage rights'
  - "• Influencer: Mia White\n            • Brand: Priceline\n            • Fee: ~$25,205\n\
    \            • Deliverables: 3 x Pinterest pin\n            • Exclusivity: 12\
    \ weeks\n            • Engagement: 4 months\n            • Usage: 11 months\n\
    \            • Start: October 2025"
  - "Chloe Lewis for Grill'd - $19,120 budget\n            Need: 3 x Behind-the-scenes\
    \ content, 2 x YouTube short, 1 x Brand ambassador content, 2 x Press conference,\
    \ 2 x Product photography\n            12w exclusivity, 5m engagement\n      \
    \      Usage rights: 7 months\n            Start: September 2025"
- source_sentence: 'Need Cameron Parker for Grill''d campaign. Budget around $17,795,
    looking for 3 x 30-second Instagram reel, 2 x Facebook post. 2 week exclusivity
    from other similar brands. Campaign runs October 2025. Usage rights for 12 months.
    Engagement period: 6 months.'
  sentences:
  - "• Influencer: Lily Walker\n            • Brand: Officeworks\n            • Fee:\
    \ ~$6,287\n            • Deliverables: 3 x LinkedIn post, 1 x TV appearance, 3\
    \ x Facebook post\n            • Exclusivity: 10 weeks\n            • Engagement:\
    \ 4 months\n            • Usage: 12 months\n            • Start: January 2026"
  - 'Need Kevin Baker for Myer campaign. Budget around $5,125, looking for 3 x Radio
    interview. 12 week exclusivity from other fashion brands. Campaign runs September
    2025. Usage rights for 11 months. Engagement period: 6 months.'
  - "• Influencer: Quinn Richardson\n            • Brand: Grill'd\n            • Fee:\
    \ ~$32,521\n            • Deliverables: 3 x 30-second Instagram reel, 1 x Product\
    \ photography\n            • Exclusivity: 6 weeks\n            • Engagement: 3\
    \ months\n            • Usage: 8 months\n            • Start: September 2025\n\
    \nNeed to confirm: usage rights"
- source_sentence: 'Need Rowan Bailey for Cotton On campaign. Budget arnd $21,968,
    looking for 3 x LinkedIn post, 1 x Facebook story, 2 x 90-second product demo,
    3 x Lifestyle photography, 1 x 15-second TikTok. 11 week exclusivity from other
    fashion brands. Campaign runs January 2026. Usage rights for 11 months. Engagement
    period: 2 months.'
  sentences:
  - 'Hey team, looking to work with Peyton Stewart for Cotton On. Budget is approx
    $26,631 - need 1 x Facebook post. 5 weeks exclusivity, 6 month engagement. Usage:
    9 months. Start date: November 2025.'
  - brand|campaign|client|deliverables|engagement_term|exclusivity_period|fee|usage_term
  - brand|campaign|client|deliverables|engagement_term|exclusivity_period|fee|usage_term
- source_sentence: 'Need Zoe Wright for Supercheap Auto campaign. Budget around $12,023,
    looking for 1 x 10-minute unboxing, 3 x 60-second YouTube video, 1 x 60-second
    YouTube video. 10 week exclusivity from other similar brands. Campaign runs December
    2025. Usage rights for 10 months. Engagement period: 3 months.'
  sentences:
  - 'Need Scarlett Turner for Pizza Hut campaign. Budget around $13,991, looking for
    2 x Instagram story, 1 x Media interview. 3 week exclusivity from other similar
    brands. Campaign runs September 2025. Usage rights for 10 months. Engagement period:
    3 months.'
  - 'Hey team, looking to work with Morgan Morris for Supercheap Auto. Budget is approx
    $12,633 - need 3 x 10-minute unboxing, 2 x 15-second TikTok, 3 x Instagram story,
    1 x 60-second YouTube video, 2 x Behind-the-scenes content. 9 weeks exclusivity,
    3 month engagement. Usage: 9 months. Start date: January 2026.'
  - brand|campaign|client|deliverables|engagement_term|exclusivity_period|fee|usage_term
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-mpnet-base-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) <!-- at revision 12e86a3c702fc3c50205a8db88f0ec7c0b6b94a0 -->
- **Maximum Sequence Length:** 384 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 384, 'do_lower_case': False}) with Transformer model: MPNetModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Need Zoe Wright for Supercheap Auto campaign. Budget around $12,023, looking for 1 x 10-minute unboxing, 3 x 60-second YouTube video, 1 x 60-second YouTube video. 10 week exclusivity from other similar brands. Campaign runs December 2025. Usage rights for 10 months. Engagement period: 3 months.',
    'Hey team, looking to work with Morgan Morris for Supercheap Auto. Budget is approx $12,633 - need 3 x 10-minute unboxing, 2 x 15-second TikTok, 3 x Instagram story, 1 x 60-second YouTube video, 2 x Behind-the-scenes content. 9 weeks exclusivity, 3 month engagement. Usage: 9 months. Start date: January 2026.',
    'Need Scarlett Turner for Pizza Hut campaign. Budget around $13,991, looking for 2 x Instagram story, 1 x Media interview. 3 week exclusivity from other similar brands. Campaign runs September 2025. Usage rights for 10 months. Engagement period: 3 months.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 2,688 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                         | sentence_1                                                                         | label                                                          |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             | float                                                          |
  | details | <ul><li>min: 38 tokens</li><li>mean: 65.94 tokens</li><li>max: 99 tokens</li></ul> | <ul><li>min: 19 tokens</li><li>mean: 54.51 tokens</li><li>max: 99 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.33</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                              | sentence_1                                                                                                                                                                                                                                                  | label            |
  |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Brooklyn Edwards for Bed Bath N' Table - $9,175 budget<br>            Need: 1 x Instagram story<br>            11w exclusivity, 3m engagement<br>            Usage rights: 7 months<br>            Start: September 2025<br><br>Additional notes: TBC on exact dates, around $9,175 budget</code> | <code>Hey team, looking to work with Victoria Carter for Bed Bath N' Table. Budget is approx $27,996 - need 2 x Instagram story. 10 weeks exclusivity, 3 month engagement. Usage: 7 months. Start date: January 2026.</code>                                | <code>0.0</code> |
  | <code>Aria Perez for Repco - $2,077 budget<br>            Need: 3 x Product photography, 1 x Pinterest pin, 1 x Product photography<br>            4w exclusivity, 3m engagement<br>            Usage rights: 8 months<br>            Start: September 2025 fair dinkum</code>                          | <code>Hey team, looking to work with Aria Perez for Priceline. Budget is approx $11,625 - need 1 x Facebook post. 12 weeks exclusivity, 3 month engagement. Usage: 11 months. Start date: November 2025.</code>                                             | <code>0.0</code> |
  | <code>Need Chloe Lewis for David Jones campaign. Budget around $7,283, looking for 2 x Twitter post. 4 week exclusivity from other fashion brands. Campaign runs December 2025. Usage rights for 7 months. Engagement period: 5 months.</code>                                                          | <code>Hey team, looking to work with Chloe Lewis for Country Road. Budget is approx $23,248 - need 2 x Twitter post, 2 x 5-minute review, 2 x Meet and greet. 10 weeks exclusivity, 2 month engagement. Usage: 6 months. Start date: September 2025.</code> | <code>0.0</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 64
- `num_train_epochs`: 4
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 64
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 4
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `tp_size`: 0
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch | Step |
|:-----:|:----:|
| 1.0   | 42   |
| 2.0   | 84   |


### Framework Versions
- Python: 3.12.3
- Sentence Transformers: 4.1.0
- Transformers: 4.51.3
- PyTorch: 2.7.1+cpu
- Accelerate: 1.10.0
- Datasets: 3.6.0
- Tokenizers: 0.21.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->