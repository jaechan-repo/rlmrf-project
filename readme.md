# Retrieval-Augmented Multimodal Preference Alignment (RAMPA)

![img](assets/images/art1.png)

We use the perplexity of a multimodal LLM augmented by text/image retrieval systems as a reward signal to fine-tune LLaMA 7B. Our aim is to improve the accuracy of content generated by LLaMA 7B without sacrificing the naturalness and coherence that tool-augmented LLMs tend to lack.


```
python generate_passage_embeddings.py --model_name_or_path /mmfs1/gscratch/ark/chan0369/rampa-project/models/contriever --output_dir /mmfs1/gscratch/ark/chan0369/rampa-project/data/contriever_msmarco/wikipedia_embeddings --passages /mmfs1/gscratch/ark/chan0369/rampa-project/data/psgs_w100.tsv --shard_id 0 --num_shards 1
```

