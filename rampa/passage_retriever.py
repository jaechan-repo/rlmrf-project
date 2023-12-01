from .contriever.src.contriever import load_retriever
from .contriever.src.index import Indexer
from .contriever.src.data import load_passages
from .contriever.src.normalize_text import normalize
from .contriever.passage_retrieval import index_encoded_data
from glob import glob
import os
import time
import torch
from typing import List

class ContrieverRetriever():
    # see: contriever/passage_retrieval.py
    def __init__(self, passages: str):
        print("Loading the passages...")
        self.passages = load_passages(passages)
        self.passage_id_map = {x["id"]: x for x in self.passages}
    
    def prepare_model(self, 
                      model_path: str,
                      passages_embeddings: str = None, 
                      save_or_load_index: bool = True, 
                      projection_size: int = 768, 
                      n_subquantizers: int = 0, 
                      n_bits: int = 8,
                      indexing_batch_size: int = 1000000):
        self.index = Indexer(projection_size, n_subquantizers, n_bits)
        input_paths = sorted(glob(passages_embeddings))
        embeddings_dir = os.path.dirname(input_paths[0])
        index_path = os.path.join(embeddings_dir, "index.faiss")
        if save_or_load_index and os.path.exists(index_path):
            self.index.deserialize_from(embeddings_dir)
        else: 
            print(f"Indexing passages from files {input_paths}")
            start_time_indexing = time.time()
            index_encoded_data(self.index, input_paths, indexing_batch_size)
            print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")
            if save_or_load_index:
                self.index.serialize(embeddings_dir)
        self.model, self.tokenizer, _ = load_retriever(model_path)
        self.model.eval().cuda()

    def retrieve_doc_ids(self, 
                         queries: List[str],
                         per_gpu_batch_size: int = 64, 
                         question_maxlength: int = 512,
                         n_docs: int = 8):
        
        # Load retriever if not already exists
        if self.model is None:
            raise Exception("Prepare the model first")
        question_embedding = embed_queries(queries, self.model, self.tokenizer, 
                                           per_gpu_batch_size=per_gpu_batch_size,
                                           question_maxlength=question_maxlength)
        top_ids_and_scores = self.index.search_knn(question_embedding, n_docs)
        return top_ids_and_scores
    
    def id_to_text(self, doc_id: str):
        doc = self.passages_id_map[doc_id]
        return {
            'title': doc['title'],
            'text': doc['text']
        }



def embed_queries(queries, model, tokenizer, 
                  lowercase=True, normalize_text=True, 
                  per_gpu_batch_size=64, question_maxlength=512):
    model.eval()
    embeddings, batch_question = [], []
    with torch.no_grad():

        for k, q in enumerate(queries):
            if lowercase:
                q = q.lower()
            if normalize_text:
                q = normalize(q)
            batch_question.append(q)

            if len(batch_question) == per_gpu_batch_size or k == len(queries) - 1:

                encoded_batch = tokenizer.batch_encode_plus(
                    batch_question,
                    return_tensors="pt",
                    max_length=question_maxlength,
                    padding=True,
                    truncation=True,
                )
                encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                output = model(**encoded_batch)
                embeddings.append(output.cpu())

                batch_question = []

    embeddings = torch.cat(embeddings, dim=0)
    print(f"Questions embeddings shape: {embeddings.size()}")

    return embeddings.numpy()