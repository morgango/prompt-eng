"""
Elasticsearch LLM Cache Library
==================================
This library provides an Elasticsearch-based caching mechanism for Language Model (LLM) responses.
Through the ElasticsearchLLMCache class, it facilitates the creation, querying, and updating
of a cache index to store and retrieve LLM responses based on user prompts.

Key Features:
-------------
- Initialize a cache index with specified or default settings.
- Create the cache index with specified mappings if it does not already exist.
- Query the cache for similar prompts using a k-NN (k-Nearest Neighbors) search.
- Update the 'last_hit_date' field of a document when a cache hit occurs.
- Generate a vector for a given prompt using Elasticsearch's text embedding.
- Add new documents (prompts and responses) to the cache.

Requirements:
-------------
- Elasticsearch
- Python 3.6+
- elasticsearch-py library

Usage Example:
--------------
```python
from elasticsearch import Elasticsearch
from elasticsearch_llm_cache import ElasticsearchLLMCache

# Initialize Elasticsearch client
es_client = Elasticsearch()

# Initialize the ElasticsearchLLMCache instance
llm_cache = ElasticsearchLLMCache(es_client)

# Query the cache
prompt_text = "What is the capital of France?"
query_result = llm_cache.query(prompt_text)

# Add to cache
prompt = "What is the capital of France?"
response = "Paris"
add_result = llm_cache.add(prompt, response)
```

This library is covered in depth in the blog post
Elasticsearch as a GenAI Caching Layer
https://www.elastic.co/search-labs/elasticsearch-as-a-genai-caching-layer

Author: Jeff Vestal
Version: 1.0.0

"""

from datetime import datetime
from typing import Dict, List, Optional
from elasticsearch import Elasticsearch
import logging
from icecream import ic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ElasticsearchLLMCache:
    def __init__(self,
                 es_client: Elasticsearch,
                 index_name: Optional[str] = None,
                 es_model_id: Optional[str] = 'sentence-transformers__all-distilroberta-v1',
                 create_index=True
                 ):
        """
        Initialize the ElasticsearchLLMCache instance.

        :param es_client: Elasticsearch client object.
        :param index_name: Optional name for the index; defaults to 'llm_cache'.
        :param es_model_id: Model ID for text embedding; defaults to 'sentence-transformers__all-distilroberta-v1'.
        :param create_index: Boolean to determine whether to create a new index; defaults to True.
        """
        self.es = es_client
        self.index_name = index_name or 'llm_cache'
        self.es_model_id = es_model_id
        self.dims = 0
        if create_index:
            self.create_index()
        self.msg_prompt_already_exists = 'A similar prompt already exists in the cache.'

    def create_index(self,
                     dims: Optional[int] = 768
                     ) -> Dict:
        """
        Create the index if it does not already exist.

        :return: Dictionary containing information about the index creation.
        """
        if not self.es.indices.exists(index=self.index_name):
            mappings = {
                "mappings": {
                    "properties": {
                        "prompt": {"type": "text"},
                        "response": {"type": "text"},
                        "create_date": {"type": "date"},
                        "last_hit_date": {"type": "date"},
                        "prompt_vector": {"type": "dense_vector",
                                          "dims": dims,
                                          "index": True,
                                          "similarity": "dot_product"
                                          }
                    }
                }
            }

            self.es.indices.create(index=self.index_name, body=mappings, ignore=400)

            self.dims = dims
            ic(f"Index {self.index_name} created with {self.dims} dimensions.")

            return {'cache_index': self.index_name, 'created_new': True}
        else:
            logger.info(f"Index {self.index_name} already exists.")
            return {'cache_index': self.index_name, 'created_new': False}

    def update_last_hit_date(self, doc_id: str):
        """
        Update the 'last_hit_date' field of a document to the current datetime.

        :param doc_id: The ID of the document to update.
        """
        update_body = {
            "doc": {
                "last_hit_date": datetime.now()
            }
        }
        self.es.update(index=self.index_name, id=doc_id, body=update_body)

    def query(self,
              prompt_text: str,
              similarity_threshold: Optional[float] = 0.5,
              num_candidates: Optional[int] = 1000,
              create_date_gte: Optional[str] = "now-1y/y"
              ) -> dict:
        """
        Query the index to find similar prompts and update the `last_hit_date` for that document if a hit is found.

        :param prompt_text: The text of the prompt to find similar entries for.
        :param similarity_threshold: The similarity threshold for filtering results; defaults to 0.5.
        :param num_candidates: The number of candidates to consider; defaults to 1000.
        :param create_date_gte: The date range to consider results; defaults to "now-1y/y".
        :return: A dictionary containing the hits or an empty dictionary if no hits are found.
        """

        knn = [
            {
                "field": "prompt_vector",
                "k": 1,
                "num_candidates": num_candidates,
                "similarity": similarity_threshold,
                "query_vector_builder": {
                    "text_embedding": {
                        "model_id": self.es_model_id,
                        "model_text": prompt_text
                    }
                },
                "filter": {
                    "range": {
                        "create_date": {
                            "gte": create_date_gte

                        }
                    }
                }
            }
        ]

        fields = [
            "prompt",
            "response"
        ]

        resp = self.es.search(index=self.index_name,
                              knn=knn,
                              fields=fields,
                              size=1,
                              source=False
                              )

        if resp['hits']['total']['value'] == 0:
            return {}
        else:
            doc_id = resp['hits']['hits'][0]['_id']
            self.update_last_hit_date(doc_id)
            return resp['hits']['hits'][0]['fields']

    def _generate_vector(self,
                         prompt: str
                         ) -> List[float]:
        """
        Generate a vector for a given prompt using Elasticsearch's text embedding.

        :param prompt: The text prompt to generate a vector for.
        :return: A list of floats representing the vector.
        """
        docs = [
            {
                "text_field": prompt
            }
        ]

        embedding = self.es.ml.infer_trained_model(model_id=self.es_model_id,
                                                   docs=docs
                                                   )

        return embedding['inference_results'][0]['predicted_value']

    def _is_similar_prompt(self, prompt, threshold=0.95, size=1):
        """
        Check if there is an existing prompt that is similar to the new one.

        :param prompt: The text representation of the new prompt.
        :param threshold: The similarity threshold to consider a prompt as a match.
        :return: True if a similar prompt exists, False otherwise.
        """

        prompt_vector = self._generate_vector(prompt=prompt)

        search_result = self.es.search(index=self.index_name, body={
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'prompt_vector') + 1.0",
                        "params": {"query_vector": prompt_vector}
                    }
                }
            },
            "size": size  # We only need the top match
        })

        # Check if the highest scoring document meets the threshold
        if search_result['hits']['hits']:
            for hit in search_result['hits']['hits']:
                score = float((hit['_score'] - 1.0) )
                ic(hit['_score'], hit['_source']['prompt'], score, threshold)
                if score >= threshold:
                    return True
            
        return False

    def list(self) -> Dict:

        query = {
            "query": {
                "match_all": {}
            }
        }

        # Perform the search
        response = self.es.search(index=self.index_name, body=query)

        # Extract the hits
        hits = response['hits']['hits']

        all = {}
        count = 1
        # Print or process the results
        for hit in hits:
            ic(hit['_source']['prompt'])
            all[count] = {'prompt': hit['_source']['prompt'], \
                          'response': hit['_source']['response'], \
                            'create_date': hit['_source']['create_date'], \
                                'last_hit_date': hit['_source']['last_hit_date'], }
            count += 1
        
        return all
        
    def add(self, prompt: str,
            response: str,
            source: Optional[str] = None
            ) -> Dict:
        """
        Add a new document to the index.

        :param prompt: The user prompt.
        :param response: The LLM response.
        :param source: Optional source identifier for the LLM.
        :return: A dictionary indicating the successful caching of the new prompt and response.
        """
        if self._is_similar_prompt(prompt):
            return {'success': False, 'error': self.msg_prompt_already_exists}

        prompt_vector = self._generate_vector(prompt=prompt)

        doc = {
            "prompt": prompt,
            "response": response,
            "create_date": datetime.now(),
            "last_hit_date": datetime.now(),
            "prompt_vector": prompt_vector,
            "source": source  # Optional
        }
        try:
            self.es.index(index=self.index_name, document=doc)
            return {'success': True}
        except Exception as e:
            logger.error(e)
            return {'success': False,
                    'error': e}

    def clear(self):
        self.es.indices.delete(index=self.index_name)
        # filter = ElasticsearchLLMFilter(es_client=es_client, index_name=filter_index_name, es_model_id=model_id, create_index=False)
        # need to make sure we have the right index_name, model_id when we create the index
        ic(self.dims, self.index_name, self.es_model_id)
        self.create_index(dims=self.dims)

    def add_bulk(self, documents: List[Dict]) -> Dict:
        """
        Pre-loads or "warms up" the cache with a list of documents.

        :param documents: A list of dictionaries, each representing a document with 'prompt' and 'response'.
        :return: A dictionary containing the count of successfully added, already cached, and failed documents and
        a list of all the details of each attempt.
        """
        # Initialize counters for successful and already cached additions
        success_count = 0        
        cached_count = 0
        cumulative_results = []

        # Iterate over each document in the provided list
        for doc in documents:
            # Add the document using the 'add' method
            results = self.add(prompt=doc['prompt'], response=doc['response'])
            cumulative_results.append({'prompt': doc['prompt'], 'response': doc['response'], 'results': results})

            # Increment success count if the document is successfully added
            if results['success']:
                success_count += 1
            # Increment cached count if the document is already present
            elif not results['success'] and results['error'] == self.msg_prompt_already_exists:
                cached_count += 1

        # Calculate the number of failed additions
        failed_count = len(documents) - (success_count + cached_count)

        # Return a summary of the operation
        return {'added': success_count, 'already_cached': cached_count, 'failed': failed_count, 'details': cumulative_results}

class ElasticsearchLLMFilter(ElasticsearchLLMCache):

    def __init__(self,
                 es_client: Elasticsearch,
                 index_name: Optional[str] = None,
                 es_model_id: Optional[str] = 'sentence-transformers__all-distilroberta-v1',
                 create_index: bool = True
                 ):
        """
        Initialize the EnhancedElasticsearchLLMCache instance.

        Inherits initialization from ElasticsearchLLMCache and adds any additional initialization.

        :param es_client: Elasticsearch client object.
        :param index_name: Optional name for the index; defaults to 'llm_cache'.
        :param es_model_id: Model ID for text embedding; defaults to 'sentence-transformers__all-distilroberta-v1'.
        :param create_index: Boolean to determine whether to create a new index; defaults to True.
        """
        self.index_name = index_name or 'llm_filter'

        super().__init__(es_client, index_name, es_model_id, create_index)

        self.prompt_off_limits = 'This prompt is off limits and should be filtered out.'

    # # Add any new methods or override existing ones
    # def new_method(self, arg1, arg2):
    #     # Implementation of new method
    #     pass

    def query(self,
              prompt_text: str,
              similarity_threshold: Optional[float] = 0.85,
              num_candidates: Optional[int] = 1000,
              create_date_gte: Optional[str] = "now-1y/y",
              size=1,
              ) -> dict:

        ic(prompt_text)

        candidates = self._is_similar_prompt(prompt=prompt_text, threshold=similarity_threshold, size=size)
        # candidates = super().query(prompt_text=prompt_text, similarity_threshold=similarity_threshold, num_candidates=num_candidates, create_date_gte=create_date_gte) 
        return candidates

