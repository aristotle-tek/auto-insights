
# this is based on a (2023 version of) code from Eimen Hamedat, which is licensed under the MIT License. 
# This can be found here https://github.com/eimenhmdt/autoresearcher/blob/main/autoresearcher/data_sources/web_apis/semantic_scholar_loader.py

import os
import requests
from abc import ABC, abstractmethod
import wikipedia
import jellyfish

class BaseWebAPIDataLoader(ABC):
    def __init__(self, base_url, api_key=None):
        self.base_url = base_url
        self.api_key = api_key

    @abstractmethod
    def fetch_data(self, search_query, **kwargs):
        pass

    def make_request(self, endpoint, params=None):
        url = f"{self.base_url}{endpoint}"
        headers = {'x-api-key': self.api_key}
        response = requests.get(url, params=params, headers=headers)

        if response.status_code == 200:
            data = response.json()
            return data
        else:
            raise Exception(f"Failed to fetch data from API: {response.status_code}")


class SemanticScholarLoader(BaseWebAPIDataLoader):
    def __init__(self):
        super().__init__("https://api.semanticscholar.org/graph/v1/paper/search", os.getenv('S2_API_KEY'))

    def fetch_data(self, search_query, limit=100, year_range=None):
        params = {
            "query": search_query,
            "limit": limit,
            "fields": "title,url,abstract,authors,citationStyles,journal,citationCount,year,externalIds"
        }

        if year_range is not None:
            params["year"] = year_range

        data = self.make_request("", params=params)
        return data.get('data', [])

    def fetch_and_sort_papers(self, search_query, limit=100, top_n=20, year_range=None, keyword_combinations=None, weight_similarity=0.5):
        papers = []
        if keyword_combinations is None:
            keyword_combinations = [search_query]

        for combination in keyword_combinations:
            papers.extend(self.fetch_data(combination, limit, year_range))

        max_citations = max(papers, key=lambda x: x['citationCount'])['citationCount']
        
        for paper in papers:
            similarity = jellyfish.jaro_similarity(search_query, paper['title'])
            if max_citations == 0:
                normalized_citation_count = 0
            else:
                normalized_citation_count = paper['citationCount'] / max_citations
            paper['combined_score'] = (weight_similarity * similarity) + ((1 - weight_similarity) * normalized_citation_count)

        sorted_papers = sorted(papers, key=lambda x: x['combined_score'], reverse=True)

        return sorted_papers[:top_n]
