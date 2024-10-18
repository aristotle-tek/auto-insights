# AI Research Insights Generator

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python: 3.x](https://img.shields.io/badge/Python-3.x-blue)

## Objective

This project explores how large language models (LLMs) can assist in human-centered research. Rather than performing end-to-end research autonomously, the AI complements human expertise by leveraging its broad knowledge base, while leaving deep reasoning and decision-making to the researcher.

Thus, for example, in trying to think through how AI might transform society, rather than just relying on the historical cases that a human happens to be familiar with or has in mind, they could turn to this tool to:
1. have an LLM brainstorm possible other (e.g. historical) examples, analogical realms of scientific inquiry that are adjacent, etc.
2. pull and summarize scientific articles on those examples.
3. the researcher can then peruse the LLM suggestions and decide which, if any, to pursue.


- For more information on the thinking behind the project, see a draft explanation and the original roadmap in `/whitepaper_draft/`. See also [a discussion here](https://elenchos.substack.com/)

- Sample outputs can be viewed in the `sample_outputs` folder.



## Status

This repository is no longer actively maintained and is primarily for archival purposes. It may be useful as a reference point for others developing related projects or thinking about how LLMs reason about scientific research.

## Unresolved issues

1. The sources pulled from Semantic Scholar are often not relevant, causing the process to go off-track at times. Possible solutions include (a) adding an additional step to verify relevance or pull a larger set of sources then re-rank them as in RAG setups, and (b) qualifying sources based on journal quality and/ or citation counts.
    
2. Occasionally, issues arise when compiling LaTeX files.

## Getting started

0. Users will need an OPENAI_API_KEY and Semantic Scholar key "S2_API_KEY". Technically Semantic scholar can work without one, but it often fails and so is not reliable. Compiling the .tex file would require a TeX installation.
1. Modify the `config.json` file with your research question, etc.
2. run `python autoinsights.py config.json`

- NB: The model versions should be updated to "gpt-4o" and "gpt-4o-mini" in the `autoinsights.py` file but the original versions used for generating the current outputs are preserved in the repository.

## License
This project is licensed under the [MIT License](./LICENSE).

## Acknowledgements
This project incorporates a code from Eimen Hamedat (in `paper_loaders.py`), which is licensed under the MIT License. 
The original code can be found [here](https://github.com/eimenhmdt/autoresearcher/blob/main/autoresearcher/data_sources/web_apis/semantic_scholar_loader.py)