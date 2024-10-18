# Before running, export 
# 1. OPENAI_API_KEY 
# 2. S2_API_KEY (Semantic Scholar, in theory can work without, but often fails...)

import os
import numpy as np
import json
import re
import pandas as pd
import openai
import pickle
import time
import sys
from datetime import datetime
from termcolor import colored
from pathlib import Path
from llama_index.readers import download_loader

from insight_utils import *
from paper_loaders import SemanticScholarLoader
from compile_TeX import *

verbose = True


#----------------

#----------------


def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def save_config_to_log(config, log_file): # nb - will overwrite existing log file
    if os.path.exists(log_file):
        print("Log file already exists. adding extension...")
        log_file = re.sub('.json', '2.json', log_file)
    with open(log_file, 'w') as f:
        json.dump(config, f, indent=4)


def write_topics_tofile(topics_list, filename):
    with open(filename, 'w') as f:
        f.write(string_to_write)


if __name__=="__main__":
    try:
        config_file = sys.argv[1]
    except:
        print("No config file specified. Using default `config.json` (or ctr+z to abort)...")
        time.sleep(5)
        config_file = 'config.json'
    config = load_config(config_file)
    filename = config['filename']
    start_time = datetime.now()
    formatted_datestr = format(start_time.strftime('%Y-%m-%d_%H-%M'))

    logfile = "logfile_" + filename + "_" +formatted_datestr + '.json'
    save_config_to_log(logfile, 'config.log')
    start_time = datetime.now()
    human_RQ = config['filename']
    context_prompt_alt = config['context_prompt_alt'] # # """Focus on obscure examples that are not well known, perhaps explore examples from other countries and cultures."""
    projname = config['projname']
    start_type = config['start_type']
    sys_template_text = config['sys_template_text']
    human_template_text = config['human_template_text']
    analogies_temp = float(config['analogies_temp'])
    weight_sim = float(config['weight_sim'])
    year_range = config['year_range']
    output_type = config['output_type']
    if config['output_folder'] == "":
        output_folder = Path.cwd()
    else:
        output_folder = config['output_folder']
    paper_source = config['paper_source']
    title_text =  config['title_text']
    abstract_text = "Here is the source research question that served as a prompt: `` " + human_RQ
    SemanticScholar = SemanticScholarLoader()

    if output_type == "latex":
        from compile_teX import *


    if start_type == "analogies":
        topics_text = analogy_generator(context_prompt_alt, human_RQ, model='gpt-4', temperature=analogies_temp)
        print(topics_text.content)
    else:
        topics_text = research_queries_generator(context_prompt, question_text, sys_template_text, human_template_text, model='gpt-3.5-turbo', temperature=0)


    topics_text_content = topics_text.content.strip()
    if topics_text_content.startswith("```python"):
        topics_text_content = topics_text_content.replace("```python", "").replace("```", "").strip()  # Remove markdown code block

    topics_list = eval(topics_text_content)

    for analogy in topics_list:
        print("\\item " + analogy)

    approval = input("Do you approve of the above topics? Enter 'yes' to continue: ").strip().lower()
    if approval != 'yes':
        print("Approval rejected. Aborting...")
        sys.exit()

    log = []
    start = 0
    ctr = start
    for topic in topics_list[start:]:
        log = generate_topic(topic, projname, log, ctr, SemanticScholar, year_range=year_range, weight_sim=weight_sim, save=True)
        ctr += 1

    maintext, full_biblist = compile_text_bib_from_log(log)

    bibstr = compile_unique_bib_string(full_biblist, bibstr)
    # can't do this - might have a table...bibstr = re.sub(r'&', r'\\&', bibstr)
    bibstr = re.sub(r'R&D', r'R\\&D', bibstr)
    bibstr = re.sub(r'M&A', r'M\\&A', bibstr)
    #assert len(curr_biblist) == len(pprdf)

    max_keywords = 5
    keywords_text = ""
    for kw in list(allkeywords)[:max_keywords]:
        keywords_text += kw + ", "

    tex_str = create_tex_str(title_text, keywords_text, abstract_text, maintext)

    texfilepath = prep_tex_documents(tex_str, bibstr, filename, output_folder, ascii_only=True)
    
    time.sleep(1)

    print("Compiling PDF...")
    compile_pdf(texfilepath, output_folder)
    print(colored(f"It's a miracle- no LaTeX errors - done!", "green"))
    end_time = datetime.now()
    time_diff = end_time - start_time
    print('Total time elapsed (hh:mm:ss.ms) {}'.format(time_diff))