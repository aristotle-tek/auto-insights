
import os
from typing import List
import pandas as pd
import numpy as np
import re
from termcolor import colored
import pickle
import json

from paper_loaders import SemanticScholarLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain


from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)



def generate_keyword_combinations(research_question, model='gpt-3.5-turbo'):
        # prompt = keyword_combination_prompt.format(research_question=research_question)
        # response = openai_call(prompt, use_gpt4=False, temperature=0, max_tokens=200)
        # combinations = response.split("\n")
        # # Extract keyword combinations and handle cases where there's no colon
        # return [combination.split(": ")[1] for combination in combinations if ": " in combination]
    chat = ChatOpenAI(temperature=0, model=model)
    sys_prompt=PromptTemplate(
        template="""`reset`
        `no quotes`
        `no explanations`
        `no prompt`
        `no self-reference`
        `no apologies`
        `no filler`
        `just answer`{task_description}
    """,
        input_variables=["task_description"],
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt=sys_prompt)

    extract_human_template="""Generate several keyword combinations based on the following research question:\n{research_question}\nThe output should be structured like this:\nWrite "KeywordCombination:" and then list the keywords like so "Keyword,Keyword,Keyword" """
    human_message_prompt = HumanMessagePromptTemplate.from_template(extract_human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    res = chat(chat_prompt.format_prompt(task_description="", research_question=research_question).to_messages())
    combinations = res.content.split("\n")
    return [combination.split(": ")[1] for combination in combinations if ": " in combination]


def get_citation_by_doi(doi):
    url = f"https://api.citeas.org/product/{doi}?email={EMAIL}"
    response = requests.get(url)
    try:
        data = response.json()
        return data["citations"][0]["citation"]
    except ValueError:
        return response.text
    

def analogy_generator(context_prompt, question_text, model='gpt-3.5-turbo', temperature=0):
    chat = ChatOpenAI(temperature=temperature, model=model)
    sys_prompt=PromptTemplate(
        template="""You are an expert, advanced scientific analogy generator, who when presented with a scientific question or research program suggests a variety of different areas of research that might help clarify and suggest ways to go about finding solutions to this problem. {context_prompt}""",
        input_variables=["context_prompt"],
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt=sys_prompt)

    extract_human_template="""Here is the problem to solve:\n{question_text}\nPlease provide as output ONLY a python list of strings. The list should contain analogous problems or areas of research that might help clarify and suggest ways to go about finding solutions to this problem. Provide a list of insightful suggestions that cover a variety of different appraoches to the problem. Feel free to 'think outside the box', but make sure the suggestions are meaningful and relevant to the problem at hand."""
    human_message_prompt = HumanMessagePromptTemplate.from_template(extract_human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    res = chat(chat_prompt.format_prompt(context_prompt=context_prompt, question_text=question_text).to_messages())
    return res

def research_queries_generator(context_prompt, question_text, sys_template_text, human_template_text, model='gpt-3.5-turbo', temperature=0):
    chat = ChatOpenAI(temperature=temperature, model=model)
    sys_prompt=PromptTemplate(
        template=sys_template_text,
        input_variables=["context_prompt"],
    )
    system_message_prompt = SystemMessagePromptTemplate(prompt=sys_prompt)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template_text)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    res = chat(chat_prompt.format_prompt(context_prompt=context_prompt, question_text=question_text).to_messages())
    return res


def parse_semanticscholar(top_papers):
    rows = []
    paper_list_text = ""
    paper_list_wabstracts = ""
    title_set = set()

    i = 0
    for paper in top_papers:    
        title = paper["title"]
        if title in title_set:
            print(colored(f"Duplicate title: {title}", "red")) # remove duplicates
            continue
        else:
            #print(paper)
            title_set.add(title)
            try:
                pprid = paper["paperId"]
            except:
                pprid = ''
            try:
                abstract = paper["abstract"]
            except:
                abstract = None
            try:
                ncitations = paper["citationCount"]
            except:
                ncitations = np.NaN
            try:
                journalname = paper['journal']['name']
            except:
                journalname = 'unk'
            try:
                authors = paper['authors']
            except:
                authors = 'unk'
            try:
                doi = paper['externalIds']['DOI']
            except:
                doi = np.NaN
            try:
                url = paper['url']
            except:
                url = ''
            paper_list_text += f"{i}. {title} ({journalname})\n" #  {ncitations} citations)\n"
            if abstract is None:
                paper_list_wabstracts += f"{i}. {title} ({journalname})\n"
                rows.append([i, title, journalname, "", authors, doi, url, ncitations])
            else:
                paper_list_wabstracts += f"{i}. {title} ({journalname}) Abstract extract:{abstract}\n"
                rows.append([i, title, journalname, abstract, authors, doi, url, ncitations, pprid])
            i += 1

    df= pd.DataFrame(rows, columns=["index", "title", "journal", "abstract", "authors", "doi", "url", "ncitations",'paperid'])
    return df, paper_list_text, paper_list_wabstracts



def simple_RQ_outline(research_question, output_format="12-part outline", model='gpt-3.5-turbo'):
    """
    Generate just the structure/ outline of the output in hiearchical json format.
    """
    chat = ChatOpenAI(temperature=0, model=model)
    sys_prompt=PromptTemplate(
        template="""`reset`\n`no quotes`\n`no explanations`\n`no prompt`\n`no self-reference`\n`no apologies`\n`no filler`\n`just answer` {task_description}""",
        input_variables=["task_description"],)
    task_description = """Create an outline for a literature review in the form of list with elements separated by a newline like this: "Topic1\nTopic2\nTopic3" etc. """ #a comma separated python list."
    system_message_prompt = SystemMessagePromptTemplate(prompt=sys_prompt)

    extract_human_template="""Create a {output_format} on this topic:\n{research_question}\nYour response should be a list of comma separated values, where the first column is the top category, followed (if applicable) by a second column with a sub-category, etc: `foo, bar, baz` """
    human_message_prompt = HumanMessagePromptTemplate.from_template(extract_human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    res = chat(chat_prompt.format_prompt(task_description=task_description, output_format=output_format, research_question=research_question).to_messages())
    return res.content.split("\n")



def textbased_RQ_outline(research_question, texts, output_format="12-part outline", model='gpt-3.5-turbo'):
    """
    Generate just the structure/ outline of the output in hiearchical json format.\
    Alternative - hierarchical - of comma separated values, where the first column is the top category, followed (if applicable) by a second column with a sub-category, etc: `foo, bar, baz` 
    """
    chat = ChatOpenAI(temperature=0, model=model)
    sys_prompt=PromptTemplate(
        template="""`reset`\n`no quotes`\n`no explanations`\n`no prompt`\n`no self-reference`\n`no apologies`\n`no filler`\n`just answer` {task_description}""",
        input_variables=["task_description"],)
    task_description = """Create an outline for a literature review in the form of list with elements separated by a newline like this: "1. Topic1\n2. Topic2\n3. Topic3" etc. """ #a comma separated python list."
    system_message_prompt = SystemMessagePromptTemplate(prompt=sys_prompt)

    extract_human_template="""Create a {output_format} on this topic:\n{research_question}\nThe outline should be able to incorporate the following content, but does not need to be narrowly constrained by these texts: {texts}\nYour response should be a list separated by newlines like this: "1. Topic1\n2. Topic2\n3. Topic3" etc. """
    human_message_prompt = HumanMessagePromptTemplate.from_template(extract_human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    res = chat(chat_prompt.format_prompt(task_description=task_description, output_format=output_format, research_question=research_question, texts=texts).to_messages())
    return res.content.split("\n")




def update_outline(research_question, texts, curr_outline, output_format="12-part outline", model='gpt-3.5-turbo'):
    """
    Update outline of the output in hiearchical json format based on the texts.
    """
    chat = ChatOpenAI(temperature=0, model=model)
    sys_prompt=PromptTemplate(
        template="""`reset`\n`no quotes`\n`no explanations`\n`no prompt`\n`no self-reference`\n`no apologies`\n`no filler`\n`just answer` {task_description}""",
        input_variables=["task_description"],)
    task_description = """Update the following outline for a literature review in the form of list with elements separated by a newline like this: "Topic1\nTopic2\nTopic3" etc. Do not remove elements. Add to the outline only if necessary; try to avoid additional sections, and add subsections rather than new sections when possible.""" #a comma separated python list."
    system_message_prompt = SystemMessagePromptTemplate(prompt=sys_prompt)

    extract_human_template="""Update if necessary the outline on the following topic:\n{research_question}\nHere is the current outline to update if necessary: {curr_outline}\nThe outline should be able to incorporate the following content, but does not need to be narrowly constrained by these texts: {texts}\nYour response should be a list of comma separated values, where the first column is the top category, followed (if applicable) by a second column with a sub-category, etc: `foo, bar, baz` """
    human_message_prompt = HumanMessagePromptTemplate.from_template(extract_human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    res = chat(chat_prompt.format_prompt(task_description=task_description, research_question=research_question, curr_outline=curr_outline, texts=texts).to_messages())
    return res.content.split("\n")


def df_to_text_paperlist(df, abstract_charlen=300):
    paper_list = ""
    for i, row in df.iterrows():
        title = row["title"]
        journalname = row["journal"]
        abstract = row["abstract"]
        abstract = abstract[:abstract_charlen]
        paper_list += f"{i}. {title} ({journalname}) Abstract extract:{abstract}\n"
    return paper_list


def order_texts_chunked_retry(df, research_question, textoutline, maxrows=7):
    # Iterate over the groups to get each chunk
    # Ideally modify this to make it async.
    # GPT unable to provide text order for large lists - keep it to say 7 papers at a time.
    groups = df.groupby(df.index // maxrows)
    dfindx = pd.DataFrame()
    errs = []
    for name, group in groups:
        print("Group: ", name)
        paperlist = df_to_text_paperlist(group)
        indexes = categorize_texts(research_question, paperlist, textoutline, model='gpt-4')
        indx_list = indexes.split(',')
        if len(indx_list) != len(group):
            print("error getting index, retrying with shorter abstracts")
            paperlist2 = df_to_text_paperlist(df, abstract_charlen=100)
            indexes = categorize_texts(research_question, paperlist2, textoutline, model='gpt-4')
            indx_list = indexes.split(',')
            if len(indx_list) != len(group):
                print("error getting index, retrying")
                errs.append((name, len(indx_list), len(group)))
                indx_list = range(len(group)) # continue
        group['nindx'] = [int(x) for x in indx_list]
        dfindx = pd.concat([dfindx, group])
    return dfindx, errs


def gen_draftext(dd, textoutline):
    drafttext = ""
    idx = 1
    for section in textoutline:
        print(section)
        dfrel = dd[dd['nindx'] == idx]
        print(len(dfrel))
        drafttext += f"\n\n[Section {section}]\n\n"
        if len(dfrel) == 0:
            drafttext += ""
        else:
            for i, row in dfrel.iterrows():
                title = row["title"]
                journalname = row["journal"]
                abstract = row["abstract"]
                if len(abstract) == 0:
                    drafttext += "" # (only include if have the abstract text?) f"Paper {i}) {title}\n\n"
                else:
                    drafttext += f"(Paper {i}) {title}\nAbstract:{abstract}\n\n"
        idx += 1
    return drafttext

#max_abstr_len = max([len(x.text) for x in abstracts]) # 1852 - say it was 2k -> use max 12
def categorize_texts(research_question, texts, outline, model='gpt-3.5-turbo'):
    """ 
    Approaches: (a) df/list - ask gpt to return index number for each text. (b) ask gpt to return the texts as a list in the new order.
    Inputs: (1) textdf with text column. 
                (2) enumerated outline with newlines: "1. Intro\n2. Historical Examples\n3. ..."
    Output - same df but with a new column with the section of the outline that each text belongs to.
    
    Omitted:  Do not remove elements. Add to the outline only if necessary; try to avoid additional sections, and add subsections rather than new sections when possible. #a comma separated python list."
    """
    #assert 0< len(textdf) <=12
    #texts_list = textdf.text.tolist()

    chat = ChatOpenAI(temperature=0, model=model)
    sys_prompt=PromptTemplate(
        template="""`reset`\n`no quotes`\n`no explanations`\n`no prompt`\n`no self-reference`\n`no apologies`\n`no filler`\n`just answer` {task_description}""",
        input_variables=["task_description"],)
    task_description = """For each text in the list, identify where to best place it in the outline. Return only a comma-separated list (the same length as the number of texts) with the index of the outline that each text corresponds to. Such as for example "3,2,6" etc."""
    system_message_prompt = SystemMessagePromptTemplate(prompt=sys_prompt)

    extract_human_template="""Organize the texts on the following topic:\n{research_question}\nHere is the outline to reference: {outline}\nHere are the texts to categorize according to the outline. Return just a comma separated list of the index of the outline that each text corresponds to. {texts}"""
    human_message_prompt = HumanMessagePromptTemplate.from_template(extract_human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    res = chat(chat_prompt.format_prompt(task_description=task_description, research_question=research_question, outline=outline, texts=texts).to_messages())
    return res.content # .split("\n")




def write_in_eqn_or_plot(research_question, text, model='gpt-4'):
    """ 
    
    Inputs: lateX text and research question
    Output: 

    """

    chat = ChatOpenAI(temperature=0, model=model)
    sys_prompt=PromptTemplate(
        template=f"""`reset`\n`no quotes`\n`no explanations`\n`no prompt`\n`no self-reference`\n`no apologies`\n`no filler`\n`just answer` {task_description}""",
        input_variables=["task_description"],)
    
    task_description = """You are an expert researcher, with expertise in math, data science, and Python. You will be given an extract of text written in LaTex. Your job is to identify the single most useful equation or plot that could be added to the text. Return either (1) Python code to generate one plot with MatPlotLib, Plot.ly or Seaborn, OR (2) LaTeX code for ONE equation to add, starting with the last 20 characters of text you would like to append it to, and enclosed in the usual `\begin{{equation\}}...\end\{{equation\}}`.  """
    system_message_prompt = SystemMessagePromptTemplate(prompt=sys_prompt)

    extract_human_template=f"""This is the overall topic to address:\n{research_question}\nHere is the text you will add an equation or plot to: {text}"""
    human_message_prompt = HumanMessagePromptTemplate.from_template(extract_human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    res = chat(chat_prompt.format_prompt(task_description=task_description, research_question=research_question, text=text).to_messages())
    return res.content 


def gpt_query(problem, answer, model='gpt-4'):
    """
    """
    chat = ChatOpenAI(temperature=0, model=model)
    sys_prompt=PromptTemplate(
        template=f"""`reset`\n`no quotes`\n`no explanations`\n`no prompt`\n`no self-reference`\n`no apologies`\n`no filler`\n`just answer` {task_description}""",
        input_variables=["task_description"],)
    
    task_description = """Provide code to plot a diagram, illustration, or graph that illustrates the problem to be solved and or the solution. """
    system_message_prompt = SystemMessagePromptTemplate(prompt=sys_prompt)

    extract_human_template=f"""Here is the problem to solve:\n{problem}\nHere is the text you will write a plot for: {answer}. plot.js code:\n"""
    human_message_prompt = HumanMessagePromptTemplate.from_template(extract_human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    res = chat(chat_prompt.format_prompt(task_description=task_description, problem=problem, answer=answer).to_messages())
    return res.content 




def textbased_outline(research_question, text_df, outline_list, model='gpt-3.5-turbo'):
    """
    Function to be part of a dataflow to add another column to a df with the section of the outline that each text belongs to.
    Loop over df in chunks of size np.floor(12000/max_len)
    """
    if text_df is None:
        outline = generate_outline(research_question, model=model)
    else:
        len_alltexts = 0
        for text in text_df['text']:
            len_alltexts += len(text)
        #if len_alltexts
        currlen = 0
        texts = ""
        while currlen < 10000:
            text = text_df.sample(1)['text'].iloc[0]
            texts += text + "\n"
            currlen += len(text)
        outline = textbased_RQ_outline(research_question, texts)

    # loop over df, add column to categorize each text according to the row of the outline.
    return outline


def write_litreview_1text(research_question, text, model='gpt-3.5-turbo', temperature=0, verbose=True):
    """ Rewrite lit summary in one pass
    """
    if model=='gpt-3.5-turbo':
        if len(text)>15000:
            print('Warning! only using first 15k chars!')
            text = text[:12000]
    if model=='gpt-4':
        if len(text)>31000:
            print('Warning! only using first 31k chars!')
            text = text[:31000]
    llm = ChatOpenAI(temperature=temperature, model=model) #llm = OpenAI(temperature=0)    
    human_message_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template="""Below is a literature review, or a section thereof, on the following topic: {research_question}. Rewrite the following literature review into a more readible form. Keep each source document reference in the same format - in parentheses with a paper number as in "(Paper 2)" or "(Paper 99)", etc. {text}""",
            input_variables=["research_question","text"],
        )
    )
    chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])
    chain = LLMChain(llm=llm, prompt=chat_prompt_template, verbose=verbose)
    #res = chain.run(text)
    res = chain({"research_question": research_question, "text": text}, return_only_outputs=True)
    return res

def write_teX_litreview_1text(research_question, text, model='gpt-3.5-turbo', temperature=0, verbose=True):
    """ Rewrite lit summary in one pass - LaTeX version
    """
    if model=='gpt-3.5-turbo':
        if len(text)>15000:
            print('Warning! only using first 15k chars!')
            text = text[:12000]
    if model=='gpt-4':
        if len(text)>31000:
            print('Warning! only using first 31k chars!')
            text = text[:31000]
    llm = ChatOpenAI(temperature=temperature, model=model) #llm = OpenAI(temperature=0)    
    human_message_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template="""Below is a literature review, or a section thereof, on the following topic: {research_question}. Rewrite the following literature review into a more readible form and using LaTeX for any formatting. As this is in the middle of a LaTeX document, do not open or close the document. You may include `\section` divisions but keep this to a minimum. Keep each source document reference in the same format - in parentheses with a paper number as in "(Paper 2)" or "(Paper 99)", etc. Here is the draft text to reformulate: {text}""",
            input_variables=["research_question","text"],
        )
    )
    chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])
    chain = LLMChain(llm=llm, prompt=chat_prompt_template, verbose=verbose)
    #res = chain.run(text)
    res = chain({"research_question": research_question, "text": text}, return_only_outputs=True)
    return res


def extract_bibref(bibtex_str):
    """ Extract a bibtex reference from a string:
    '@Article{Moayedi2019TwoNN,\n -> Moayedi2019TwoNN
    """
    match = re.search(r'@.*\{([^,]*),', bibtex_str)
    if match:
        return match.group(1)
    else:
        return None



def update_bib_list(papers, all_biblist, ascii_only=True):
    # checks for duplicates within this list, but later need 2nd check.
    paperids = []
    for source in papers:
        paperid = source['paperId']
        if paperid in paperids:
            print("duplicate paper id!")
            continue
        paperids.append(paperid)
        try:
            citcount = source['citationCount']
        except:
            citcount = np.NaN
        try:
            bibtex_str = source['citationStyles']['bibtex']
            bibref = extract_bibref(bibtex_str)
            if ascii_only:
                bibref = bibref.encode('ascii', 'ignore').decode()
            all_biblist.append([paperid, bibtex_str, bibref, citcount])
        except:
            print("no bibtex!")
    return all_biblist



def compile_unique_bib_string(full_biblist, biblio_string='', ascii_only=False):
    paperids = []
    for bibentry in full_biblist:
        paperid = bibentry[0]
        if paperid in paperids:
            print("duplicate paper id!")
            continue
        try:
            curr_str = bibentry[1]
            if ascii_only:
                curr_str = curr_str.encode('ascii', 'ignore').decode('ascii')
            biblio_string += curr_str + "\n" # .append(bibtex_str)
        except:
            print("no bibtex!")
    return biblio_string



def latex_quotations(text):
    text = re.sub(r'[\s]"([\w])', r' ``\1', text)
    text = re.sub(r'([\w,])"[\s]', r"\1'' ", text) 
    return text


def undefined_control_seqs(text):
    """ \C replaced by \mathbb{C}:"""
    text = re.sub(r'\\C', r'\\mathbb{C}', text)
    text = re.sub(r' R&D ', r'R\&D', text)
    return text


def generate_bibkey_series(pprdf, full_biblist):
    poss_errs = []
    i = 0
    bibkeys = []
    for ref in full_biblist:
        pprid = ref[0]
        print(pprid)
        bib_key = ref[2]
        bibkeys.append(bib_key)
        # pprdf.loc[i, 'bib_key'] = bib_key
        if pprid != pprdf.loc[i, 'paperid']:
            print("possible error in bib key assignment")
            poss_errs.append([pprid, pprdf.loc[i, 'paperid']])
        #print(pprid)
        i += 1

    pprdf['bib_key'] = bibkeys
    bibkey_series= pprdf['bib_key']
    return bibkey_series, poss_errs


def replace_parenthetical_refs(text, bibkey_series):
    def repl(match):
        refnumber = int(match.group(1))  # Extract the reference number
        if 0 <= refnumber < len(bibkey_series):
            bibkey = bibkey_series[refnumber]  # Get the corresponding bibkey
            return r'\cite{%s}' % bibkey  # Return the replacement string
        else:
            return match.group(0)  # Return the original match string
    return re.sub(r'\(Paper (\d+)\)', repl, text)



def generate_topic(topic, projname, log, ctr, paperloader, year_range="2000-2023", paper_source='semanticscholar', weight_sim=0.5, save=True):
    print("generating literature review for: %s" % (topic,) )
    print(colored(f"getting keyword combinations...", "yellow"))
    research_question = topic
    keyword_combinations = generate_keyword_combinations(topic)
    log.append(["keywords", topic, keyword_combinations])
    if paper_source == 'semanticscholar':
        print(colored(f"getting papers from Semantic Scholar...", "blue"))
        top_papers = paperloader.fetch_and_sort_papers(research_question, keyword_combinations=keyword_combinations, year_range=year_range, top_n=20, weight_similarity=weight_sim) #  search_query, limit=100, top_n=20, year_range=None, keyword_combinations=None, weight_similarity=0.5
        log.append(["papers", topic, top_papers])
        try:
            df, paper_list_text, paper_list_wabstracts = parse_semanticscholar(top_papers)
        except:
            df, paper_list_text, paper_list_wabstracts = parse_semanticscholar(top_papers)
    elif paper_source == 'arxiv':
        print("Not currently implemented!!!!!\n\n\n----------\n")#top_papers = 
        sys.exit()
    print("%d generating literature review outline..." % (ctr,))
    output_format= "6-part outline, omit the introduction and conclusion"
    textoutline = textbased_RQ_outline(research_question, paper_list_text, output_format=output_format, model='gpt-4') #'gpt-3.5-turbo'
    log.append(["outline", topic, textoutline])
    print("%d ordering texts..." % (ctr,))
    try:
        dfindx, errs = order_texts_chunked_retry(df, research_question, textoutline, maxrows=7)
        log.append(["ordering", topic, dfindx['nindx']])
    except:
        print(colored(f"error in ordering texts- unable to complete!", "red"))
        return log
    drafttext = gen_draftext(dfindx, textoutline)
    log.append(["drafttext", topic, drafttext])
    print("Editing final draft...")
    revised = write_teX_litreview_1text(research_question, drafttext, model='gpt-4', temperature=0)
    log.append(["revisedtext", topic, revised])
    if save:
        print("%d saving results..." % (ctr,))
        with open('insights/litreview_%s_%d.txt' % (projname, ctr,), 'w') as f:
            f.write(topic + '\n\n')
            f.write(revised['text'] + '\n\n')
        with open('insights/LR_log_%s.pkl' % (projname,), 'wb') as handle:
            pickle.dump(log, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(colored(f"done for this topic!", "green"))
    return log


def compile_text_bib_from_log(log):
    full_biblist = []
    maintext = ''
    for lg in log:
        whtype = lg[0]
        if whtype == 'papers':
            papers = lg[2]
            pprdf, paper_list_text, paper_list_wabstracts = parse_semanticscholar(papers)
            curr_biblist = update_bib_list(papers, [])
            full_biblist += curr_biblist
            curr_bibanalogy = lg[1]
            if len(curr_biblist) == len(pprdf):
                bibkey_series, _ = generate_bibkey_series(pprdf, curr_biblist)
            else:
                bibkey_series = matchbased_gen_bibkey_series(pprdf, curr_biblist)
        elif whtype == 'revisedtext':
            text = re.sub(r'\\section', r'\\subsection', lg[2]['text'])
            text = latex_quotations(text)
            text = undefined_control_seqs(text)
            curr_analogy = lg[1]
            tex_text = "\\section{" + curr_analogy + "}\n\n" + text + "\n\n\n"
            assert curr_bibanalogy == curr_analogy
            new_text = replace_parenthetical_refs(tex_text, bibkey_series)
            maintext += new_text
            print("len text: %d" % (len(maintext),))
    return maintext, full_biblist







Rquestion_prompt_template = """Use the following portion of a document to develop a set of reasoning steps to justify the statement below. 
Return any relevant text in list format, separated by newlines ('\n). Here's is the document to use:
{context}
Here is the statement for which you should provide the reasoning steps: {question}
Reasoning justification for the statement, in list format:"""
RMAP_PROMPT = PromptTemplate(
    template=Rquestion_prompt_template, input_variables=["context", "question"]
)

Rcombine_prompt_template = """Given the following extracted reasoning steps, extracted from a longer document and a research claim, write a final list of reasoning steps to justify the claim. Do not include duplicate steps, and make sure the logical order of the reasoning steps is correct.

CLAIM TO JUSTIFY: {question}
=========
{summaries}
=========
Answer:"""

RCOMBINE_PROMPT = PromptTemplate(
    template=Rcombine_prompt_template, input_variables=["summaries", "question"]
)

def identify_reasoning_steps(documents, statement, model='gpt-4', max_docs_per_request=5, verbose=True):
    llm = ChatOpenAI(temperature=0, model=model)
    chain = load_summarize_chain(llm, chain_type="map_reduce", return_intermediate_steps=verbose, map_prompt=RMAP_PROMPT, combine_prompt=RCOMBINE_PROMPT)
    return chain({"input_documents": documents, "question": statement}, return_only_outputs=verbose, max_docs_per_request=max_docs_per_request)


