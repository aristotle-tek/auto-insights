import os
import shutil

import subprocess
import time


"""
Input: various text components - title, abstract, maintext.
Input: output filepath
Input: template folder (to copy style files from)
Output: compiled pdf with sources.

Example usage:
from compile_teX import create_tex_str, create_pdf
tex_str = create_tex_str(title, abstract, maintext)


Run pdflatex on your .tex file, which will create a .aux file among others.
Run bibtex on the .aux file, which will create a .bbl file with formatted references.
Run pdflatex on your .tex file again, which will incorporate the .bbl file into the document.
Run pdflatex on your .tex file one more time, to resolve any remaining references.

# arxiv style file from https://github.com/kourgeorge/arxiv-style
"""

template_folder = ""



def compile_no_bilbiography(input_tex, filename, output_folder, template_folder=template_folder):
    if not os.path.exists(output_folder):
        print("Creating output folder: ", output_folder)
        os.makedirs(output_folder)
    
    # copy style file  template_folder + arxiv.sty to output_folder
    shutil.copy(template_folder + "arxiv.sty", output_folder)

    # save input_tex to output_folder
    texfilepath = output_folder + filename +  ".tex"
    with open(texfilepath, "w") as f:
        f.write(input_tex)

    time.sleep(1) # wait for file to be written to disk
    orig_wd = os.getcwd()
    os.chdir(output_folder)
    process = subprocess.Popen(['pdflatex', texfilepath])
    process.communicate()

    process = subprocess.Popen(['pdflatex', texfilepath])
    process.communicate()
    os.chdir(orig_wd)



#def replace_non_ascii(s):
#    return ''.join(c if ord(c) < 128 else '?' for c in s)
def replace_non_ascii(curr_str):
    return curr_str.encode('ascii', 'ignore').decode('ascii')




def prep_tex_documents(input_tex, bib_tex, filename, output_folder, ascii_only=False, template_folder=template_folder):
    """Input tex string, bib str, filename,
    copies arxiv.sty from template_folder to output_folder.
    creates tex file in output_folder.
    creates references.bib file in output_folder.
    """
    
    if not os.path.exists(output_folder):
        print("Creating output folder: ", output_folder)
        os.makedirs(output_folder)

    # copy style file  template_folder + arxiv.sty to output_folder
    shutil.copy(template_folder + "arxiv.sty", output_folder)

    # save input_tex to output_folder
    texfilepath = output_folder + filename +  ".tex"
    with open(texfilepath, "w") as f:
        f.write(input_tex)

    if ascii_only:
        bib_tex = replace_non_ascii(bib_tex)
        print("nb: replaced non-ascii characters in references.bib")

    # save references.bib to output_folder
    bibfilepath = output_folder + "references.bib"
    with open(bibfilepath, "w") as f:
        f.write(bib_tex)
    return texfilepath



def compile_pdf(texfilepath, output_folder):
    """Input tex string, bib str, filename,
    copies arxiv.sty from template_folder to output_folder.
    creates tex file in output_folder.
    creates references.bib file in output_folder.
    output pdf."""

    orig_wd = os.getcwd()
    os.chdir(output_folder)
    print("initial pdflatex compile...")
    process = subprocess.Popen(['latex', texfilepath]) # pdflatex
    process.communicate()
    print("bibtex compile...")
    process = subprocess.Popen(['bibtex', texfilepath.replace('.tex', '.aux')])
    process.communicate()
    print("additional pdflatex compiles...")
    process = subprocess.Popen(['latex', texfilepath])
    process.communicate()

    process = subprocess.Popen(['latex', texfilepath])
    process.communicate()
    os.chdir(orig_wd)
    print('done.')




doc_header = r"""\documentclass{article}

\usepackage{arxiv}

\usepackage[utf8]{inputenc}
\usepackage{hyperref}
\usepackage{url}
\usepackage{booktabs}       % professional-quality tables
\usepackage{amsfonts}       % bb math symbols
\usepackage{amsmath}
\usepackage{nicefrac}       % compact symbols for 1/2, etc.
\usepackage{microtype}      % microtypography
\usepackage{cleveref}       % smart cross-referencing
\usepackage{graphicx}
\usepackage{natbib}
\usepackage{doi}
\usepackage{algorithm2e}

\title{"""

post_title = r"""
}
\date{}
\author{\hspace{1mm}GPT-4}% \thanks{} \\
\renewcommand{\undertitle}{AI Assistant Brainstorm}
\hypersetup{
pdftitle={AI Research Assistant},
pdfsubject={AI, scientific research, scientific insights},
pdfauthor={AI Research Assistant},
pdfkeywords={"""

post_keywords = r"""},
}

\begin{document}
\maketitle

\begin{abstract}
"""

post_abstract = r"""
\end{abstract}

"""

doc_footer = r"""
\bibliographystyle{unsrtnat}
\bibliography{references}

%\appendix
%\section{Parameters}

\end{document}
"""

def create_tex_str(title, keywords, abstract, maintext, doc_header=doc_header, post_title=post_title, post_keywords=post_keywords, post_abstract=post_abstract, doc_footer=doc_footer):
    full_latex_doc = doc_header + title + post_title + keywords + post_keywords + abstract + post_abstract  + maintext + doc_footer
    return full_latex_doc


def matchbased_gen_bibkey_series(pprdf, curr_biblist):
    bibkey_dict = {}
    for i in range(len(curr_biblist)):
        currref = curr_biblist[i]
        docid = currref[0]
        curr_bibkey = currref[2]
        try:
            dfindx = pprdf[pprdf['paperid'] == docid].index[0]
        except:
            print(f"No match for docid {docid} in pprdf")
            continue
        #if type(dfindx) == int:
        bibkey_dict[dfindx] = curr_bibkey
    return pd.Series(bibkey_dict)