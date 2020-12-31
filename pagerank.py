import os
import random as r
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    transition_model = {}

    # retrving the number of files in the corpus for the random probability  
    file_num         = len(corpus)

    # retrieving the number of links in the corpus for the specific probability 
    link_num         = len(corpus[page])

    if link_num != 0:
        random_prbability    = (1 - damping_factor) / file_num
        specific_probability = damping_factor / link_num

    else:
        random_prbability    = (1 - damping_factor) / file_num
        specific_probability = 0

    for file in corpus:
        # see if the current page has any links 
        if len(corpus[page]) == 0:
            transition_model[file] = 1 / file_num

        else:
            transition_model[file] = specific_probability + random_prbability


    return transition_model



    ##raise NotImplementedError


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    sample_pagerank = {}
    for page in corpus:
        sample_pagerank[page] = 0

    # the intial sample starts off as None 
    sample = None

    # iterate over the pages to preform the sampling process
    for i in range(n):
        # for the first iteration case the sample is None 
        if sample == None:
            # create a lsit of choices 
            choices = list(corpus.keys())
            # pick a sample at random 
            sample = r.choice(choices)
            sample_pagerank[sample] += 1

        else:
            #find the probability distribution from the current sample
            next_sample_probability = transition_model(corpus, sample, damping_factor)
            choices                 = list(next_sample_probability.keys())
            # finding the weights for the choices based on the transition model 
            w                       = [next_sample_probability[i] for i in choices]
            # retreving a new sample 
            sample                  = r.choices(choices, w).pop()
            sample_pagerank[page]   += 1 

    sample_pagerank = {key: value/n for key, value in sample_pagerank.items()}

    return sample_pagerank

    #raise NotImplementedError


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # create a empty dictionary to fill in later in the function 
    iterate_pagerank = {}

    # initially assign 1/(number of pages) as the pagerank value for each page
    for page in corpus:
        iterate_pagerank[page] = 1/(len(corpus))

    changes    = 1
    iterations = 1
    # iterate through the pages and adjust the pagerank until it converges to a value 
    while changes > 0.001:
        changes = 0
        # store the old pagerank values to determin the new 'changes' value
        prior_state = iterate_pagerank.copy()

        for page in iterate_pagerank:
            # find the parent of the current page 
            parents = [link for link in corpus if page in corpus[link]]
            # the page rank equation has two parts the first is a static equation 
            first_half = ((1 - damping_factor) / len(corpus))
            # the second part of the equation involves looking at the parent pages 
            second_half = []
            if len(parents) != 0:
                for parent in parents:
                    value    = prior_state[parent] / len(corpus[parent])
                    second_half.append(value)

            second_half = sum(second_half)
            iterate_pagerank[page] = first_half + (damping_factor * second_half)
            # find the new change value after each time through the loop 
            new_changes = abs(iterate_pagerank[page] - prior_state[page])

            # updating the changes value
            if changes > new_changes:
                changes = new_changes

        iterations += 1

    # normilization of the pagerank values 
    iterate_pagerank_sum = sum(iterate_pagerank.values())
    iterate_pagerank     = {key: value / iterate_pagerank_sum for key, value in iterate_pagerank.items()}


    # print the page ranks 
    print(f'Iteration Number: {iterations}\n')
    print(f'iteration_pagerank values: {round(sum(iterate_pagerank.values()), 10)}')

    return iterate_pagerank
    #raise NotImplementedError


if __name__ == "__main__":
    main()
