This is old ugh come back to it!


TODO(Adriano) here are our tasks by noon, lets go go go!
<disk space>
XXX (0.5) as a bonus evaluate everything we already have on code evaluation (that will
    enable us to select the easiest problems from the test/validation dataset and get
    better/really numberic metrics)---that is to say, those top filtered ones
</disk space>
<training 1/4>
XXX (0) launch SAEs on the entire dataset we had already/before
    (while we won't implement a better hyperparameter tuning sweep, we can choose really
    large k's and possibly widths if it fits...; select less loss fns, layers, etc...
    but choose them based on -3's results from above)
</training 1/4>
<dataset augmentatio 1/3>
XXX (1) create the generation pipeline and let it run basically (to generate prompts)
    - collect responses from the qwen models
    - collect responses from the original models 
    (this should yield around 1B tokens; i.e. about 1M exchanges; 33x our 30K seeds)
XXX (2) generate around 30K code knowledge/understanding questions not about this code
    but about software in general (try to source from huggingface if possible) and then
    let the same models generate answers
    - Examples:
        - `Vineeshsuiii/Software_Engineering_interview_datasets`
        - software slacks?
        - etc...
    (this should also yield close to 1B token)
</dataset augmentation 1/3>

<training 2/4>
XXX (3) For our second iteration of training we will try our larger big dataset and run
it with a more optimized hyperparameter sweep over K and width. To do that, we should
define the space of valid stuff and then do a 2-way binary search of some kind...

To achieve this we need to launch and make sure our code-evaluation server works OK.
</training 2/4>

<dataset augmentation 2/3>
XXX (4) collect some code "pretraining" data if possible (this shoudl also be around
    1B tokens or more; ideally we can get it close to around 10B tokens or so)
    - Chunks of code
    - Possibly the dirtier "software slacks" style datasets
    - Possibly larger amounts of chat about code
    - Maybe stack-overflow and other languages, library-specific stuff (is fine), etc...
Basically this should include the features that we need to be able to do short software
help snippets.
</dataset augmentation 2/3>

<training 3/4>
XXX (5) train a model on the dataset we have collected. Do curriculumn leanrning by
(a) training first on the pretraining, (b) training next on the software questions and
more generic, (c) training lastly on our highest quality data (probably the stuff
that specifically comes from the original model and has questions most-like the ones
that we will be working with in the end).

Also trench-up the pretraining data based on lengths, so for example we train up to
context 1024, then 2048, then 4096, then 16384, then finally 32768. (we might combine
the first two...). 
</training 3/4>

<dataset augmentation 3/3>
XXX (5) Generate multi-turn conversations by having an LLM generate a question for the
    response from the datasets above. Simulate conversations (but mostly do it with the
    big models I think). Also do length-traunching here.
</dataset augmentation 3/3>

From each state of the process of training, evaluating, etc... above try to get _some_
evaluation on both supervised loss and code evaluation. Also, generate a few sampled
answers, delete the ones that take up too much space, 

<mechanistic interpretability>
XXX (6) Try asking about non-code and see what happens; collect in original model too
XXX (7) Try jailbreaking and see what happens; collect in original model too
XXX (8) Find a good SAE and a bad SAE
XXX (9) Try some hypotheses for 6, 7, 8 to understand why there is or is not a difference
    (for example, are OOD sent to the kernel? are the good SAEs more attuned to text
    (can look at loss on those tokens specifically) etc...)
</mechanistic interpretability>

# Data generation pipeline description
Every data generation step should basically be an Iterator that takes in the samples
from the previous step and modifies them or adds to them (etc...) in some way. This
shoudl be done by classes with __call__ being the method IMO. We will figure this out
more next but basicaly I think you should init, prepare, and then stream. There should
be a base class.

The generation of data is done by sampling in the following (hierarchical/chained) way:
1. For the different components of the DatasetEntry, randomly select some of them (with
    a probability distribution defined). Each of the components has a rendering function
    basically.
2. Sample some "instructions" which varies (i.e. describe this code, implement this
    code, refactor this code, etc...) based on the prior results (basically you can
    only refactor if the `answers` are provided, etc...)
3. Sample a template to combine things into (i.e. markdown, with separators, xml, etc...)
4. Sample an AI-generated modification/diff (change the theme, change it to be asking
    for a specific thing, etc...)
5. Add typos randomly. Possibly also clip random text, etc... (idk, tbd)
6. Yield the results from the above process

(1) handles no vs. tiny vs. partial vs. full answers and the like
(2) handles the type of task
(3) handles the formatting on a high level
(4) gives the option to reformat in interesting ways and gain more variety, diversity,
    etc... (i.e. primarilly for _semantic_ post-hoc diffs/modifications)
(5) handles typos and other spurious modifications, mistakes, etc... (i.e. _lexical_
    post-hoc diffs/modifications)