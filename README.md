# __News-based Sentiment Analysis with GDELT__

## __Introduction__

In this project, we explore the feasibility of using news articles to predict/interpolate the relationship (friendliness) between important geo-political entities, such as big companies, politicians, and military representatives. We hope to analyze and, ideally, forecast the trend of socioeconomic conflicts centered around these entities. For instance, we ask questions like _"Is the relationship between Shell and Nigeria governors worsening?"_ or _"How many 'hidden parties' exist within these politicians?"_.

In doing so, we first scrape all relevant news articles using the URLs from the [GDELT Project](https://www.gdeltproject.org/) database. Next, we tokenize the articles into sentences, and we detect the entity co-mentions within each sentence. Whenever there is a co-mention detected within a sentence, e.g., _"A and B failed to resolve their disputes across a wide range of issue areas."_, we calculate a __Sentiment Score__ based on the [Goldstein Conflict Score](http://web.pdx.edu/~kinsella/jgscale.html) by detecting the event(s) mentioned in that sentence. We use these sentiment scores as a proxy for the friendliness between the interested geo-political players. Finally, we construct a relationship graph using the co-mention edges, together with tonality scores, to perform analysis, e.g., graph clustering, and visualizations.

To sum, in this repo, we archived the code snippets for,

- [x] News scraping given URLs, text cleaning, and sentence tokenization
- [ ] Entity mention detection (workable, but in development)
- [ ] Co-reference resolution (experimenting)
- [x] Event detection and scoring (experimenting with advanced event extraction features)
- [x] Graph clustering and visualization

## __Environment Setup__

__CONDA IS REQUIRED FOR SETUP__. To create the same environment, please run `conda create -f environment.yml` in terminal/command line. The environment file locates in the project home folder. To activate the newly created environment, run `conda activate pennguin`

## __Get Started with Source Code__

For low-level APIs, i.e., scraping, co-mention detection, and event detection & grading, please directly refer to the source code under `$REPO_FOLDER/src`. Detailed description and sample usage are documented in the source file.

For graph clustering and visualization, please refer to `$REPO_FOLDER/examples` and read the Jupyter notebooks.

## __Misc__

- `$REPO_FOLDER/analysis` contains code for past analysis. Each analysis has its own source code folder, data folder, and output folder.
- `$REPO_FOLDER/data` contains global data files shared across the entire project.
