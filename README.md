# CS 4641 Team 6 Proposal

[Go to Midterm Report](midterm-report/MidtermReport.md)

## Introduction & Background 
The anticipated yearly NBA draft is crucial for teams seeking new professional talent [4]. Organizations want championship teams, but franchise players are expensive & scarce. We seek a new approach for rookie player selection based on professional player similarity. Similar analysis has been done, but limited to professional players [1], whereas we will include collegiate players. We will explore this through unsupervised & semi-supervised methods. The 1st data source contains NBA player performance statistics across multiple seasons ([Source](https://www.nbastuffer.com/nba-stats/player/)). The 2nd data source contains an index of college player stats ([Source](https://www.sports-reference.com/cbb/players/)). The last data source contains collegiate stats for drafted players ([Source](https://www.basketball-reference.com/draft/)). 

## Problem Definition
### Problem & Motivation
Draft decisions have major financial and strategic reprecussions for NBA franchises. A good draft pick can completely change the season trajectory for a team, potentially resulting in increased viewership and greater revenue opportunities through ticket sales, sponsorships, and merchandise. Traditional scouting methods and draft predictions typically rely on subjective assessments that may not fully capture a player's long term potential or compatability with a given team. While existing approaches work to cluster NBA players together, there is limited work attempting to bridge the gap between the collegiate level and professional level using data-driven methods. This lack of structured methods for talent identification limits a franchise's ability to make informed and objective decisions during the NBA Draft. The objective of our project is to enable organizations to identify draft prospects who most closely resemble the statistical profiles of desirable NBA players so they can make the best possible decisions in the NBA Draft.

## Methods 
### Data Pre-Processing Methods
Data pre-processing will include **feature scaling**, **dimensionality reduction** [2], **handling null values**, **data aggregation**, & **predicting and imputing missing numerical values**.

### Machine Learning Algorithms and Methods 
We will explore **hierarchial clustering** & its use for determining the # of cluster centroids for K-means [3]. **K-Means** (`sklearn.cluster.KMeans`) may allow us to find clusters that could identify different player types including Big Men, Superstars, Benchwarmers, & more [2]. **DBSCAN** (`sklearn.cluster.DBSCAN`) may prove to be a strong alternative if the strongest features have a non-spherical shape. **Cosine similarity** (`sklearn.metrics.pairwise.cosine_similarity`) would allow us to determine player similarity based on the strongest "n" feature, and we may use different pairs as we alter the # of features used.

## (Potential) Results & Discussion 
### Quantitative Metrics 

We will use various evaluatory metrics, starting with the elbow method for finding an optimal K in K-means. For all our unsupervised algorithms, we will use various methods for determining the quality of our clusters including the **Silhouette Score** and **Davies-Bouldin Index**. Additionally, we will employ a supervised method by creating our own clusters of players who are peceived to be similar or different and using the **Adjusted Rand Index** to compare that to the results of our approaches [5].



#### Ethical consideration: 
One potential bias could be toward only Division I athletes, which may lead to the exclusion of talented players from lower divisions. To prevent this, our model will be designed to consider athletes across all collegiate levels when identifying NBA-like profiles.

#### Sustainability
Our plan for a sustainable model is to update with new player data using webscraper. Additionally, because we  work with structured tabular data, the computational requirements will remain low, making ongoing maintenance and retraining of the model resource-friendly.


### Expected Results 
We expect dense intraclusters & spread out interclusters. We will determine player similarity pairs for athletes entering the 2025 draft. We also expect that our clusters will somewhat reflect common understanding of what players in the NBA are similar and different, and we expect that NBA players are in some cases clustered closely to NCAA records of themselves.


## References 
[1] A. Chun, “Using K-Means Clustering to Identify NBA Player Similarity,” Medium, Oct. 30, 2023. [https://medium.com/@allenmchun/using-k-means-clustering-to-identify-nba-player-similarity-2b33f11e3aa7](https://medium.com/@allenmchun/using-k-means-clustering-to-identify-nba-player-similarity-2b33f11e3aa7)

[2] A. Smith, “Clustering NBA Players Based on Statistics — an intro into Unsupervised Machine Learning,” Medium, May 28, 2020. [https://medium.com/@aaroncolesmith/clustering-nba-players-based-on-statistics-an-intro-into-unsupervised-machine-learning-597ba8ea795a](https://medium.com/@aaroncolesmith/clustering-nba-players-based-on-statistics-an-intro-into-unsupervised-machine-learning-597ba8ea795a)

[3] A. Gascón, “PCA and Clustering in Sports Analytics - GoPenAI,” Medium, Jan. 06, 2025. [https://medium.com/gopenai/pca-and-clustering-in-sports-analytics-4e7d92972f5c](https://medium.com/gopenai/pca-and-clustering-in-sports-analytics-4e7d92972f5c).

[4] S. Mir, “NBA Draft Analysis: Using Machine Learning to Project NBA Success” Towards Data Science, Feb. 07, 2022. [https://towardsdatascience.com/nba-draft-analysis-using-machine-learning-to-project-nba-success-a1c6bf576d19/](https://towardsdatascience.com/nba-draft-analysis-using-machine-learning-to-project-nba-success-a1c6bf576d19/)

[5] “Adjusted Rand Index (ARI),” OECD.AI, 2023. [https://oecd.ai/en/catalogue/metrics/adjusted-rand-index-ari](https://oecd.ai/en/catalogue/metrics/adjusted-rand-index-ari). 


## Project Award Eligibility 
We would like to be considered for the "Outstanding Project" award.

## Gantt Chart 
[Gantt Chart](https://gtvault-my.sharepoint.com/:x:/g/personal/icoriolan3_gatech_edu/EYpQbtKJPKFKvGStyM0aYYQBilzmHhuhUHnsH97ha5lsLA?e=8GEGr2)

## Contributions Table

| Name | Proposal Contributions | 
| :--- | :--- |
| Harmony Nagle | Problem & Motivation, Video Presentation, Presentation Slides, Video Presentation Editing |
| William Silva | Potential Results and Discussion |
| Isaiah Coriolan | Gantt Chart, Github Pages Setup, Video Presentation, Introduction/Background, Methods | 
| Philip Wu | Video Presentation, Contributions Table, Github Pages | 
| Shuhan Lin | Expected Results and Discussion | 

## Youtube Unlisted Video Link
[Proposal Video Presentation](https://www.youtube.com/watch?v=QEr-rTR48TY&ab_channel=HarmonyNagle)


