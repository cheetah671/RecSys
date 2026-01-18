# Understanding the Prompt's Core

Based on the `Project_Description.md` file in your workspace, here is an explanation of the project and a breakdown of potential technical approaches.

## Project Overview

You are building a **Personalized Search System** for news articles. Unlike a standard search engine that returns the same results for everyone, your system needs to learn individual user preferences to re-rank results.

The core loop of the project is:

1. **Receive Context:** Get a User ID and a Query from the simulation platform.
2. **Retrieve & Rank:** Search your dataset (`articles.jsonl`) and produce a ranked list of articles.
3. **Simulate:** Send this list to the simulator.
4. **Feedback:** The simulator acts as a "black box" user, returning actions (Clicks, Dwell Time, Likes, Shares).
5. **Learn:** You use this feedback to update your ranking model for that user.

## System Components

- **The Data:** `articles.jsonl` (News articles).
- **The Simulator (Docker):**
  - `GET /query`: Provides the input (User + Query).
  - `POST /ranklist`: Provides the reward signal (User Actions).
- **Your System:** Needs to sit in the middle, indexing articles and serving requests.

---

## Suggested Approaches

Here is a roadmap of approaches ranging from baseline to advanced:

### 1. The Baseline: Elasticsearch (BM25)

As suggested in the description, start by setting up Elasticsearch.

- **Action:** Index `articles.jsonl` into Elasticsearch.
- **Strategy:** For every `/query`, simply run a standard text match (BM25) against the `query_text`.
- **Purpose:** This establishes a "control" group for your experiments. It ignores the `user_id`.

### 2. Feedback Loop & Scoring

You need to quantify "success." The simulator returns complex objects (Clicks, Dwell time). You should convert these into a numerical **Reward Score**.

**Example Metric:**

$$Score = (Click \times 1) + (Like \times 5) + (Share \times 10) + (DwellTime \times 0.1)$$

- **Storage:** Log every interaction: `(user_id, query, article_id, article_features, reward_score)`.

### 3. Content-Based Filtering (Solving for "Topics")

The description states preference depends on topics.

- **User Profiling:** If User A clicks mostly on "Technology" articles, build a vector or frequency map for User A favoring "Technology".
- **Re-ranking:** When User A searches for "Apple", boost articles tagged with "Technology" over articles tagged with "Fruit/Agriculture".

### 4. Finding the "Hidden Feature"

The description mentions one hidden feature drives preference.

- **Exploratory Data Analysis:** Once you have collected some log data, analyze the articles that get high engagement.
- **Hypotheses to test:**
  - Is it **Sentiment**? (Do users prefer positive news?)
  - Is it **Recency**? (Do users only click articles from the last 24h?)
  - Is it **Length**? (Do users prefer short reads vs. long reads?)
  - Is it **Source**? (Do specific users prefer specific publishers?)
- **Implementation:** Extract these features from the text and add them to your ranking model.

### 5. Contextual Bandits (Advanced)

Since the goal is to improve ranking with "as few calls to the simulation platform as possible," this is a classic **Exploration vs. Exploitation** problem.

- **Algorithm:** Use **LinUCB** (Linear Upper Confidence Bound).
- **How it works:** The model predicts the expected reward of an article for a specific user. It adds a "confidence interval." If the model is unsure about an article (high uncertainty), it might rank it higher to "explore" and learn, rather than just "exploiting" what it already knows.

---

## Next Steps

1. Load the docker image as described in `Project_Description.md`.
2. Write a script to ingest `articles.jsonl` into an Elasticsearch instance.
3. Write a Python script to hit `GET /query`, query your Elasticsearch, and send the IDs to `POST /ranklist`.