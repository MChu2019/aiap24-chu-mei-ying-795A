# AIAP Technical Assessment — Decision Log

**Candidate name (as in NRIC):**
CHU MEI YING

**Email (as used in your application):**
mychuy2k@gmail.com

---

## A note on this document

This decision log is the primary instrument by which we understand your thinking. The questions below cover the reasoning behind your work — from how you define the problem at the start to the decisions you made during the work itself. Do answer all five questions in your own words. 

You may use AI assistance freely on the technical deliverables (the EDA and the ML pipeline), but this Decision Log itself should be written by you — it is the record of your own thinking that we cross-check against your chat history.


---

## 1. Clarifying questions

What questions would you ask to better define and narrow the problem statement? For each question, briefly explain how the answer would meaningfully change your approach. 

Note: If it helps your decision-making, you may assume and list out the stakeholders' likely answers.

**Your answer:**
Q1: How should we handle deliveries where the customer did not leave a rating?
Explanation: About 65% of the records had no customer rating. I wasn’t sure whether to treat them as good deliveries or remove them. I decided to only use deliveries with actual ratings so the model learns from real feedback.

Q2: What is more important for the operations team — catching as many problem deliveries as possible, or making sure the alerts they get are mostly accurate?
Explanation: This affects how we tune the model. I assumed the business wants to be proactive, so I focused more on finding problem cases even if it means some extra false alarms.

Q3: Are real-time features like traffic or weather data available?
Explanation: Knowing this helps decide whether the model can be real-time or should only use information available before the driver leaves. I assumed we only have historical operational data.

---

## 2. Defining the Problem Statement

Restate, in your own words, the refined problem you decided to solve. List your key assumptions. Briefly note what other framings you considered, and what you deliberately left out or scoped down, and why.

**Your answer:**
Refined Problem: I built a configurable machine learning pipeline that reads delivery data from the SQLite database and predicts which deliveries are likely to have issues (late or low customer rating). The goal is to help the dispatch team take action before the driver starts the trip.
Key Assumptions:

We can reliably join delivery records with customer feedback.
The features available at dispatch time are enough to make useful predictions.
Missing ratings should not be guessed because it could introduce wrong information.

What was scoped down/left out:
I did not use NLP on the customer comments because they are written after the delivery is completed. I also excluded deliveries without ratings from training to keep the target variable clean. No external real-time data was added.

---

## 3. Key decisions during Solution Development

Walk through three key decisions you made during Solution Development. For each: what options did you consider, what did you choose, and why? These can be technical (modelling choices, feature handling, evaluation metrics) or about the work itself (what to prioritise, what to drop, how to spend your time).

**Your answer:**
Decision 1: Choice of Models
I considered using ready-made models from scikit-learn or building simpler models myself. In the end, I went with custom logistic regression and ridge regression using NumPy. This was safer because I wanted to make sure the code runs smoothly even if some libraries are not available in the assessment environment.

Decision 2: Feature Handling
I had a choice between using the columns as they are or doing some feature engineering. I decided to clean the categorical columns and create a few new features like trip duration and time-based information. This was based on what I saw during EDA — delays and certain patterns were clearly linked to poor outcomes.

Decision 3: How to Evaluate the Model
I could have just looked at accuracy, but I chose to use multiple metrics (F1 score, ROC AUC for classification, and MAE/RMSE for regression). This gives a better picture, especially since there are many more good deliveries than bad ones.

---

## 4. Use of the AI assistant

Where did you use the AI assistant in this work? Give three specific examples of something the assistant suggested that you changed, rejected, or significantly modified — and explain your reasoning.

**Your answer:**
I used the AI assistant (mainly Codex and Grok) to help me build the full script-based pipeline. However, I reviewed and changed several things to better match the requirements.
Example 1 : Data Loading
The AI suggested using pd.read_csv(). I rejected it because the task clearly requires using SQLite to load delivery.db.
Example 2 : Pipeline Format
The AI proposed developing everything inside a Jupyter notebook. I rejected this because the assignment specifically says not to develop the ML pipeline in an interactive notebook.
Example 3 : Saving the Model
The AI suggested using pickle to save the trained model. I modified this and only added documentation in the README instead of actual model saving code.
---

## 5. Next Steps

If you had one more week to continue this project, what would you do next, and why? What signals from your current work make those the right next steps?

**Your answer:**
If I had one more week, I would:

Improve the way I handle missing ratings to make use of more data.
Try more configurations in the model to get better results.
Add clearer charts and simpler explanations in the README for non-technical people.
Look deeper into the geographic differences (like branch performance) to see if they can improve predictions.
Clean up small issues like duplicate records that I noticed during EDA.




---