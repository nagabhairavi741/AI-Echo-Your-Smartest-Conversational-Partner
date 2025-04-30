import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from collections import Counter
import re
import itertools

# Load trained model and vectorizer
import pickle

# Corrected file paths (use raw strings to handle backslashes properly)
model_path = r"C:\Users\Muthu\best_sentimental_model.pkl"
vectorizer_path = r"C:\Users\Muthu\tf_idf_vectoriser.pkl"

# Load trained model
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Load TF-IDF vectorizer
with open(vectorizer_path, "rb") as file:
    vectorizer = pickle.load(file)


# Streamlit UI
st.title("🧠 AI Echo: Your Smartest Conversational Partner")

# Sidebar Navigation
option = st.sidebar.radio("Choose a section:", ["🏙️ Home","📊 EDA", "📝 Sentiment Prediction","📈 Model Evaluation"])

# =========================================
# 📊 1️⃣ HOME
# =========================================
if option == "🏙️ Home":
    st.markdown("""

### 🔍 Overview

Welcome to **AI Echo**, an intelligent platform designed to **analyze user sentiments** from ChatGPT reviews. Whether it's praise, feedback, or criticism — AI Echo listens, learns, and gives insights that help improve conversational AI experiences.

### 🚀 What It Does
- ✅ **Sentiment Analysis** on ChatGPT user reviews  
- ✅ Detects whether the feedback is **Positive** or **Negative** or **Neutral**
- ✅ Powered by advanced **NLP and Machine Learning algorithms**
- ✅ Helps product teams, researchers, and developers understand user satisfaction at scale

### 💡 Use Case
Got a review? Paste it in. AI Echo will instantly analyze it and tell you what the sentiment behind the text is. Ideal for:
- UX teams monitoring product feedback
- AI researchers exploring human-AI interaction
- Community moderators understanding user pain points
""") 


# =========================================
# 📊 1️⃣ EXPLORATORY DATA ANALYSIS (EDA)
# =========================================
if option == "📊 EDA":
    st.header("🔎 Exploratory Data Analysis (EDA)")

    df = pd.read_csv(r"C:\Users\Muthu\processed_reviews.csv")
        
    if "text" in df.columns and "sentiment" in df.columns:
        st.title("📌 Data Preview:")
        st.dataframe(df.head())

        # Sentiment Distribution
        st.title("📊 Sentiment Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x=df["sentiment"], palette="viridis", ax=ax)
        st.pyplot(fig)

        # sentiments vary by rating
        st.title("📊 Sentiment by Rating")
        fig,ax = plt.subplots(figsize=(8,5))
        sentiment_by_rating = df.groupby(['rating','sentiment']).size().unstack(fill_value=0)
        sentiment_by_rating.plot(kind='barh',colormap='viridis',ax=ax)
        plt.ylabel("Rating")
        plt.xlabel("No of Reviews")
        plt.legend()
        st.pyplot(fig)

        #Which keywords or phrases are most associated with each sentiment class
        st.title("📝WordCloud for Sentiments")
        fig, axes = plt.subplots(3, 1, figsize=(8,5))
        sentiments = df["sentiment"].unique()
        
        for i, sentiment in enumerate(sentiments):
            reviews = df[df["sentiment"]==sentiment]["text"].astype('str')

            # Step 2: Join all reviews into one string
            all_text = " ".join(reviews)

            # Step 3: Remove unwanted characters (quotes, braces, symbols, digits)
            cleaned_text = re.sub(r"[^\w\s]", "", all_text)  # Removes punctuation
            cleaned_text = re.sub(r"\d+", "", cleaned_text)  # Removes numbers

            # Optional: Lowercase
            cleaned_text = cleaned_text.lower()

            # Step 4: Tokenize and count word frequency
            words = cleaned_text.split()

            # Remove unwanted terms (if they appear as keys in a dictionary)
            unwanted_keys = ["sentences", "tokenized_words", "stemmed_words", "lemmatized_words"]
            filtered_words= [word for word in words if word not in unwanted_keys and len(word) > 2]

            filtered_text = " ".join(filtered_words)
            wordcloud = WordCloud(width=500, height=300, background_color="white").generate(filtered_text)
            axes[i].imshow(wordcloud, interpolation="bilinear")
            axes[i].set_title(f"{sentiment} Sentiment")
            axes[i].axis("off")
        
        st.pyplot(fig)

        #How has sentiment changed over time
        st.title("📅 Sentiment Trend Over Time")
        fig,ax = plt.subplots(figsize=(8,5))
        df["date"]=pd.to_datetime(df['date'])
        sentiment_over_time = df.groupby([df['date'].dt.to_period('M'),'sentiment']).size().unstack(fill_value=0)
        sentiment_over_time.index = sentiment_over_time.index.to_timestamp()
        sentiment_over_time.plot(ax=ax,marker='o')
        plt.xlabel('Month')
        plt.ylabel('sentiment')
        plt.legend()
        st.pyplot(fig)

    
        #Do verified users tend to leave more positive or negative reviews?
        st.title("📝 Verified users vs reviews")
        fig,ax = plt.subplots(figsize=(8,5))
        verified_users_review = df.groupby(['verified_purchase','sentiment']).size().unstack(fill_value=0)
        verified_users_review.plot(kind='bar',colormap='viridis',ax=ax)
        plt.ylabel("Review counts")
        st.pyplot(fig)

        #Are longer reviews more likely to be negative or positive?
        st.title("📝 Are Longer Reviews more positive or Negative? ")
        fig,ax = plt.subplots(figsize=(8,5))
        review_length_sentiment = df.groupby(['sentiment'])["review_length"].mean()
        review_length_sentiment.plot(kind='bar',color=["red","gray","green"],ax=ax)
        plt.ylabel("Average Review length")
        plt.xlabel("Sentiment")
        st.pyplot(fig)


        #7.Which locations show the most positive or negative sentiments?
        st.title("🗺️ locations vs sentiments")
        location_wise_sentiments = df.groupby(["location","sentiment"]).size().unstack(fill_value=0)
        fig,ax = plt.subplots(figsize=(8,5))
        location_wise_sentiments.plot(kind='barh',colormap='viridis',ax=ax)
        plt.ylabel("Locations")
        plt.xlabel("Sentiment")
        st.pyplot(fig)


        #8.Is there a difference in sentiment across platforms (Web vs Mobile)?
        st.title("📟 Platforms vs Reviews")
        fig,ax = plt.subplots(figsize=(8,5))
        verified_users_review = df.groupby(['platform','sentiment']).size().unstack(fill_value=0)
        verified_users_review.plot(kind='bar',colormap='coolwarm',ax=ax)
        plt.ylabel("Review counts")
        plt.xlabel("Platform")
        st.pyplot(fig)


        #9.Which ChatGPT versions are associated with higher/lower sentiment?
        st.title("🎯 chatgpt versions vs Reviews")
        fig,ax = plt.subplots(figsize=(8,5))
        verified_users_review = df.groupby(['version','sentiment']).size().unstack(fill_value=0)
        verified_users_review.plot(kind='bar',colormap='viridis',ax=ax)
        plt.ylabel("Review counts")
        plt.xlabel("Platform")
        st.pyplot(fig)


        #10.What are the most common negative feedback themes?

        st.title("🚨 Most Common Negative Feedback Themes")

        #step 1:get negative reviews
        negative_reviews = df[df["sentiment"]=='Negative']["text"].astype('str')

        # Step 2: Join all reviews into one string
        all_text = " ".join(negative_reviews)

        # Step 3: Remove unwanted characters (quotes, braces, symbols, digits)
        cleaned_text = re.sub(r"[^\w\s]", "", all_text)  
        cleaned_text = re.sub(r"\d+", "", cleaned_text)  

        # Optional: Lowercase
        cleaned_text = cleaned_text.lower()

        # Step 4: Tokenize and count word frequency
        words = cleaned_text.split()

        # Remove unwanted terms (if they appear as keys in a dictionary)
        unwanted_keys = ["sentences", "tokenized_words", "stemmed_words", "lemmatized_words"]
        filtered_words= [word for word in words if word not in unwanted_keys and len(word) > 2]

        word_freq = Counter(filtered_words)

        # Generate word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate_from_frequencies(word_freq)

        # Display with matplotlib
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)


            
# =========================================
# 📝 2️⃣ SENTIMENT PREDICTION
# =========================================
elif option == "📝 Sentiment Prediction":
    st.header("📝 Sentiment Prediction")

    user_input = st.text_area("Enter your text:")

    if st.button("Analyze Sentiment"):
        if user_input:
            input_vectorized = vectorizer.transform([user_input])
            prediction = model.predict(input_vectorized)[0]
            prediction_prob = model.predict_proba(input_vectorized)

            st.write(f"### 🎯 Predicted Sentiment: **{prediction}**")
            labels = ["Negative", "Neutral", "Positive"]
            prob_df = pd.DataFrame(prediction_prob, columns=labels)
            st.dataframe(prob_df)
        else:
            st.warning("⚠️ Please enter text before predicting!")

# =========================================
# 📈 3️⃣ MODEL EVALUATION
# =========================================
elif option == "📈 Model Evaluation":
    st.header("📈 Model Performance Evaluation")
    
    # Load test data
    test_df = pd.read_csv(r"C:\Users\Muthu\processed_reviews.csv")  # Adjust path
    if "text" in test_df.columns and "sentiment" in test_df.columns:
        X_test = vectorizer.transform(test_df["text"])
        y_test = test_df["sentiment"]

        # Label encoding for metrics if necessary
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_test)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # Classification Report
        st.subheader("📋 Classification Report")
        report = classification_report(y_encoded, le.transform(y_pred), target_names=le.classes_, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}"}))

        # Confusion Matrix
        st.subheader("🔍 Confusion Matrix")
        cm = confusion_matrix(y_encoded, le.transform(y_pred))
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # ROC Curve (only for binary classification)
        if len(le.classes_) == 2:
            st.subheader("🧠 ROC Curve & AUC Score")
            fpr, tpr, _ = roc_auc_score(y_encoded, y_proba[:, 1], average=None), [0], [0]
            auc_score = roc_auc_score(y_encoded, y_proba[:, 1])
            fpr, tpr, _ = roc_curve(y_encoded, y_proba[:, 1])

            fig2, ax2 = plt.subplots()
            ax2.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
            ax2.plot([0, 1], [0, 1], "k--")
            ax2.set_xlabel("False Positive Rate")
            ax2.set_ylabel("True Positive Rate")
            ax2.set_title("ROC Curve")
            ax2.legend()
            st.pyplot(fig2)

            st.success(f"🟢 ROC-AUC Score: **{auc_score:.2f}**")

    else:
        st.warning("🚫 Test dataset must contain 'text' and 'sentiment' columns.")