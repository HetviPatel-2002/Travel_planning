# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # Load dataset
# file_path = "D:/Make_my_trip/previous_ds/2.csv"
# df = pd.read_csv(file_path)

# # Filter users who booked flights and hotels but NOT a car
# users_without_cars = df[(df['flight_price'].notna()) & 
#                         (df['total_hotel'].notna()) & 
#                         (df['carType'].isna())].copy()

# # Users who have booked a car (used for recommendations)
# users_with_cars = df[df['carType'].notna()].copy()

# # Fill missing values in feature columns using .loc to avoid warnings
# feature_cols = ['pickupLocation', 'dropoffLocation', 'rentalAgency', 'fuelPolicy']
# users_with_cars.loc[:, feature_cols] = users_with_cars[feature_cols].fillna("Unknown")
# users_without_cars.loc[:, feature_cols] = users_without_cars[feature_cols].fillna("Unknown")

# # Combine feature columns into a single text feature
# users_with_cars.loc[:, 'combined_features'] = users_with_cars[feature_cols].astype(str).agg(' '.join, axis=1)
# users_without_cars.loc[:, 'combined_features'] = users_without_cars[feature_cols].astype(str).agg(' '.join, axis=1)

# # TF-IDF Vectorization
# tfidf = TfidfVectorizer(stop_words='english')
# tfidf_matrix_with_cars = tfidf.fit_transform(users_with_cars['combined_features'])
# tfidf_matrix_without_cars = tfidf.transform(users_without_cars['combined_features'])

# # Compute similarity between users who need cars and those who booked cars
# cosine_sim = cosine_similarity(tfidf_matrix_without_cars, tfidf_matrix_with_cars)

# # Function to recommend a car for a specific user
# def recommend_car(user_id):
#     if user_id not in users_without_cars['user_id'].values:
#         return f"User {user_id} has already booked a car or does not exist in the dataset."

#     # Find the index of the user in users_without_cars
#     user_index = users_without_cars[users_without_cars['user_id'] == user_id].index[0]
    
#     # Find the most similar user who booked a car
#     most_similar_index = cosine_sim[user_index].argmax()
    
#     # Get the recommended car details
#     recommended_car = users_with_cars.iloc[most_similar_index][['carType', 'rentalAgency']]
    
#     return f"Recommended Car: {recommended_car['carType']} from {recommended_car['rentalAgency']}"

# # Example: Recommend a car for user with ID 5
# user_id = 5  # Change this to any user ID you want to check
# print(recommend_car(user_id))

# from sklearn.metrics import precision_score, recall_score, f1_score

# # Step 1: Identify users who booked a car later (as ground truth for evaluation)
# users_who_booked_later = df[(df['user_id'].isin(users_without_cars['user_id'])) & (df['carType'].notna())]

# # Step 2: Make predictions for these users
# y_true = users_who_booked_later['carType'].values  # Actual booked car types
# y_pred = []

# for user_id in users_who_booked_later['user_id']:
#     recommendation = recommend_car(user_id)
#     if "Recommended Car" in recommendation:
#         y_pred.append(recommendation.split(": ")[1].split(" from ")[0])  # Extract only the car type
#     else:
#         y_pred.append("Unknown")  # If no recommendation, treat as unknown

# # Step 3: Calculate Precision, Recall, F1-Score
# precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
# recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
# f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

# # Step 4: Print Evaluation Metrics
# print(f"Precision: {precision:.2f}")
# print(f"Recall: {recall:.2f}")
# print(f"F1-score: {f1:.2f}")
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score

# Load dataset
file_path = "D:/Make_my_trip/previous_ds/2.csv"
df = pd.read_csv(file_path)

# Filter users who booked flights and hotels but NOT a car
users_without_cars = df[(df['flight_price'].notna()) & 
                        (df['total_hotel'].notna()) & 
                        (df['carType'].isna())].copy()

# Users who have booked a car (used for recommendations)
users_with_cars = df[df['carType'].notna()].copy()

# Reset index to avoid indexing issues
users_without_cars = users_without_cars.reset_index(drop=True)
users_with_cars = users_with_cars.reset_index(drop=True)

# Fill missing values in feature columns
feature_cols = ['pickupLocation', 'dropoffLocation', 'rentalAgency', 'fuelPolicy']
users_with_cars.loc[:, feature_cols] = users_with_cars[feature_cols].fillna("Unknown")
users_without_cars.loc[:, feature_cols] = users_without_cars[feature_cols].fillna("Unknown")

# Combine feature columns into a single text feature
users_with_cars.loc[:, 'combined_features'] = users_with_cars[feature_cols].astype(str).agg(' '.join, axis=1)
users_without_cars.loc[:, 'combined_features'] = users_without_cars[feature_cols].astype(str).agg(' '.join, axis=1)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix_with_cars = tfidf.fit_transform(users_with_cars['combined_features'])
tfidf_matrix_without_cars = tfidf.transform(users_without_cars['combined_features'])

# Compute similarity between users who need cars and those who booked cars
cosine_sim = cosine_similarity(tfidf_matrix_without_cars, tfidf_matrix_with_cars)

# Function to recommend a car for a specific user
def recommend_car(user_id):
    if user_id not in users_without_cars['user_id'].values:
        return f"User {user_id} has already booked a car or does not exist in the dataset."

    # Find user index in users_without_cars
    user_index_list = users_without_cars.index[users_without_cars['user_id'] == user_id].tolist()

    if not user_index_list:  # Ensure index exists
        return f"User {user_id} not found in recommendation dataset."

    user_index = user_index_list[0]  # Extract first match

    # Ensure index is within valid range
    if user_index >= cosine_sim.shape[0]:
        return f"User {user_id} index ({user_index}) is out of bounds."

    # Find the most similar user who booked a car
    most_similar_index = cosine_sim[user_index].argmax()

    # Get the recommended car details
    recommended_car = users_with_cars.iloc[most_similar_index][['carType', 'rentalAgency']]

    return recommended_car['carType'], recommended_car['rentalAgency']

# Function to evaluate recommendations
def evaluate_recommendations():
    actual_cars = users_with_cars['carType'].unique()
    recommended_cars = []

    for user_id in users_without_cars['user_id']:
        rec_car, rec_agency = recommend_car(user_id)
        recommended_cars.append(rec_car)

    recommended_cars = np.array(recommended_cars)

    # Precision & Recall Calculation
    y_true = np.isin(recommended_cars, actual_cars).astype(int)
    y_pred = np.ones_like(y_true)  # All recommendations are considered positive predictions

    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)

    # Coverage Calculation
    unique_recommended = len(set(recommended_cars))
    total_possible = len(actual_cars)
    coverage = unique_recommended / total_possible if total_possible > 0 else 0

    print(f"🔹 **Precision:** {precision:.2f}")
    print(f"🔹 **Recall:** {recall:.2f}")
    print(f"🔹 **Coverage:** {coverage:.2f}")

# Example: Recommend a car for user with ID 5
user_id = 2  # Change this to any user ID you want to check
print(f"Recommendation for User {user_id}: {recommend_car(user_id)}")

# Run Evaluation
evaluate_recommendations()

