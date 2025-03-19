import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score

# Step 1: Data Ingestion

def fetch_data():
    db_config = {
        "host": os.getenv("DB_HOST"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "database": "mmt"
    }
    encoded_password = quote_plus(db_config["password"])
    engine = create_engine(f"mysql+pymysql://{db_config['user']}:{encoded_password}@{db_config['host']}/{db_config['database']}")
    
    try:
        query_car = "SELECT * FROM car"
        car_df = pd.read_sql(query_car, engine)
        query_car_rental = "SELECT * FROM rentals"
        car_rental_df = pd.read_sql(query_car_rental, engine)
    except Exception as e:
        print(f"Error: {e}")
        return None, None
    
    return car_df, car_rental_df

# Step 2: Data Preprocessing

def preprocess_data(car_df, car_rental_df):
    car_df.rename(columns={"CarID": "car_id"}, inplace=True)
    merged_df = car_rental_df.merge(car_df, left_on="CarID", right_on="car_id", how="left")
    merged_df.drop(columns=["car_id", "City"], inplace=True)
    
    label_encoders = {}
    categorical_columns = ["Pickup_Location", "Make", "Model", "CarType", "Car_Agency"]
    
    for col in categorical_columns:
        le = LabelEncoder()
        merged_df[f"{col}_encoded"] = le.fit_transform(merged_df[col])
        label_encoders[col] = le
    
    merged_df.drop(columns=categorical_columns, inplace=True)
    return merged_df, label_encoders

# Step 3: Compute User Similarity

def compute_similarity(df):
    interaction_matrix = df.pivot_table(index="UserID", columns="CarID", values="TotalAmount", aggfunc=np.sum, fill_value=0)
    interaction_sparse = csr_matrix(interaction_matrix.values)
    user_similarity = cosine_similarity(interaction_sparse, dense_output=False)
    
    user_sim_dict = {i: user_similarity[i].toarray().flatten() for i in range(user_similarity.shape[0])}
    top_n_users = {user: np.argsort(-similarities)[:10] for user, similarities in user_sim_dict.items()}
    
    return interaction_matrix, top_n_users

# Step 4: Car Recommendation

def recommend_cars(user_id, df, top_n_users, interaction_matrix, label_encoders, num_recommendations=5):
    if user_id not in interaction_matrix.index:
        print("User not found in dataset.")
        return []
    
    similar_users = top_n_users[user_id]
    user_pickup_location = df.loc[df["UserID"] == user_id, "Pickup_Location_encoded"].values
    if len(user_pickup_location) == 0:
        print("No pickup location found for user.")
        return []
    
    user_pickup_location = user_pickup_location[0]
    recommended_cars = []
    
    for sim_user in similar_users:
        sim_user_cars = df[(df["UserID"] == sim_user) & (df["Pickup_Location_encoded"] == user_pickup_location)]["CarID"].unique()
        recommended_cars.extend(sim_user_cars)
    
    recommended_cars = list(set(recommended_cars))[:num_recommendations]
    if not recommended_cars:
        print("No similar users found with cars from the same pickup location.")
        return []
    
    car_details = df[df["CarID"].isin(recommended_cars)][["CarID", "Make_encoded", "Model_encoded", "CarType_encoded", "Pickup_Location_encoded"]].drop_duplicates()
    car_details["Make"] = label_encoders["Make"].inverse_transform(car_details["Make_encoded"])
    car_details["Model"] = label_encoders["Model"].inverse_transform(car_details["Model_encoded"])
    car_details["CarType"] = label_encoders["CarType"].inverse_transform(car_details["CarType_encoded"])
    car_details["Pickup_Location"] = label_encoders["Pickup_Location"].inverse_transform(car_details["Pickup_Location_encoded"])
    car_details.drop(columns=["Make_encoded", "Model_encoded", "CarType_encoded", "Pickup_Location_encoded"], inplace=True)
    
    return car_details

# Step 5: Evaluation Metrics

def evaluate_recommendations(user_id, df, recommended_cars, interaction_matrix, k=5):
    actual_rented_cars = df[df["UserID"] == user_id]["CarID"].unique()
    if len(actual_rented_cars) == 0:
        print("No historical rentals for this user. Evaluation not possible.")
        return {}
    
    recommended_set = set(recommended_cars["CarID"]) if not recommended_cars.empty else set()
    actual_set = set(actual_rented_cars)
    
    precision_at_k = len(recommended_set & actual_set) / k
    recall_at_k = len(recommended_set & actual_set) / len(actual_set)
    f1_score = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k) if precision_at_k + recall_at_k > 0 else 0.0
    ranks = [i + 1 for i, car in enumerate(recommended_cars["CarID"]) if car in actual_set]
    mrr = 1 / min(ranks) if ranks else 0.0
    
    metrics = {
        "Precision@K": precision_at_k,
        "Recall@K": recall_at_k,
        "F1-Score@K": f1_score,
        "MRR": mrr
    }
    
    return metrics

# Execute Pipeline

car_df, car_rental_df = fetch_data()
if car_df is not None and car_rental_df is not None:
    df, label_encoders = preprocess_data(car_df, car_rental_df)
    interaction_matrix, top_n_users = compute_similarity(df)
    user_id = 5
    recommended_cars = recommend_cars(user_id, df, top_n_users, interaction_matrix, label_encoders)
    print("\nRecommended Cars for User", user_id)
    print(recommended_cars)
    metrics = evaluate_recommendations(user_id, df, recommended_cars, interaction_matrix)
    print("\n🔍 Evaluation Metrics for User", user_id)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
