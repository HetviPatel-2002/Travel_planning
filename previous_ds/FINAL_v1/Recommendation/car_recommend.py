import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime

class LightweightTravelCarRecommender:
    def __init__(self):
        self.flight_type_scores = {
            'economic': 1,
            'premium': 2.5,
            'firstclass': 4,
            'economy': 1,
            'first class': 4
        }
    
    def preprocess_data(self, df):
        """Basic preprocessing with essential features"""
        df = df.copy()
        
        # Date processing
        date_columns = ['flight_date', 'check_in']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format='%d-%m-%Y', errors='coerce')
                df[f'{col}_is_weekend'] = df[col].dt.weekday.isin([5, 6]).astype(int)
        
        # Basic features
        df['price_per_distance'] = df['flight_price'] / np.maximum(df['distance'], 1)
        df['hotel_price_per_day'] = df['total_hotel'] / np.maximum(df['days'], 1)
        
        # Flight class scoring
        df['flightType'] = df['flightType'].str.lower()
        df['flight_class_score'] = df['flightType'].map(self.flight_type_scores).fillna(1)
        
        return df
    
    def create_user_profile(self, df):
        """Create basic user profile"""
        user_profile = {
            'flight_type': df['flightType'].iloc[0],
            'avg_trip_duration': df['days'].mean(),
            'avg_flight_price': df['flight_price'].mean(),
            'avg_hotel_price': df['total_hotel'].mean(),
            'prefers_weekend': df['flight_date_is_weekend'].mean() > 0.5,
            'age': df['age'].iloc[0],
            'gender': df['gender'].iloc[0]
        }
        return user_profile
    
    def recommend_car_type(self, user_profile):
        """Recommend car types based on user profile"""
        scores = {}
        car_types = ['Economy', 'Standard', 'Premium', 'SUV', 'Luxury']
        
        for car_type in car_types:
            score = 0
            
            # Flight class alignment (35%)
            if user_profile['flight_type'] == 'firstclass':
                if car_type in ['Luxury', 'Premium']:
                    score += 3.5
                elif car_type == 'SUV':
                    score += 2
            elif user_profile['flight_type'] == 'premium':
                if car_type in ['Premium', 'Standard']:
                    score += 3
                elif car_type == 'SUV':
                    score += 2
            else:  # economic
                if car_type in ['Economy', 'Standard']:
                    score += 3
            
            # Trip duration impact (25%)
            if user_profile['avg_trip_duration'] >= 4:
                if car_type in ['SUV', 'Premium']:
                    score += 2.5
            elif user_profile['avg_trip_duration'] >= 2:
                if car_type in ['Standard', 'Premium']:
                    score += 2
            else:
                if car_type in ['Economy', 'Standard']:
                    score += 1.5
            
            # Price sensitivity (25%)
            avg_total_cost = user_profile['avg_flight_price'] + user_profile['avg_hotel_price']
            if avg_total_cost > 5000:  # High budget
                if car_type in ['Luxury', 'Premium']:
                    score += 2.5
            elif avg_total_cost > 2500:  # Medium budget
                if car_type in ['Standard', 'Premium']:
                    score += 2
            else:  # Low budget
                if car_type in ['Economy', 'Standard']:
                    score += 1.5
            
            # Weekend preference (15%)
            if user_profile['prefers_weekend']:
                if car_type in ['SUV', 'Premium', 'Luxury']:
                    score += 1.5
            
            scores[car_type] = round(score, 1)
        
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

class RecommendationEvaluator:
    def __init__(self, recommender):
        self.recommender = recommender
        self.metrics = {}
        
    def evaluate(self, test_data):
        """
        Evaluate the recommendation system using multiple metrics
        """
        print("Starting evaluation...")
        self.metrics = {}
        
        # Convert test data to list of actual car choices
        actual_choices = test_data['carType'].tolist()
        
        # Get recommendations for each user in test data
        predicted_choices = []
        user_satisfactions = []
        recommendation_relevance = []
        
        for _, user_data in test_data.groupby('user_id'):
            # Process user data
            processed_data = self.recommender.preprocess_data(user_data)
            user_profile = self.recommender.create_user_profile(processed_data)
            
            # Get recommendations
            recommendations = self.recommender.recommend_car_type(user_profile)
            top_recommendation = recommendations[0][0]  # Get the highest scored car
            predicted_choices.append(top_recommendation)
            
            # Calculate user satisfaction score
            actual_car = user_data['carType'].iloc[0]
            satisfaction = self._calculate_satisfaction_score(
                recommendations, actual_car, user_profile)
            user_satisfactions.append(satisfaction)
            
            # Calculate recommendation relevance
            relevance = self._calculate_relevance_score(
                recommendations, actual_car, user_profile)
            recommendation_relevance.append(relevance)
        
        # Calculate accuracy metrics
        self.metrics['accuracy'] = sum(
            [1 for pred, actual in zip(predicted_choices, actual_choices) 
             if pred.lower() == actual.lower()]) / len(actual_choices)
        
        # Calculate average satisfaction and relevance
        self.metrics['avg_satisfaction'] = np.mean(user_satisfactions)
        self.metrics['avg_relevance'] = np.mean(recommendation_relevance)
        
        # Calculate recommendation distribution
        self.metrics['recommendation_distribution'] = self._calculate_recommendation_distribution(
            predicted_choices)
        
        # Calculate business metrics
        self.metrics['business_metrics'] = self._calculate_business_metrics(
            test_data, predicted_choices)
        
        return self.metrics
    
    def _calculate_satisfaction_score(self, recommendations, actual_car, user_profile):
        """Calculate user satisfaction based on recommendation ranking"""
        recommendation_dict = dict(recommendations)
        max_score = max(recommendation_dict.values())
        
        # Check if actual choice was recommended
        if actual_car in recommendation_dict:
            position = [car for car, _ in recommendations].index(actual_car)
            score = recommendation_dict[actual_car] / max_score
            return (1 / (position + 1)) * score  # Position penalty
        return 0
    
    def _calculate_relevance_score(self, recommendations, actual_car, user_profile):
        """Calculate how relevant the recommendations were based on user profile"""
        score = 0
        recommendation_dict = dict(recommendations)
        
        # Check flight class alignment
        flight_type = user_profile['flight_type']
        if flight_type == 'firstclass' and recommendation_dict.get('Luxury', 0) > 5:
            score += 0.3
        elif flight_type == 'premium' and recommendation_dict.get('Premium', 0) > 5:
            score += 0.3
        elif flight_type == 'economic' and recommendation_dict.get('Economy', 0) > 5:
            score += 0.3
            
        # Check trip duration alignment
        avg_duration = user_profile['avg_trip_duration']
        if avg_duration >= 4 and recommendation_dict.get('SUV', 0) > 5:
            score += 0.2
        elif avg_duration < 2 and recommendation_dict.get('Economy', 0) > 5:
            score += 0.2
            
        # Check weekend preference alignment
        if user_profile['prefers_weekend'] and recommendation_dict.get('SUV', 0) > 5:
            score += 0.2
            
        return score
    
    def _calculate_recommendation_distribution(self, predictions):
        """Calculate the distribution of recommendations"""
        distribution = defaultdict(int)
        total = len(predictions)
        
        for pred in predictions:
            distribution[pred] += 1
            
        # Convert to percentages
        return {k: (v/total)*100 for k, v in distribution.items()}
    
    def _calculate_business_metrics(self, test_data, predictions):
        """Calculate business-relevant metrics"""
        metrics = {
            'avg_booking_value': test_data['total_rent_price'].mean(),
            'booking_rate': (test_data['bookingStatus'] == 'Confirmed').mean(),
            'premium_recommendation_rate': sum(1 for p in predictions 
                                            if p in ['Luxury', 'Premium']) / len(predictions)
        }
        return metrics
    
    def print_evaluation_report(self):
        """Print a detailed evaluation report"""
        if not self.metrics:
            print("No evaluation metrics available. Run evaluate() first.")
            return
            
        print("\n=== Recommendation System Evaluation Report ===")
        print("\n1. Accuracy Metrics:")
        print(f"Overall Accuracy: {self.metrics['accuracy']:.2%}")
        print(f"Average User Satisfaction: {self.metrics['avg_satisfaction']:.2f}/1.0")
        print(f"Average Recommendation Relevance: {self.metrics['avg_relevance']:.2f}/1.0")
        
        print("\n2. Recommendation Distribution:")
        for car_type, percentage in self.metrics['recommendation_distribution'].items():
            print(f"{car_type}: {percentage:.1f}%")
        
        print("\n3. Business Metrics:")
        bm = self.metrics['business_metrics']
        print(f"Average Booking Value: ${bm['avg_booking_value']:.2f}")
        print(f"Booking Rate: {bm['booking_rate']:.2%}")
        print(f"Premium Recommendation Rate: {bm['premium_recommendation_rate']:.2%}")

def main():
    try:
        # Create instances
        recommender = LightweightTravelCarRecommender()
        evaluator = RecommendationEvaluator(recommender)
        
        # Load test data
        print("Loading test data...")
        test_data = pd.read_csv(r"D:\Make_my_trip\previous_ds\2.csv")
        
        # Run evaluation
        metrics = evaluator.evaluate(test_data)
        
        # Print detailed report
        evaluator.print_evaluation_report()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()