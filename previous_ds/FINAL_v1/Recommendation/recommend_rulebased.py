import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class TravelDataProcessor:
    def __init__(self, data_path):
        """Initialize with the travel booking dataset"""
        self.raw_data = pd.read_csv(data_path)
        self.processed_data = None
        
    def preprocess_data(self):
        """Preprocess the raw data and engineer features for car recommendations"""
        df = self.raw_data.copy()
        
        # Convert dates to datetime
        df['flight_date'] = pd.to_datetime(df['flight_date'])
        df['check_in'] = pd.to_datetime(df['check_in'])
        
        # Calculate check_out date
        df['check_out'] = df['check_in'] + pd.to_timedelta(df['days'], unit='D')
        
        # Engineer features relevant for car recommendations
        df['trip_duration'] = df['days']
        df['is_long_distance'] = df['distance'] > df['distance'].median()
        
        # Calculate potential car need features
        df['needs_airport_transfer'] = True  # Since all bookings include flights
        
        # Calculate price per day for reference
        df['price_per_day'] = df['total_price'] / df['days']
        
        # Create binary features
        df['is_business_class'] = df['flightType'].isin(['firstClass', 'premium'])
        df['is_expensive_hotel'] = df['hotel_base'] > df['hotel_base'].median()
        
        # Create potential car rental locations
        df['suggested_pickup_location'] = df['arrival']
        df['suggested_dropoff_location'] = df['arrival']
        
        # Suggest car rental duration (same as hotel stay)
        df['suggested_rental_duration'] = df['days']
        
        # Create feature for potential car recommendation
        df['recommend_car'] = (
            (df['distance'] > 500) | 
            (df['days'] >= 3) | 
            (df['price_per_day'] > df['price_per_day'].median())
        )
        
        self.processed_data = df
        return df
    
    def generate_car_recommendations(self, user_id):
        """Generate car rental recommendations for a specific user"""
        user_data = self.processed_data[self.processed_data['user_id'] == user_id]
        
        recommendations = []
        
        for _, booking in user_data.iterrows():
            rec = {
                'booking_id': booking['travelCode'],
                'destination': booking['arrival'],
                'recommended_car_type': self._suggest_car_type(booking),
                'pickup_location': booking['arrival'],  # Airport by default
                'dropoff_location': booking['arrival'],
                'suggested_duration': booking['days'],
                'estimated_price': self._estimate_car_price(booking),
                'recommendation_reason': self._get_recommendation_reason(booking)
            }
            recommendations.append(rec)
            
        return recommendations
    
    def _suggest_car_type(self, booking):
        """Suggest car type based on booking details"""
        if booking['flightType'] in ['firstClass', 'premium']:
            return 'Luxury'
        elif booking['total_price'] > self.processed_data['total_price'].median():
            return 'Premium'
        elif booking['days'] > 3:
            return 'Midsize'
        else:
            return 'Economy'
    
    def _estimate_car_price(self, booking):
        """Estimate car rental price based on booking patterns"""
        # Simple estimation based on hotel price as reference
        base_price_per_day = booking['hotel_base'] * 0.5
        return base_price_per_day * booking['days']
    
    def _get_recommendation_reason(self, booking):
        """Generate reasoning for car recommendation"""
        reasons = []
        
        if booking['distance'] > 500:
            reasons.append("Long distance travel")
        if booking['days'] >= 3:
            reasons.append("Extended stay duration")
        if booking['flightType'] in ['firstClass', 'premium']:
            reasons.append("Premium travel preference")
            
        return ", ".join(reasons) if reasons else "Convenience for local travel"

def main():
    # Initialize processor with your data
    processor = TravelDataProcessor('sample_ds.csv')
    
    # Preprocess the data
    processed_data = processor.preprocess_data()
    
    # Generate recommendations for user 0
    recommendations = processor.generate_car_recommendations(0)
    
    # Print recommendations
    print("\nCar Rental Recommendations:")
    for rec in recommendations:
        print("\nBooking ID:", rec['booking_id'])
        print(f"Destination: {rec['destination']}")
        print(f"Recommended Car Type: {rec['recommended_car_type']}")
        print(f"Pickup Location: {rec['pickup_location']}")
        print(f"Dropoff Location: {rec['dropoff_location']}")
        print(f"Suggested Duration: {rec['suggested_duration']} days")
        print(f"Estimated Price: ${rec['estimated_price']:.2f}")
        print(f"Reason: {rec['recommendation_reason']}")

if __name__ == "__main__":
    main()