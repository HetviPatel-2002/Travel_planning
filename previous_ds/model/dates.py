import pandas as pd

# File paths
flights_file = "D:\\Flask_tutorial\\mytrip\\model\\Flights(old).csv"
hotel_stay_file = "D:\\Flask_tutorial\\mytrip\\model\\Hotel_stay(old).csv"
rent_a_car_file = "D:\\Flask_tutorial\\mytrip\\model\\Rent_a_car (old).csv"

# Load the datasets
flights_df = pd.read_csv(flights_file)
hotel_stay_df = pd.read_csv(hotel_stay_file)
rent_a_car_df = pd.read_csv(rent_a_car_file)

# Standardize date formats
# Flights: Convert 'date' to datetime in 'YYYY-MM-DD' format without altering other data
if 'date' in flights_df.columns:
    flights_df['date'] = pd.to_datetime(flights_df['date'], errors='coerce').dt.strftime('%Y-%m-%d')

# Ensure flight arrival aligns with hotel check-in dates without altering unrelated data
for index, row in hotel_stay_df.iterrows():
    matching_flight_date = flights_df.loc[flights_df['travelCode'] == row['travelCode'], 'date']
    if not matching_flight_date.empty:
        flight_date = pd.to_datetime(matching_flight_date.iloc[0], errors='coerce')
        check_in_date = pd.to_datetime(row['checkInDate'], errors='coerce')
        # Adjust check-in date only if it is earlier than the flight date
        if not pd.isna(flight_date) and not pd.isna(check_in_date) and check_in_date < flight_date:
            hotel_stay_df.at[index, 'checkInDate'] = flight_date.strftime('%Y-%m-%d')

# Ensure 'checkInDate' and 'checkOutDate' remain in 'YYYY-MM-DD' format without other changes
if 'checkInDate' in hotel_stay_df.columns:
    hotel_stay_df['checkInDate'] = pd.to_datetime(hotel_stay_df['checkInDate'], errors='coerce').dt.strftime('%Y-%m-%d')
if 'checkOutDate' in hotel_stay_df.columns:
    hotel_stay_df['checkOutDate'] = pd.to_datetime(hotel_stay_df['checkOutDate'], errors='coerce').dt.strftime('%Y-%m-%d')

# Ensure 'pickupDateTime' and 'dropoffDateTime' remain in 'YYYY-MM-DD' format without altering other data
if 'pickupDateTime' in rent_a_car_df.columns:
    rent_a_car_df['pickupDateTime'] = pd.to_datetime(rent_a_car_df['pickupDateTime'], errors='coerce').dt.strftime('%Y-%m-%d')
if 'dropoffDateTime' in rent_a_car_df.columns:
    rent_a_car_df['dropoffDateTime'] = pd.to_datetime(rent_a_car_df['dropoffDateTime'], errors='coerce').dt.strftime('%Y-%m-%d')

# Ensure car rental pickup dates align with hotel check-out dates or flight arrival dates without altering unrelated data
for index, row in rent_a_car_df.iterrows():
    matching_hotel_checkout = hotel_stay_df.loc[hotel_stay_df['travelCode'] == row['travelCode'], 'checkOutDate']
    matching_flight_date = flights_df.loc[flights_df['travelCode'] == row['travelCode'], 'date']

    suggested_pickup_date = None
    if not matching_hotel_checkout.empty:
        suggested_pickup_date = matching_hotel_checkout.iloc[0]
    elif not matching_flight_date.empty:
        suggested_pickup_date = matching_flight_date.iloc[0]

    if suggested_pickup_date and pd.notna(suggested_pickup_date):
        rent_a_car_df.at[index, 'pickupDateTime'] = pd.to_datetime(suggested_pickup_date, errors='coerce').strftime('%Y-%m-%d')

# Save the updated datasets
flights_df.to_csv("D:\\Flask_tutorial\\mytrip\\model\\flights_aligned.csv", index=False)
hotel_stay_df.to_csv("D:\\Flask_tutorial\\mytrip\\model\\hotel_stay_aligned.csv", index=False)
rent_a_car_df.to_csv("D:\\Flask_tutorial\\mytrip\\model\\rent_a_car_aligned.csv", index=False)

print("Datasets have been updated and saved with aligned dates without altering unrelated data.")
