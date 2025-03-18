# import pandas as pd

# # Load the datasets
# car_df = pd.read_excel("D:\\Make_my_trip\\FinalDataset\\CarFINALdataset.xlsx")
# flight_df = pd.read_excel("D:\\Make_my_trip\\FinalDataset\\FlightFINALdataset.xlsx")
# hotel_df = pd.read_excel("D:\\Make_my_trip\\FinalDataset\\HotelFINALdataset.xlsx")
# updated_user_df = pd.read_csv("D:\\Make_my_trip\\updated_user_1.csv")

# # Standardize column names for merging
# car_df.rename(columns={'User_ID': 'user_id'}, inplace=True)
# flight_df.rename(columns={'User_ID': 'user_id'}, inplace=True)
# hotel_df.rename(columns={'User_ID': 'user_id'}, inplace=True)
# updated_user_df.rename(columns={'code': 'user_id'}, inplace=True)

# # Combine datasets to get unique travelCode per user_id
# dataframes = [car_df[['user_id', 'travelCode']], 
#               flight_df[['user_id', 'travelCode']], 
#               hotel_df[['user_id', 'travelCode']]]

# merged_df = pd.concat(dataframes).drop_duplicates().reset_index(drop=True)

# # Merge the travelCode into updated_user_df
# updated_user_df = updated_user_df.merge(merged_df, on='user_id', how='left')

# # Save updated dataset
# updated_user_df.to_csv("user_1.csv", index=False)
#  version two..
import pandas as pd

# Load the datasets
car_df = pd.read_excel("D:\\Make_my_trip\\FinalDataset\\CarFINALdataset.xlsx")
flight_df = pd.read_excel("D:\\Make_my_trip\\FinalDataset\\FlightFINALdataset.xlsx")
hotel_df = pd.read_excel("D:\\Make_my_trip\\FinalDataset\\HotelFINALdataset.xlsx")
updated_user_df = pd.read_csv("D:\\Make_my_trip\\updated_user_1.csv")

# Standardize column names for consistency
car_df.rename(columns={'User_ID': 'user_id'}, inplace=True)
flight_df.rename(columns={'User_ID': 'user_id'}, inplace=True)
hotel_df.rename(columns={'User_ID': 'user_id'}, inplace=True)
updated_user_df.rename(columns={'code': 'user_id'}, inplace=True)

# Combine travelCode information per user_id
travel_code_df = pd.concat([
    car_df[['user_id', 'travelCode']], 
    flight_df[['user_id', 'travelCode']], 
    hotel_df[['user_id', 'travelCode']]
]).dropna().drop_duplicates()

# Assign a single travelCode per user_id (using the first occurrence)
travel_code_df = travel_code_df.groupby('user_id')['travelCode'].first().reset_index()

# Merge the travelCode into updated_user_df without duplicating rows
updated_user_df = updated_user_df.merge(travel_code_df, on='user_id', how='left')

# Save updated dataset
updated_user_df.to_csv("user_2.csv", index=False)
