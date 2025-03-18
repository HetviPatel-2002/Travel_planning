import pandas as pd
import numpy as np
import random
from faker import Faker

faker = Faker()

# Load the existing CSV datasets
flights_df = pd.read_csv('D:\\Flask_tutorial\\mytrip\\model\\flights.csv')
hotels_df = pd.read_csv('D:\\Flask_tutorial\\mytrip\\model\\hotels.csv')
users_df = pd.read_csv('D:\\Flask_tutorial\\mytrip\\model\\users.csv')

num_records = 10000

# Ensure correct column usage for user references
user_code_column = 'code'

# Generate Car Rental Data
def generate_car_rental_data(num_records):
    travel_codes = flights_df['travelCode'].tolist() + hotels_df['travelCode'].tolist()
    rental_agencies = ['Hertz', 'Avis', 'Enterprise', 'Budget', 'Sixt']
    car_types = ['Sedan', 'SUV', 'Luxury', 'Hatchback']
    fuel_policies = ['Full-to-Full', 'Prepaid', 'Partial']
    booking_statuses = ['Confirmed', 'Cancelled', 'Pending']

    car_rental_data = []
    for _ in range(num_records):
        travel_code = random.choice(travel_codes)
        user_code = random.choice(users_df[user_code_column].tolist())
        pickup_location = faker.city()
        dropoff_location = faker.city()
        car_type = random.choice(car_types)
        rental_agency = random.choice(rental_agencies)
        pickup_date = faker.date_time_this_year()
        dropoff_date = pickup_date + pd.to_timedelta(random.randint(1, 5), unit='d')
        rental_duration = (dropoff_date - pickup_date).days
        price = round(random.uniform(30, 200) * rental_duration, 2)
        total_distance = random.randint(50, 500)
        fuel_policy = random.choice(fuel_policies)
        booking_status = random.choice(booking_statuses)

        car_rental_data.append([
            travel_code, user_code, pickup_location, dropoff_location, car_type, rental_agency,
            pickup_date, dropoff_date, rental_duration, price, total_distance, fuel_policy, booking_status
        ])

    columns = ['travelCode', 'code', 'pickupLocation', 'dropoffLocation', 'carType', 'rentalAgency',
               'pickupDateTime', 'dropoffDateTime', 'rentalDuration', 'price', 'totalDistance', 'fuelPolicy', 'bookingStatus']

    car_rental_df = pd.DataFrame(car_rental_data, columns=columns)
    car_rental_df.to_csv('./rent_a_car.csv', index=False)
    return car_rental_df

# Generate Passenger Profile Data
def generate_passenger_profile_data(num_records):
    travel_codes = flights_df['travelCode'].tolist()
    passenger_profiles = []

    for _ in range(num_records):
        pax_id = faker.uuid4()
        travel_code = random.choice(travel_codes)
        user_code = random.choice(users_df[user_code_column].tolist())
        name = faker.name()
        gender = random.choice(['Male', 'Female', 'Other'])
        age = random.randint(18, 75)
        passport_number = faker.uuid4() if random.random() > 0.3 else ''  # Passport for international flights
        nationality = faker.country()
        seat_preference = random.choice(['Window', 'Aisle', 'Middle'])
        meal_preference = random.choice(['Veg', 'Non-Veg', 'Special Meals'])
        baggage_details = f'{random.randint(1, 2)} bag(s)'

        passenger_profiles.append([
            pax_id, travel_code, user_code, name, gender, age, passport_number, nationality,
            seat_preference, meal_preference, baggage_details
        ])

    columns = ['paxID', 'travelCode', 'code', 'name', 'gender', 'age', 'passportNumber', 'nationality',
               'seatPreference', 'mealPreference', 'baggageDetails']

    pax_df = pd.DataFrame(passenger_profiles, columns=columns)
    pax_df.to_csv('./pax_profile.csv', index=False)
    return pax_df

# Generate Guest Profile Data
def generate_guest_profile_data(num_records):
    travel_codes = hotels_df['travelCode'].tolist()
    guest_profiles = []

    for _ in range(num_records):
        guest_id = faker.uuid4()
        travel_code = random.choice(travel_codes)
        user_code = random.choice(users_df[user_code_column].tolist())
        name = faker.name()
        gender = random.choice(['Male', 'Female', 'Other'])
        age = random.randint(18, 75)
        contact_number = faker.phone_number()
        email = faker.email()
        special_requests = random.choice(['Extra bed', 'Late check-in', 'None'])
        room_preference = random.choice(['Single', 'Double', 'Suite'])
        id_proof = faker.uuid4()

        guest_profiles.append([
            guest_id, travel_code, user_code, name, gender, age, contact_number, email,
            special_requests, room_preference, id_proof
        ])

    columns = ['guestID', 'travelCode', 'code', 'name', 'gender', 'age', 'contactNumber', 'email',
               'specialRequests', 'roomPreference', 'idProof']

    guest_df = pd.DataFrame(guest_profiles, columns=columns)
    guest_df.to_csv('./guest_profile.csv', index=False)
    return guest_df

# Generate Hotel Stay Data
def generate_hotel_stay_data(num_records):
    travel_codes = hotels_df['travelCode'].tolist()
    hotel_stays = []

    for _ in range(num_records):
        stay_id = faker.uuid4()
        travel_code = random.choice(travel_codes)
        user_code = random.choice(users_df[user_code_column].tolist())
        hotel_name = faker.company()
        location = faker.city()
        room_type = random.choice(['Deluxe', 'Suite', 'Standard'])
        check_in_date = faker.date_this_year()
        check_out_date = pd.to_datetime(check_in_date) + pd.to_timedelta(random.randint(1, 5), unit='d')
        total_nights = (check_out_date - pd.to_datetime(check_in_date)).days
        num_guests = random.randint(1, 4)
        price_per_night = round(random.uniform(50, 300), 2)
        total_cost = price_per_night * total_nights
        booking_status = random.choice(['Confirmed', 'Cancelled', 'Pending'])
        payment_status = random.choice(['Paid', 'Pending', 'Partially Paid'])

        hotel_stays.append([
            stay_id, travel_code, user_code, hotel_name, location, room_type, check_in_date,
            check_out_date, total_nights, num_guests, price_per_night, total_cost,
            booking_status, payment_status
        ])

    columns = ['stayID', 'travelCode', 'code', 'hotelName', 'location', 'roomType',
               'checkInDate', 'checkOutDate', 'totalNights', 'numGuests', 'pricePerNight',
               'totalCost', 'bookingStatus', 'paymentStatus']

    hotel_stay_df = pd.DataFrame(hotel_stays, columns=columns)
    hotel_stay_df.to_csv('D:/Flask_tutorial/mytrip/model/hotel_stay(old).csv', index=False)
    return hotel_stay_df

if __name__ == "__main__":
    # car_rental_df = generate_car_rental_data(num_records)
    # pax_df = generate_passenger_profile_data(num_records)
    # guest_df = generate_guest_profile_data(num_records)
    hotel_stay_df = generate_hotel_stay_data(num_records)

    print("Data generation complete and CSV files saved.")
