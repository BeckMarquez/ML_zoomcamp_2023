import requests

url = "http://localhost:9696/predict"

client = {"gender": "male", "customer_type": "loyal_customer", "age": 39,
          "type_of_travel": "business_travel", "class": "business", "flight_distance": 1785, "seat_comfort": 4,
          "departure/arrival_time_convenient": 4, "food_and_drink": 5, "gate_location": 4, "inflight_wifi_service": 5,
          "inflight_entertainment": 5, "online_support": 5, "ease_of_online_booking": 5, "on-board_service": 5,
          "leg_room_service": 4, "baggage_handling": 5, "checkin_service": 5, "cleanliness": 5,
          "online_boarding": 5, "departure_delay_in_minutes": 15, "arrival_delay_in_minutes": 0}


response = requests.post(url, json=client).json()

print(response)