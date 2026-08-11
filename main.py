import random
from tokenize import Triple

from data.hotel import Hotel
from data.transport import Transport, TransportType
from optimizer.trip_optimizer_ga import TripOptimizerGA


def generate_mock_data():
    locations = ['Home', 'Paris', 'Berlin', 'Rome']
    destinations = ['Paris', 'Berlin', 'Rome']
    transports = []

    # Fully connect all locations for 14 days
    t_id = 1
    for day in range(14):
        for orig in locations:
            for dest in locations:
                if orig != dest:
                    # Plane
                    transports.append(
                        Transport(t_id, orig, dest, day, price=random.uniform(30, 200), duration=120,
                                  co2_kg=15.0, type=TransportType.PLANE))
                    t_id += 1

                    # Train
                    transports.append(
                        Transport(t_id, orig, dest, day, price=random.uniform(50, 200), duration=360,
                                  co2_kg=15.0, type=TransportType.TRAIN))
                    t_id += 1


    hotels = []
    for dest in destinations:
        # Expensive Hotels
        prices_A = {day: random.uniform(120, 150) for day in range(14)}
        hotels.append(Hotel(f"H_{dest}_Luxury", dest, prices_A, 3))
        # Cheap Hotels
        prices_B = {day: random.uniform(50, 100) for day in range(14)}
        hotels.append(Hotel(f"H_{dest}_Budget", dest, prices_B, 2))

    return transports, hotels, destinations


if __name__ == "__main__":
    random.seed(42)  # For reproducible mock data
    transports, hotels, destinations = generate_mock_data()
    print(f"Transport Options:\n{transports}")
    print(f"Hotel Options:\n{hotels}")

    print("Running Travel Optimizer Genetic Algorithm...")
    ga = TripOptimizerGA(
        transports=transports,
        hotels=hotels,
        destinations=destinations,
        start_point='Home',
        min_duration=5,
        max_duration=7,
        min_destinations=2,
        pop_size=50,
        generations=50,
        mutation_rate=0.3,
        elitism=3
    )

    best_itinerary = ga.run()

    print("\n" + "=" * 50)
    print("🏆 BEST ITINERARY FOUND 🏆")
    print("=" * 50)

    transport_dict = {t.id: t for t in transports}
    current_day = 0
    total_cost = 0

    for i, step in enumerate(best_itinerary):
        t = transport_dict[step['transport_in']]
        print(
            f"🗓️ Day {current_day}: Take {t.type.value.upper()} from {t.origin} to {t.dest} (Cost: ${t.price:.2f}, CO2: {t.co2_kg}kg, Comfort: {t.comfort}/10)")
        total_cost += t.price

        if step['type'] == 'stop':
            hotel_cost = sum([h.prices[day] for h in hotels if h.id == step['hotel'] for day in
                              range(current_day, current_day + step['duration'])])
            total_cost += hotel_cost
            print(f"   🏨 Check into {step['hotel']} for {step['duration']} nights (Cost for stay: ${hotel_cost:.2f})")
            current_day += step['duration']

    print(f"\n💰 Approximate Total Financial Cost: ${total_cost:.2f}")
    print(f"⚖️ Final GA Fitness Score (including CO2/Comfort penalties): {ga.calculate_fitness(best_itinerary):.2f}")