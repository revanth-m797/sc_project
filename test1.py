import csv
import sys
import os
import random
from collections import deque

def parse_time(time_str):
    try:
        time_str = time_str.strip()
        parts = time_str.split(' ')
        time_parts = parts[0].split(':')
        h = int(time_parts[0])
        m = int(time_parts[1])
        meridian = parts[1].upper() if len(parts) > 1 else ""
        if meridian == "PM" and h != 12:
            h += 12
        elif meridian == "AM" and h == 12:
            h = 0
        return h * 60 + m
    except:
        return 0

def load_data(filepath):
    graph = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            duration_str = row['Duration']
            try:
                parts = duration_str.split(':')
                if len(parts) == 3:
                    # CSV format often has '00:09:45' for 9 hrs 45 mins
                    hours, mins = int(parts[1]), int(parts[2])
                    dur_mins = hours * 60 + mins
                elif len(parts) == 2:
                    hours, mins = int(parts[0]), int(parts[1])
                    dur_mins = hours * 60 + mins
                else: 
                    dur_mins = 0
            except:
                dur_mins = 0
                
            try:
                cost = float(row['Cost'])
            except:
                cost = 0.0
                
            f_city = row['From'].strip()
            t_city = row['To'].strip()
            
            # Additional keys to make filtering easy
            edge = {
                'id': idx,
                'from': f_city,
                'to': t_city,
                'operator': row['Operator'],
                'duration': dur_mins,
                'cost': cost,
                'departure': row['Departure'],
                'arrival': row['Arrival'],
                'dep_mins': parse_time(row['Departure']),
                'arr_mins': parse_time(row['Arrival'])
            }
            if f_city not in graph:
                graph[f_city] = []
            graph[f_city].append(edge)
    return graph

def find_city_paths(graph, source, dest, max_depth=3):
    queue = deque([[source]])
    paths = []
    while queue:
        path = queue.popleft()
        curr = path[-1]
        
        if curr == dest and len(path) > 1:
            paths.append(path)
            continue
            
        if len(path) < max_depth + 1:
            if curr in graph:
                next_cities = set(e['to'] for e in graph[curr])
                for nxt in next_cities:
                    if nxt not in path:
                        queue.append(path + [nxt])
    return paths

def get_wait_time(edge1, edge2):
    arrive = edge1['arr_mins']
    depart = edge2['dep_mins']
    wait_time = (depart - arrive) % 1440
    return wait_time

# 1. Hybrid Population Initialization
def create_individual_heuristic(graph, city_path, wt_cost=0.5, wt_dur=0.5):
    individual = []
    for i in range(len(city_path)-1):
        c1, c2 = city_path[i], city_path[i+1]
        possible_edges = [e for e in graph[c1] if e['to'] == c2]
        
        # Heuristic: choose best edge based on weighted cost + duration
        if individual:
            prev_edge = individual[-1]
            def cost_fn(e):
                w = get_wait_time(prev_edge, e)
                # Soft penalty for wait times > 120 so the heuristic avoids them
                penalty = 10000 if w > 120 else 0
                return (e['cost'] * wt_cost) + (e['duration'] * wt_dur) + w + penalty
            best = min(possible_edges, key=cost_fn)
        else:
            best = min(possible_edges, key=lambda e: (e['cost'] * wt_cost) + (e['duration'] * wt_dur))
            
        individual.append(best)
    return individual

def create_individual_random(graph, city_path):
    individual = []
    for i in range(len(city_path)-1):
        c1, c2 = city_path[i], city_path[i+1]
        possible_edges = [e for e in graph[c1] if e['to'] == c2]
        individual.append(random.choice(possible_edges))
    return individual

def eval_fitness(individual, wt_cost=0.5, wt_dur=0.5):
    if not individual:
        return 0.0
    total_cost = sum(e['cost'] for e in individual)
    total_dur = sum(e['duration'] for e in individual)
    num_transfers = len(individual) - 1
    
    total_wait = 0
    penalty = 0
    for i in range(num_transfers):
        wait_time = get_wait_time(individual[i], individual[i+1])
        
        # Penalize drastically if wait time is > 120 mins
        if wait_time > 120:
            penalty += 100000 
        
        total_wait += wait_time
        
    total_dur += total_wait
    
    # Applied weights based on preference
    score = (total_cost * wt_cost) + (total_dur * wt_dur) + (num_transfers * 200) + penalty
    
    if score == 0:
        return float('inf')
    return 1000000.0 / score

# Hybrid Smart Crossover
def crossover(ind1, ind2, wt_cost=0.5, wt_dur=0.5):
    cities1 = [ind1[0]['from']] + [e['to'] for e in ind1]
    cities2 = [ind2[0]['from']] + [e['to'] for e in ind2]
    
    intersections = list(set(cities1[1:-1]) & set(cities2[1:-1]))
    if not intersections:
        return ind1, ind2
        
    # Smart Crossover: Score the intersection
    def score_intersection(city):
        idx1 = cities1.index(city)
        idx2 = cities2.index(city)
        
        # The transition would be ind1[:idx1] + ind2[idx2:]
        if idx1 > 0 and idx2 < len(ind2):
            w = get_wait_time(ind1[idx1-1], ind2[idx2])
        else:
            w = 0
            
        c = sum(e['cost'] for e in ind1[:idx1]) + sum(e['cost'] for e in ind2[idx2:])
        d = sum(e['duration'] for e in ind1[:idx1]) + sum(e['duration'] for e in ind2[idx2:])
        
        return w + (wt_cost * c) + (wt_dur * d)
        
    intersect = min(intersections, key=score_intersection)
    
    idx1 = cities1.index(intersect)
    idx2 = cities2.index(intersect)
    
    child1 = ind1[:idx1] + ind2[idx2:]
    child2 = ind2[:idx2] + ind1[idx1:]
    return child1, child2

# Adaptive & Guided Mutation
def mutate(graph, ind, city_paths, mutation_rate, wt_cost=0.5, wt_dur=0.5):
    if not ind: return ind
    
    if random.random() < mutation_rate:
        if random.random() < 0.6: # 60% chance for Guided Mutation on a single edge
            mut_idx = random.randint(0, len(ind)-1)
            c1 = ind[mut_idx]['from']
            c2 = ind[mut_idx]['to']
            possible_edges = [e for e in graph[c1] if e['to'] == c2]
            
            # Guided mutation based on weights
            if wt_cost > wt_dur: # Cost preferred
                best = min(possible_edges, key=lambda e: e['cost'])
            elif wt_dur > wt_cost: # Dur preferred
                best = min(possible_edges, key=lambda e: e['duration'])
            else: # Balanced
                if random.random() < 0.5:
                    best = min(possible_edges, key=lambda e: e['cost'])
                else:
                    best = min(possible_edges, key=lambda e: e['duration'])
                
            ind[mut_idx] = best
        else:
            # 40% chance to Reseed randomly from valid physical paths
            if not city_paths: return ind
            cp = random.choice(city_paths)
            if random.random() < 0.5:
                return create_individual_heuristic(graph, cp, wt_cost, wt_dur)
            else:
                return create_individual_random(graph, cp)
    return ind

def print_route(route, fitness):
    print(f"Fitness: {fitness:.4f}")
    total_cost = sum(e['cost'] for e in route)
    total_dur_travel = sum(e['duration'] for e in route)
    
    total_wait = 0
    num_transfers = len(route) - 1
    for i in range(num_transfers):
        wait_time = get_wait_time(route[i], route[i+1])
        total_wait += wait_time
        
    total_dur = total_dur_travel + total_wait
    
    print(f"Total Transfers: {len(route) - 1} | Cost: {total_cost} INR | Duration: {total_dur//60}h {total_dur%60}m")
    warning_flag = False
    for i, e in enumerate(route):
        if i > 0:
            wait_time = get_wait_time(route[i-1], route[i])
            if wait_time > 120:
                warning = " (⚠️ WARNING: wait time > 120 mins)"
                warning_flag = True
            else:
                warning = ""
            print(f"  [Wait Time at {e['from']}: {wait_time//60}h {wait_time%60}m{warning}]")
        print(f"  Step {i+1}: {e['from']} -> {e['to']} | Operator: {e['operator']} | Depart: {e['departure']} - Arrive: {e['arrival']} | Cost: {e['cost']}")
    
    if warning_flag:
         print(f"  * Note: This route has wait times > 120m which heavily hurt its fitness score. It's listed because no better paths could be generated without this wait constraint.")
    print("-" * 60)

def main(source, dest, pref):
    # Lock the seed so the "randomness" is strictly identical on every run
    # This completely eliminates the "same input giving different outputs" problem
    random.seed(42)
    
    print("Loading bus routes...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'bus_route.csv')
    
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return
        
    graph = load_data(csv_path)
    
    all_cities = set(graph.keys())
    for edges in graph.values():
        for e in edges:
            all_cities.add(e['to'])
            
    lower_map = {c.lower(): c for c in all_cities}
    
    try:
        # source_in = input("Enter Source City: ").strip().lower()
        # dest_in = input("Enter Destination City: ").strip().lower()
        # pref_in = input("Optimization Preference (cost/duration/leave empty for balanced): ").strip().lower()
        source_in = source.strip().lower()
        dest_in = dest.strip().lower()
        pref_in =pref.strip().lower()
    except EOFError:
        return
        
    wt_cost, wt_dur = 0.5, 0.5
    if pref_in == 'cost':
        wt_cost, wt_dur = 1.0, 0.0
        print("Optimization set heavily for: LOWEST COST")
    elif pref_in == 'duration':
        wt_cost, wt_dur = 0.0, 1.0
        print("Optimization set heavily for: SHORTEST DURATION")
    else:
        print("Optimization set for: BALANCED (Cost & Duration)")
        
    if source_in not in lower_map:
        print(f"Error: Could not find source city '{source_in}' in the dataset.")
        return
    if dest_in not in lower_map:
        print(f"Error: Could not find destination city '{dest_in}' in the dataset.")
        return
        
    source = lower_map[source_in]
    dest = lower_map[dest_in]
    
    if source not in graph:
        print(f"Error: '{source}' has no outgoing buses.")
        return
        
    print(f"Searching for possible paths from {source} to {dest}...")
    city_paths = find_city_paths(graph, source, dest, max_depth=3)
    if not city_paths:
        print(f"No possible routes found from {source} to {dest} within 3 transfers.")
        return
    
    pop_size = 200
    generations = 1000
    
    # Sort city paths by number of hops to ensure direct routes are evaluated in the initial population
    city_paths.sort(key=len)
    population = []
    
    heuristic_count = int(pop_size * 0.70)
    
    # Populate the heuristic chunk
    for cp in city_paths:
        for _ in range(3): # Spread across multiple paths
            population.append(create_individual_heuristic(graph, cp, wt_cost, wt_dur))
            if len(population) >= heuristic_count: break
        if len(population) >= heuristic_count: break
        
    # Fill remaining heuristic quota if not enough short paths
    while len(population) < heuristic_count:
        population.append(create_individual_heuristic(graph, random.choice(city_paths), wt_cost, wt_dur))
        
    # Populate random chunk
    while len(population) < pop_size:
        population.append(create_individual_random(graph, random.choice(city_paths)))
        
    unique_bests = []
    
    for gen in range(generations):
        # Adaptive Mutation Rate
        mutation_rate = 0.3 * (1 - gen / generations)
        
        pop_fit = [(ind, eval_fitness(ind, wt_cost, wt_dur)) for ind in population]
        pop_fit.sort(key=lambda x: x[1], reverse=True)
        
        for p, f in pop_fit:
            route_sig = tuple((e['operator'], e['departure'], e['arrival']) for e in p)
            if not any(route_sig == tuple((ue['operator'], ue['departure'], ue['arrival']) for ue in up) for up, uf in unique_bests):
                unique_bests.append((p, f))
                unique_bests.sort(key=lambda x: x[1], reverse=True)
                unique_bests = unique_bests[:10]
                
        selected = []
        for _ in range(pop_size):
            # Tournament selection with K=5
            participants = random.sample(pop_fit, k=5)
            winner = max(participants, key=lambda x: x[1])
            selected.append(winner[0])
            
        next_population = []
        next_population.append(pop_fit[0][0])
        next_population.append(pop_fit[1][0])
        
        while len(next_population) < pop_size:
            p1 = random.choice(selected)
            p2 = random.choice(selected)
            
            c1, c2 = p1[:], p2[:]
            
            if random.random() < 0.7:
                c1, c2 = crossover(c1, c2, wt_cost, wt_dur)
                
            c1 = mutate(graph, c1, city_paths, mutation_rate, wt_cost, wt_dur)
            c2 = mutate(graph, c2, city_paths, mutation_rate, wt_cost, wt_dur)
                
            next_population.append(c1)
            if len(next_population) < pop_size:
                next_population.append(c2)
                
        population = next_population
        
    print("\n" + "="*60)
    print("🏆 TOP 3 OPTIMIZED ROUTES FOUND 🏆")
    print("="*60)
    
    to_print = min(3, len(unique_bests))
    for i in range(to_print):
        route, fitness = unique_bests[i]
        print(f"\nRoute #{i+1}")
        print_route(route, fitness)

    results = []

    to_print = min(3, len(unique_bests))
    for i in range(to_print):
        route, fitness = unique_bests[i]

        total_cost = sum(e['cost'] for e in route)
        total_dur_travel = sum(e['duration'] for e in route)

        total_wait = 0
        for j in range(len(route)-1):
            total_wait += get_wait_time(route[j], route[j+1])

        total_duration = total_dur_travel + total_wait
        transfers = len(route) - 1

        route_info = []
        for e in route:
            route_info.append({
                "from": e['from'],
                "to": e['to'],
                "operator": e['operator'],
                "departure": e['departure'],
                "arrival": e['arrival'],
                "cost": e['cost']
            })

        results.append({
            "route_no": i+1,
            "fitness": round(fitness, 2),
            "total_cost": total_cost,
            "total_duration": f"{total_duration//60}h {total_duration%60}m",
            "transfers": transfers,
            "steps": route_info
        })

    return results
