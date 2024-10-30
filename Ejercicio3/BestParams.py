from itertools import product
import numpy as np
from typing import List, Tuple
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import HTML
from datetime import datetime
import matplotlib.animation as animation

class GeneticTSP:
    def __init__(self, population_size: int = 100, 
                generations: int = 1000,
                mutation_rate: float = 0.01,
                elite_size: int = 10):
        """
        Inicializa el algoritmo genético para TSP
        
        Args:
            population_size: Tamaño de la población
            generations: Número de generaciones
            mutation_rate: Tasa de mutación
            elite_size: Número de mejores individuos que pasan directamente a la siguiente generación
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.best_solutions_history = []


    def _calculate_distance(self, city1: Tuple[float, float], 
                          city2: Tuple[float, float]) -> float:
        """Calcula la distancia euclidiana entre dos ciudades"""
        return np.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)
    
    def _calculate_tour_distance(self, tour: List[int], 
                                cities: List[Tuple[float, float]]) -> float:
        """Calcula la distancia total de un tour"""
        distance = 0
        for i in range(len(tour)):
            from_city = cities[tour[i]]
            to_city = cities[tour[(i + 1) % len(tour)]]
            distance += self._calculate_distance(from_city, to_city)
        return distance
    
    def _create_initial_population(self, num_cities: int) -> List[List[int]]:
        """Crea una población inicial de tours aleatorios"""
        population = []
        for _ in range(self.population_size):
            tour = list(range(num_cities))
            random.shuffle(tour)
            population.append(tour)
        return population
    
    def _select_parent(self, fitness_scores: List[float]) -> List[int]:
        """Selecciona un padre usando selección por torneo"""
        tournament_size = 5
        tournament = random.sample(range(len(fitness_scores)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament]
        return tournament[tournament_fitness.index(min(tournament_fitness))]
    
    def _crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Realiza el cruce ordenado (OX) entre dos padres"""
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        # Obtener la subsecuencia del primer padre
        child = [-1] * size
        for i in range(start, end + 1):
            child[i] = parent1[i]
        
        # Completar con los elementos del segundo padre
        remaining = [x for x in parent2 if x not in child[start:end+1]]
        j = 0
        for i in range(size):
            if child[i] == -1:
                child[i] = remaining[j]
                j += 1
                
        return child
    
    def _mutate(self, tour: List[int]) -> List[int]:
        """Aplica mutación por intercambio si se cumple la probabilidad"""
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(tour)), 2)
            tour[i], tour[j] = tour[j], tour[i]
        return tour
    
    def solve(self, cities: List[Tuple[float, float]]) -> Tuple[List[int], float]:
        """
        Resuelve el problema TSP usando algoritmo genético
        
        Args:
            cities: Lista de tuplas con las coordenadas (x, y) de cada ciudad
            
        Returns:
            Tuple[List[int], float]: Mejor tour encontrado y su distancia
        """
        num_cities = len(cities)
        population = self._create_initial_population(num_cities)
        best_distance = float('inf')
        best_tour = None
        progress = []

        for generation in range(self.generations):
            fitness_scores = [self._calculate_tour_distance(tour, cities) 
                            for tour in population]
            
            min_distance = min(fitness_scores)
            if min_distance < best_distance:
                best_distance = min_distance
                best_tour = population[fitness_scores.index(min_distance)].copy()
            
            progress.append(best_distance)

            # Guardar la mejor solución cada 10 generaciones
            if generation % 10 == 0:
                self.best_solutions_history.append((best_tour.copy(), best_distance))
            
            sorted_population = [x for _, x in sorted(zip(fitness_scores, population))]
            new_population = sorted_population[:self.elite_size]
            
            while len(new_population) < self.population_size:
                parent1 = population[self._select_parent(fitness_scores)]
                parent2 = population[self._select_parent(fitness_scores)]
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)
            
            population = new_population
            
            if (generation + 1) % 2000 == 0:
                print(f"Generación {generation + 1}: Mejor distancia = {best_distance:.2f}")
        
        
        # Asegurarse de guardar la mejor solución final
        self.best_solutions_history.append((best_tour, best_distance))
        return best_tour, best_distance, progress

def create_tsp_animation(cities: List[Tuple[float, float]], 
                        solution_history: List[Tuple[List[int], float]], 
                        output_file: str = 'tsp_evolution.gif'):
    """
    Crea una animación GIF de la evolución del algoritmo
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    coords = np.array(cities)
    
    def update(frame):
        ax.clear()
        tour, distance = solution_history[frame]
        
        # Dibujar ciudades
        ax.scatter(coords[:, 0], coords[:, 1], c='red', marker='o')
        
        # Dibujar tour
        for i in range(len(tour)):
            start = coords[tour[i]]
            end = coords[tour[(i + 1) % len(tour)]]
            ax.plot([start[0], end[0]], [start[1], end[1]], 'b-', alpha=0.3)
        
        ax.set_title(f'Generación {frame*10}\nDistancia: {distance:.2f}')
        ax.grid(True)
    
    # Crear la animación
    anim = FuncAnimation(
        fig, 
        update, 
        frames=len(solution_history),
        interval=30,  # 200ms entre frames
        repeat=True
    )
    
    # Guardar como GIF
    anim.save(output_file, writer='pillow')
    plt.close()
    
    return anim

def save_final_solution_plot(cities: List[Tuple[float, float]], 
                          best_tour: List[int],
                          best_distance: float,
                          output_file: str = 'best_solution.png'):
    """
    Guarda una imagen de la mejor solución encontrada
    """
    plt.figure(figsize=(10, 10))
    coords = np.array(cities)
    
    # Dibujar ciudades
    plt.scatter(coords[:, 0], coords[:, 1], c='red', marker='o')
    
    # Dibujar mejor tour
    for i in range(len(best_tour)):
        start = coords[best_tour[i]]
        end = coords[best_tour[(i + 1) % len(best_tour)]]
        plt.plot([start[0], end[0]], [start[1], end[1]], 'b-', alpha=0.3)
    
    plt.title(f'Mejor solución encontrada\nDistancia total: {best_distance:.2f}')
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()

# Función para leer el archivo ch150.tsp
def read_tsp_file(filename: str) -> List[Tuple[float, float]]:
    """Lee el archivo TSP y devuelve lista de coordenadas"""
    cities = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        # Saltar el encabezado
        start_reading = False
        for line in lines:
            if line.strip() == "NODE_COORD_SECTION":
                start_reading = True
                continue
            if line.strip() == "EOF":
                break
            if start_reading:
                _, x, y = line.strip().split()
                cities.append((float(x), float(y)))
    return cities

#### MAIN ####

# Leer datos
cities = read_tsp_file("ch150.tsp")

# Definir los parámetros a explorar
param_grid = {
    'population_size': [50, 100, 150],
    'mutation_rate': [0.01, 0.05, 0.10, 0.15],
    'elite_size': [3, 5, 10, 15]
}

# Bucle para probar todas las combinaciones de parámetros
best_result = None
all_results = {}

# Crear archivo de logs
with open("ga_tsp_results.log", "a") as log_file:

    for params in product(*param_grid.values()):
        # Obtener hora y minuto actuales
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        population_size, mutation_rate, elite_size = params

        print(f"Probando configuración: population_size={population_size}, mutation_rate={mutation_rate}, elite_size={elite_size}")
        print(f"Hora actual: {current_time}\n")
        
        # Configurar y ejecutar el algoritmo genético
        ga_tsp = GeneticTSP(
            population_size=population_size,
            generations=10000,  # Mantener constante
            mutation_rate=mutation_rate,
            elite_size=elite_size
        )

        best_tour, best_distance, progress = ga_tsp.solve(cities)

        print(f"Distancia total: {best_distance:.2f}\n")

        # Guardar resultados en all_results
        all_results[params] = best_distance

        # Escribir en el archivo de logs
        log_file.write(f"{current_time} | Configuracion: {params} | Distancia: {best_distance:.2f}\n")

        # Guardar el mejor resultado encontrado
        if not best_result or best_distance < best_result[1]:
            best_result = (best_tour, best_distance, params)


# Mostrar todos los resultados
print("\nResultados de todas las configuraciones:")
for params, distance in all_results.items():
    print(f"Configuración: {params} - Distancia: {distance:.2f}")

# Mostrar la mejor configuración y resultado
print(f"\n\nMejor configuración encontrada: {best_result[2]}")
print(f"Mejor distancia: {best_result[1]:.2f}")
print(f"Ruta: {best_result[0]}")