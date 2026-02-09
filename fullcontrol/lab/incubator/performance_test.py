#!/usr/bin/env python3
"""
FullControl Performance Test

This script tests the performance of different approaches for:
1. Moving points by editing X values (+1)
2. Calculating distances between neighboring points

Three approaches are tested:
- FullControl Point objects
- NumPy arrays of shape 3*n (30,000 elements)
- NumPy arrays of shape n*3 (10,000 x 3)

Each operation is timed 5 times for statistical reliability.
"""

import time
import random
import numpy as np
import statistics
import fullcontrol as fc

def generate_random_points(n=10000):
    """Generate n random FullControl Points between (0,0,0) and (1,1,1)"""
    points = []
    for _ in range(n):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        z = random.uniform(0, 1)
        points.append(fc.Point(x=x, y=y, z=z))
    return points

def points_to_numpy_3n(points):
    """Convert FullControl Points to numpy array of shape (3*n,)"""
    coords = []
    for point in points:
        coords.extend([point.x, point.y, point.z])
    return np.array(coords)

def points_to_numpy_nx3(points):
    """Convert FullControl Points to numpy array of shape (n, 3)"""
    coords = []
    for point in points:
        coords.append([point.x, point.y, point.z])
    return np.array(coords)

def numpy_3n_to_points(arr):
    """Convert numpy array of shape (3*n,) back to FullControl Points"""
    points = []
    for i in range(0, len(arr), 3):
        points.append(fc.Point(x=arr[i], y=arr[i+1], z=arr[i+2]))
    return points

def numpy_nx3_to_points(arr):
    """Convert numpy array of shape (n, 3) back to FullControl Points"""
    points = []
    for row in arr:
        points.append(fc.Point(x=row[0], y=row[1], z=row[2]))
    return points

def time_point_movement(points, repeats=5):
    """Time moving FullControl Points by adding 1 to X coordinate"""
    times = []
    
    for _ in range(repeats):
        # Create a copy for each test
        test_points = [fc.Point(x=p.x, y=p.y, z=p.z) for p in points]
        
        start_time = time.perf_counter()
        
        # Move points by editing X values
        for point in test_points:
            point.x += 1
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return times

def time_numpy_3n_movement(arr, repeats=5):
    """Time moving numpy array (3*n,) by adding 1 to X coordinates"""
    times = []
    
    for _ in range(repeats):
        # Create a copy for each test
        test_arr = arr.copy()
        
        start_time = time.perf_counter()
        
        # Move X coordinates (every 3rd element starting from 0)
        test_arr[::3] += 1
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return times

def time_numpy_nx3_movement(arr, repeats=5):
    """Time moving numpy array (n, 3) by adding 1 to X coordinates"""
    times = []
    
    for _ in range(repeats):
        # Create a copy for each test
        test_arr = arr.copy()
        
        start_time = time.perf_counter()
        
        # Move X coordinates (first column)
        test_arr[:, 0] += 1
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return times

def time_point_distances(points, repeats=5):
    """Time calculating distances between neighboring FullControl Points"""
    times = []
    
    for _ in range(repeats):
        start_time = time.perf_counter()
        
        # Calculate distances between neighboring points
        distances = []
        for i in range(len(points) - 1):
            dist = fc.distance(points[i], points[i + 1])
            distances.append(dist)
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return times

def time_numpy_3n_distances(arr, repeats=5):
    """Time calculating distances between neighboring points in numpy array (3*n,)"""
    times = []
    
    for _ in range(repeats):
        start_time = time.perf_counter()
        
        # Calculate distances between neighboring points
        distances = []
        for i in range(0, len(arr) - 3, 3):
            # Extract coordinates for two consecutive points
            x1, y1, z1 = arr[i], arr[i+1], arr[i+2]
            x2, y2, z2 = arr[i+3], arr[i+4], arr[i+5]
            
            # Calculate Euclidean distance
            dist = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
            distances.append(dist)
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return times

def time_numpy_nx3_distances(arr, repeats=5):
    """Time calculating distances between neighboring points in numpy array (n, 3)"""
    times = []
    
    for _ in range(repeats):
        start_time = time.perf_counter()
        
        # Calculate distances between neighboring points using vectorized operations
        diff = arr[1:] - arr[:-1]  # Differences between consecutive points
        distances = np.sqrt(np.sum(diff**2, axis=1))  # Euclidean distances
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return times

def time_numpy_to_points_conversion(arr, repeats=5):
    """Time converting numpy array (n, 3) to FullControl Points"""
    times = []
    
    for _ in range(repeats):
        start_time = time.perf_counter()
        
        # Convert numpy array to FullControl Points
        points = numpy_nx3_to_points(arr)
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return times

def time_points_to_numpy_conversion(points, repeats=5):
    """Time converting FullControl Points to numpy array (n, 3)"""
    times = []
    
    for _ in range(repeats):
        start_time = time.perf_counter()
        
        # Convert FullControl Points to numpy array
        arr = points_to_numpy_nx3(points)
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return times

def time_hybrid_workflow_fast_distance(points, repeats=5):
    """Time a hybrid workflow: Points -> NumPy -> fast distance calc -> back to Points"""
    times = []
    
    for _ in range(repeats):
        start_time = time.perf_counter()
        
        # Convert to numpy for fast distance calculation
        arr = points_to_numpy_nx3(points)
        
        # Fast vectorized distance calculation
        diff = arr[1:] - arr[:-1]
        distances = np.sqrt(np.sum(diff**2, axis=1))
        
        # Convert distances back to a list for compatibility
        distance_list = distances.tolist()
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return times

def time_fullcontrol_point_generation(n=10000, repeats=5):
    """Time generating FullControl Points"""
    times = []
    
    for _ in range(repeats):
        start_time = time.perf_counter()
        
        # Generate random points
        points = []
        for _ in range(n):
            x = random.uniform(0, 1)
            y = random.uniform(0, 1)
            z = random.uniform(0, 1)
            points.append(fc.Point(x=x, y=y, z=z))
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return times

def time_numpy_nx3_generation(n=10000, repeats=5):
    """Time generating NumPy n×3 array"""
    times = []
    
    for _ in range(repeats):
        start_time = time.perf_counter()
        
        # Generate random coordinates directly as numpy array
        coords = np.random.uniform(0, 1, (n, 3))
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return times

def time_numpy_3n_generation(n=10000, repeats=5):
    """Time generating NumPy 3×n array"""
    times = []
    
    for _ in range(repeats):
        start_time = time.perf_counter()
        
        # Generate random coordinates as 3×n array
        coords = np.random.uniform(0, 1, (3, n))
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return times

def print_statistics(times, operation_name):
    """Print timing statistics"""
    mean_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0
    min_time = min(times)
    max_time = max(times)
    
    print(f"\n{operation_name}:")
    print(f"  Mean time: {mean_time:.6f} seconds")
    print(f"  Std dev:   {std_time:.6f} seconds")
    print(f"  Min time:  {min_time:.6f} seconds")
    print(f"  Max time:  {max_time:.6f} seconds")
    print(f"  All times: {[f'{t:.6f}' for t in times]}")

def main():
    print("FullControl Performance Test")
    print("=" * 50)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    n = 10000
    print(f"Generating {n} random points between (0,0,0) and (1,1,1)...")
    
    # Generate random points
    points = generate_random_points(n)
    
    # Convert to different formats
    print("Converting to different data structures...")
    numpy_3n = points_to_numpy_3n(points)
    numpy_nx3 = points_to_numpy_nx3(points)
    
    print(f"Data structure sizes:")
    print(f"  FullControl Points: {len(points)} objects")
    print(f"  NumPy 3*n array: {numpy_3n.shape} ({numpy_3n.size} elements)")
    print(f"  NumPy n*3 array: {numpy_nx3.shape} ({numpy_nx3.size} elements)")
    
    # Test generation operations
    print("\n" + "=" * 50)
    print("GENERATION OPERATIONS (creating 10,000 random points)")
    print("=" * 50)
    
    print("Testing FullControl Point generation...")
    fc_gen_times = time_fullcontrol_point_generation(n)
    print_statistics(fc_gen_times, "FullControl Point Generation")
    
    print("\nTesting NumPy n×3 array generation...")
    numpy_nx3_gen_times = time_numpy_nx3_generation(n)
    print_statistics(numpy_nx3_gen_times, "NumPy n×3 Array Generation")
    
    print("\nTesting NumPy 3×n array generation...")
    numpy_3n_gen_times = time_numpy_3n_generation(n)
    print_statistics(numpy_3n_gen_times, "NumPy 3×n Array Generation")
    
    # Test movement operations
    print("\n" + "=" * 50)
    print("MOVEMENT OPERATIONS (adding 1 to X coordinate)")
    print("=" * 50)
    
    print("Testing FullControl Point movement...")
    point_move_times = time_point_movement(points)
    print_statistics(point_move_times, "FullControl Point Movement")
    
    print("\nTesting NumPy 3*n array movement...")
    numpy_3n_move_times = time_numpy_3n_movement(numpy_3n)
    print_statistics(numpy_3n_move_times, "NumPy 3*n Array Movement")
    
    print("\nTesting NumPy n*3 array movement...")
    numpy_nx3_move_times = time_numpy_nx3_movement(numpy_nx3)
    print_statistics(numpy_nx3_move_times, "NumPy n*3 Array Movement")
    
    # Test distance calculations
    print("\n" + "=" * 50)
    print("DISTANCE CALCULATIONS (between neighboring points)")
    print("=" * 50)
    
    print("Testing FullControl Point distances...")
    point_dist_times = time_point_distances(points)
    print_statistics(point_dist_times, "FullControl Point Distances")
    
    print("\nTesting NumPy 3*n array distances...")
    numpy_3n_dist_times = time_numpy_3n_distances(numpy_3n)
    print_statistics(numpy_3n_dist_times, "NumPy 3*n Array Distances")
    
    print("\nTesting NumPy n*3 array distances...")
    numpy_nx3_dist_times = time_numpy_nx3_distances(numpy_nx3)
    print_statistics(numpy_nx3_dist_times, "NumPy n*3 Array Distances")
    
    # Summary comparison
    # Test conversion operations
    print("\n" + "=" * 50)
    print("CONVERSION OPERATIONS")
    print("=" * 50)
    
    print("Testing NumPy -> FullControl conversion...")
    numpy_to_points_times = time_numpy_to_points_conversion(numpy_nx3)
    print_statistics(numpy_to_points_times, "NumPy -> FullControl Points Conversion")
    
    print("\nTesting FullControl -> NumPy conversion...")
    points_to_numpy_times = time_points_to_numpy_conversion(points)
    print_statistics(points_to_numpy_times, "FullControl Points -> NumPy Conversion")
    
    print("\nTesting hybrid workflow (Points -> NumPy -> fast distance -> result)...")
    hybrid_times = time_hybrid_workflow_fast_distance(points)
    print_statistics(hybrid_times, "Hybrid Workflow: Fast Distance via NumPy")
    
    print("\n" + "=" * 50)
    print("PERFORMANCE SUMMARY")
    print("=" * 50)
    
    print("\nGeneration Operations (mean time):")
    print(f"  FullControl Points: {statistics.mean(fc_gen_times):.6f} seconds")
    print(f"  NumPy 3*n array:    {statistics.mean(numpy_3n_gen_times):.6f} seconds")
    print(f"  NumPy n*3 array:    {statistics.mean(numpy_nx3_gen_times):.6f} seconds")
    
    print("\nMovement Operations (mean time):")
    print(f"  FullControl Points: {statistics.mean(point_move_times):.6f} seconds")
    print(f"  NumPy 3*n array:    {statistics.mean(numpy_3n_move_times):.6f} seconds")
    print(f"  NumPy n*3 array:    {statistics.mean(numpy_nx3_move_times):.6f} seconds")
    
    print("\nDistance Calculations (mean time):")
    print(f"  FullControl Points: {statistics.mean(point_dist_times):.6f} seconds")
    print(f"  NumPy 3*n array:    {statistics.mean(numpy_3n_dist_times):.6f} seconds")
    print(f"  NumPy n*3 array:    {statistics.mean(numpy_nx3_dist_times):.6f} seconds")
    
    print("\nConversion Operations (mean time):")
    print(f"  NumPy -> Points:    {statistics.mean(numpy_to_points_times):.6f} seconds")
    print(f"  Points -> NumPy:    {statistics.mean(points_to_numpy_times):.6f} seconds")
    print(f"  Hybrid workflow:    {statistics.mean(hybrid_times):.6f} seconds")
    
    # Calculate speedup ratios
    print("\nSpeedup ratios (relative to FullControl Points):")
    
    # Generation speedup ratios
    fc_gen_mean = statistics.mean(fc_gen_times)
    numpy_3n_gen_mean = statistics.mean(numpy_3n_gen_times)
    numpy_nx3_gen_mean = statistics.mean(numpy_nx3_gen_times)
    
    print(f"Generation - NumPy 3*n: {fc_gen_mean / numpy_3n_gen_mean:.2f}x faster")
    print(f"Generation - NumPy n*3: {fc_gen_mean / numpy_nx3_gen_mean:.2f}x faster")
    
    # Movement speedup ratios
    fc_move_mean = statistics.mean(point_move_times)
    numpy_3n_move_mean = statistics.mean(numpy_3n_move_times)
    numpy_nx3_move_mean = statistics.mean(numpy_nx3_move_times)
    
    print(f"Movement - NumPy 3*n: {fc_move_mean / numpy_3n_move_mean:.2f}x faster")
    print(f"Movement - NumPy n*3: {fc_move_mean / numpy_nx3_move_mean:.2f}x faster")
    
    fc_dist_mean = statistics.mean(point_dist_times)
    numpy_3n_dist_mean = statistics.mean(numpy_3n_dist_times)
    numpy_nx3_dist_mean = statistics.mean(numpy_nx3_dist_times)
    hybrid_mean = statistics.mean(hybrid_times)
    
    print(f"Distance - NumPy 3*n: {fc_dist_mean / numpy_3n_dist_mean:.2f}x faster")
    print(f"Distance - NumPy n*3: {fc_dist_mean / numpy_nx3_dist_mean:.2f}x faster")
    print(f"Distance - Hybrid:    {fc_dist_mean / hybrid_mean:.2f}x faster")
    
    print("\n" + "=" * 50)
    print("PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    print("""
RELATIVE PERFORMANCE COMPARISON FROM PRIOR TESTS (NOT THE ABOVE RESULTS):

Generation Operations (creating 10,000 random points):
  - NumPy n×3 format: ~1000x faster than FullControl Points
  - NumPy 3×n format: ~1000x faster than FullControl Points
  - Both NumPy formats are similar for generation (vectorized random number generation)

Movement Operations (coordinate modification):
  - NumPy n×3 format: ~600x faster than FullControl Points
  - NumPy 3×n format: ~350x faster than FullControl Points
  - NumPy n×3 is ~1.7x faster than 3×n format

Distance Calculations (neighboring point distances):
  - NumPy n×3 format: ~18x faster than FullControl Points
  - NumPy 3×n format: ~0.2x speed (actually slower due to Python loops)
  - FullControl Points: ~4.5x faster than NumPy 3×n format

Conversion Overhead:
  - Points → NumPy: relatively fast conversion
  - NumPy → Points: ~20x slower conversion (expensive object creation)
  - Hybrid workflow: still faster than pure FullControl for bulk operations

Key Insights:
  - Generation: NumPy dramatically faster due to vectorized random number generation
  - N×3 format (rows=points, columns=coordinates) enables full vectorization and is the industry standard for point data in most applications. 3×N requires manual indexing loops
  - Conversion costs are significant but worthwhile for large datasets
  - Performance gains are most dramatic for bulk mathematical operations
""")

if __name__ == "__main__":
    main()