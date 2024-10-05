# This is the code for the PID controller of the BBT.
from acrome.controller import *  # Import the ACROME controller
import time
from maze import getSolution

# Coordinate transformation function
def maze_to_bbt_coords(maze_pos):
    x, y = maze_pos
    BBT_x = (x / 38) * 360 - 191
    BBT_y = (y / 29) * 250 - 111
    return BBT_x, BBT_y

def convert_path_to_lines(path):
    if not path:
        return []

    lines = []
    start_point = path[0]
    lines.append(start_point)

    for i in range(1, len(path)):
        if path[i][0] != start_point[0] and path[i][1] != start_point[1]:
            lines.append(path[i-1])
            start_point = path[i]

    # Add the last line move
    lines.append(path[-1])

    return lines

def move_bbt_to_position(setpointx, setpointy):
    # Main Loop
    error_sum_x = 0
    error_sum_y = 0
    err_prev = (0, 0)
    dev = BallBalancingTable('COM4')
    interval = 0.005

    _filter_size = 27
    bbt_iter = 0
    x_mov_avg_filter = [0 for i in range(_filter_size)]
    y_mov_avg_filter = [0 for i in range(_filter_size)]
    outputx = 0
    outputy = 0
    # Control Parameters
    windup_abs = 25
    calibration_x = (0, 180)
    calibration_y = (0, 252)

    kpx = 1.3
    kix = 0.3
    kdx = 1.4

    kpy = 1.3
    kiy = 0.3
    kdy = 1.4

    timeout = 1  # Timeout in seconds
    start_time = time.time()

    print_counter = 0  # Initialize print counter
    print_interval = 50000  # Number of iterations between prints

    while True:
        positionx, positiony = dev.position  # Get the current position

        positionx = 1000 if positionx > 1000 else positionx
        positiony = 1000 if positiony > 1000 else positiony

        positionx = (dev.position[0] - 106) / 2.375 - (344 / 2)  # PositionX (Analog to mm)
        positiony = (dev.position[1] - 1) / 3.169 - (272 / 2)  # PositionY (Analog to mm)
        error = ((setpointx - positionx), (setpointy - positiony))
        error_sum_x += error[0]
        error_sum_y += error[1]

        # PID for X axis
        px = error[0] * kpx
        ix = error_sum_x * kix
        ix = ix if -windup_abs <= ix <= windup_abs else (ix / abs(ix)) * windup_abs

        x_mov_avg_filter[bbt_iter] = (error[0] - err_prev[0]) * kdx / interval
        dx = sum(x_mov_avg_filter[0:_filter_size]) / _filter_size

        # PID for Y axis
        py = error[1] * kpy
        iy = error_sum_y * kiy
        iy = iy if -windup_abs <= iy <= windup_abs else (iy / abs(iy)) * windup_abs

        y_mov_avg_filter[bbt_iter] = (error[1] - err_prev[1]) * kdy / interval
        dy = sum(y_mov_avg_filter[0:_filter_size]) / _filter_size

        bbt_iter = (bbt_iter + 1) % _filter_size

        if abs(error_sum_x) > 10:
            outputx = px + ix + dx

        if abs(error_sum_y) > 10:
            outputy = py + iy + dy

        outputx += calibration_x[0] * positionx + calibration_x[1]
        outputy += calibration_y[0] * positiony + calibration_y[1]

        # Lower limit for servo
        outputx = -1000 if outputx <= -1000 else outputx
        outputy = -1000 if outputy <= -1000 else outputy

        # Upper limit for servo
        outputx = 1000 if outputx >= 1000 else outputx
        outputy = 1000 if outputy >= 1000 else outputy
        dev.set_servo(int(outputx), int(outputy))  # Set the servo to the output
        dev.update()
        currentX, currentY = (dev.position[0] - 106) / 2.375 - (344 / 2), (dev.position[1] - 1) / 3.169 - (272 / 2)

        if time.time() - start_time > timeout:
            print("Timeout reached. Moving to the next position.")
            break
        # Update error_prev and continue the loop
        err_prev = error

        if print_counter % print_interval == 0:
            print(f"Current X: {currentX}, Target X: {setpointx}, Current Y: {currentY}, Target Y: {setpointy}")
            print(print_counter)
        print_counter += 1
        time.sleep(interval)


path, mazeSize = getSolution()
lines = convert_path_to_lines(path)

for line in lines:
    print(f"Move from {line}")
    setpointx, setpointy = line[0], line[1]
    target_x_bbt, target_y_bbt = maze_to_bbt_coords([setpointx, setpointy])
    move_bbt_to_position(target_x_bbt, target_y_bbt)