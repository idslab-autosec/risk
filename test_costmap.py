import numpy as np
import matplotlib.pyplot as plt
from risk import generate_scene_cost, create_common_grid
from record import Record

def visualize_costmap():
    # Create Record object and read data
    record = Record(1)  # Use the first dataset
    frame_id = 2  # Select the 100th frame for testing
    
    # Get all vehicle data for this frame
    all_vehicle_data = record.get_all_frame_vehicle(frame_id)
    ego_id = 2  # Assume ego_id is 1
    
    # Read background image
    img = plt.imread(record.bg_path)
    img_height, img_width = img.shape[:2]
    
    # Set axis range and scaling ratio
    ratio = 0.10106 * 4  # Same scaling ratio as used in draw_frame_risk

    # Create grid
    x_min, x_max = 0, 1001
    y_min, y_max = 0, 101
    resolution = 1
    grid_x, grid_y = create_common_grid(x_min, x_max, y_min, y_max, resolution)
    
    # Calculate costmap
    costmap = generate_scene_cost(grid_x, grid_y, all_vehicle_data, ego_id)
    
    plt.figure(figsize=(16, 12))
    
    # load background image
    img = plt.imread(record.bg_path)
    img_height, img_width = img.shape[:2]
    
    # costmap
    plt.pcolormesh(grid_x/ratio, grid_y/ratio, costmap, cmap='jet', shading='auto')
    plt.colorbar(label='Cost')
    
    
    # draw vehicles
    for vehicle in all_vehicle_data:
        vid, x, y = vehicle[0], vehicle[1], vehicle[2]
        heading = vehicle[7]
        length, width = vehicle[5], vehicle[6]
        
        # draw ego vehicle
        if vid == ego_id:
            plt.plot(x/ratio, y/ratio, 'wo', markersize=8)
        # else:
        #     plt.plot(x/ratio, y/ratio, 'wo', markersize=4)
        
        # draw rect
        # rect = plt.Rectangle(((x-length/2)/ratio, (y-width/2)/ratio), 
        #                     length/ratio, 
        #                     width/ratio,
        #                     0,
        #                     fill=False, color='w' if vid != ego_id else 'k')
        # plt.gca().add_patch(rect)
    
    plt.title(f'Scene Cost Map (Frame {frame_id})')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    
   
    plt.xlim(0, img_width)
    plt.ylim(img_height, 0)  # 翻转y轴
    
   
    plt.show()

if __name__ == "__main__":
    visualize_costmap()