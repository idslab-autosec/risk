import os
import risk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from Constant import RiskConstants

class Record:
    def __init__(self, iRecord):
        base_path = os.path.abspath(os.path.dirname(__file__))
        dataPath = os.path.join(base_path, 'data/highD')
        fileName = f'{dataPath}/{iRecord:02d}_tracks.csv'
        fileNameStatic = f'{dataPath}/{iRecord:02d}_tracksMeta.csv'
        fileNameRecord = f'{dataPath}/{iRecord:02d}_recordingMeta.csv'

        # Load dataset
        self.tracks = pd.read_csv(fileName).values
        self.tracks_meta = pd.read_csv(fileNameStatic).values.tolist()
        self.recording_meta = pd.read_csv(fileNameRecord).values.tolist()

        # Adjust y to be the vehicle center line position
        self.tracks[:, 3] = self.tracks[:, 3] + self.tracks[:, 5] / 2

        self.lane_pos = [
            lane_str2num(self.recording_meta[0][13]),
            lane_str2num(self.recording_meta[0][14])
        ]

        # Calculate cumulative sum of frames for each vehicle
        numFrame = [int(row[5]) for row in self.tracks_meta]
        self.start_row_id = np.cumsum(numFrame)
        self.start_row_id = np.insert(self.start_row_id, 0, 0)

        self.frame_rate = 25
        self.isDebug = False
        self.isInfo = False
        self.bg_path = f'{dataPath}/{iRecord:02d}_highway.jpg'

    def get_id_row(self, id, frame):
        """
        Get the row index for a given vehicle id and frame.
        """
        id = id - 1  # Adjust for Python's zero-based indexing
        if frame == -1:
            return self.start_row_id[id]
        idInitFrame = int(self.tracks_meta[id][3])
        row = self.start_row_id[id] + frame - idInitFrame
        if row < self.start_row_id[id]:
            return self.start_row_id[id]
        elif row >= self.start_row_id[id + 1]:
            return self.start_row_id[id + 1] - 1
        return row

    def get_data(self, id, startFrame, endFrame):
        """
        Get data for a given vehicle id between startFrame and endFrame.
        """
        startRow = self.get_id_row(id, startFrame)
        endRow = self.get_id_row(id, endFrame)
        return self.tracks[startRow:endRow + 1, :]

    def get_data_from_id(self, id):
        """
        Get all data for a given vehicle id.
        """
        if isinstance(id, (list, tuple)):
            print('err: id should be scalar')
        startRow = self.get_id_row(id, -1)
        endRow = self.get_id_row(id + 1, -1) - 1
        return self.tracks[startRow:endRow + 1, :]

    def get_all_frame_vehicle(self, frame_id):
        """
        Get all vehicles' data in a specific frame.
        """
        frame_data = []
        vehicle_ids = np.unique(self.tracks[self.tracks[:, 0] == frame_id, 1])
        for vehicle_id in vehicle_ids:
            vehicle_data = self.get_data(int(vehicle_id), int(frame_id), int(frame_id))
            if vehicle_data.size > 0:
                x = vehicle_data[0, 2]
                y = vehicle_data[0, 3]
                vx = vehicle_data[0, 6]
                vy = vehicle_data[0, 7]
                bbox_length = vehicle_data[0, 4]
                bbox_width = vehicle_data[0, 5]
                heading = 0.01 if vx > 0 else 180.01
                frame_data.append([vehicle_id, x, y, vx, vy, bbox_length, bbox_width, heading])
        return np.array(frame_data)

    def calculate_risk(self, ego_id, frame_id):
        """
        Calculate risk for a specific vehicle (ego) in a specific frame.
        """
        ego_data = self.get_data(ego_id, frame_id, frame_id)
        if ego_data.size == 0:
            raise ValueError('Ego vehicle does not exist in the current frame')

        ego_x = ego_data[0, 2]
        ego_y = ego_data[0, 3]
        ego_vx = ego_data[0, 6]
        ego_vy = ego_data[0, 7]
        ego_speed = np.sqrt(ego_vx ** 2 + ego_vy ** 2)
        ego_length = ego_data[0, 4]
        ego_width = ego_data[0, 5]
        ego_heading = 0.01 if ego_vx > 0 else 180.01

        res = 1
        grid_x = np.arange(0, 1001, res)
        grid_y = np.arange(0, 101, res)
        X, Y = np.meshgrid(grid_x, grid_y)

        delta_fut_h = (np.pi / 180) * RiskConstants.STEERING_ANGLE / RiskConstants.SR
        phiv_a = (np.pi / 180) * ego_heading

        delta = risk.gs_delta(delta_fut_h)
        phiv = risk.gs_phiv(phiv_a)
        dla = risk.gs_dla(RiskConstants.TLA, ego_speed)
        R = risk.gs_R(RiskConstants.L, delta)
        xc, yc = risk.gs_center(ego_x, ego_y, phiv, delta, R)
        mexp1 = risk.gs_mexp(RiskConstants.KEXP1, RiskConstants.MCEXP, delta, ego_speed)
        mexp2 = risk.gs_mexp(RiskConstants.KEXP2, RiskConstants.MCEXP, delta, ego_speed)
        arc_len = risk.gs_arclen(X, Y, ego_x, ego_y, delta, xc, yc, R)
        a = risk.gs_a(arc_len, RiskConstants.PAR1, dla)
        sigma1 = risk.gs_sigma(arc_len, mexp1, RiskConstants.CEXP)
        sigma2 = risk.gs_sigma(arc_len, mexp2, RiskConstants.CEXP)
        z_prob = risk.gs_z(X, Y, xc, yc, R, a, sigma1, sigma2)

        all_vehicles_data = self.get_all_frame_vehicle(frame_id)
        scene_cost = risk.generate_scene_cost(X, Y, all_vehicles_data, ego_id)  # 修改为调用risk.py中的方法
        risk_qrf = np.sum(z_prob)
        
        # return risk_qrf
        # 计算量化感知风险
        z = np.dot(z_prob.flatten(), scene_cost.flatten())
        
        print("risk:{},ego_qrf:{}".format(z, risk_qrf))

        return risk

    def draw_frame_risk(self, frame_id):
        """
        Draw the risk map for a specific frame and save as a PNG file.
        """
        # Constants for drawing
        res = 0.2  # Grid resolution
        grid_x_start, grid_x_end = 0, 1001  # X-axis grid range from 0 to 1000
        grid_y_start, grid_y_end = 0, 101   # Y-axis grid range from 0 to 100
        fig_size = (16, 12)  # Figure size
        dpi = 300  # Image resolution
        ratio = 0.10106 * 4  # Scaling ratio

        all_vehicles_data = self.get_all_frame_vehicle(frame_id)
        if all_vehicles_data.size == 0:
            raise ValueError('No vehicles exist in the current frame')

        img = plt.imread(self.bg_path)
        img_height, img_width = img.shape[:2]

        grid_x = np.arange(grid_x_start, grid_x_end, res)
        grid_y = np.arange(grid_y_start, grid_y_end, res)
        X, Y = np.meshgrid(grid_x, grid_y)
        frame_risk = np.zeros_like(X)

        for vehicle_data in all_vehicles_data:
            vehicle_id, x, y, vx, vy, length, width, heading = vehicle_data
            speed = np.sqrt(vx ** 2 + vy ** 2)
            delta_fut_h = (np.pi / 180) * RiskConstants.STEERING_ANGLE / RiskConstants.SR
            phiv_a = (np.pi / 180) * heading

            delta = risk.gs_delta(delta_fut_h)
            phiv = risk.gs_phiv(phiv_a)
            dla = risk.gs_dla(RiskConstants.TLA, speed)
            R = risk.gs_R(RiskConstants.L, delta)
            xc, yc = risk.gs_center(x, y, phiv, delta, R)
            mexp1 = risk.gs_mexp(RiskConstants.KEXP1, RiskConstants.MCEXP, delta, speed)
            mexp2 = risk.gs_mexp(RiskConstants.KEXP2, RiskConstants.MCEXP, delta, speed)
            arc_len = risk.gs_arclen(X, Y, x, y, delta, xc, yc, R)
            a = risk.gs_a(arc_len, RiskConstants.PAR1, dla)
            sigma1 = risk.gs_sigma(arc_len, mexp1, RiskConstants.CEXP)
            sigma2 = risk.gs_sigma(arc_len, mexp2, RiskConstants.CEXP)
            Z_cur = risk.gs_z(X, Y, xc, yc, R, a, sigma1, sigma2)

            frame_risk += Z_cur

        fig_size = (16, 12)  # Increase figure size
        dpi = 300  # Set high resolution
        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
        ax.imshow(img, extent=[0, img_width, img_height, 0])

        grid_x_scaled = grid_x / ratio
        grid_y_scaled = grid_y / ratio

        plt.contourf(grid_x_scaled, grid_y_scaled, frame_risk, levels=200, cmap='jet', alpha=0.7)
        plt.scatter(all_vehicles_data[:, 1] / ratio, all_vehicles_data[:, 2] / ratio, c='white', s=20)
        plt.axis('off')
        plt.xlim(0, img_width)
        plt.ylim(img_height, 0)

        plt.savefig(f'output_example/{frame_id}.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Finish risk plot draw: {frame_id}.png")

def lane_str2num(lane_str):
    """
    Convert lane string to a list of float values.
    Example: '8.51;12.59;16.43' -> [8.51, 12.59, 16.43]
    """
    lane_pos = list(map(float, lane_str.split(';')))
    return lane_pos

def diff_n(x, n):
    """
    Compute discrete differences.
    The difference interval is chosen as n.
    """
    return x[n:] - x[:-n]

def main():
    # Record(1).draw_frame_risk(1)  # Test one frame
    
    Record(1).calculate_risk(2,1)
    
if __name__ == "__main__":
    main()
