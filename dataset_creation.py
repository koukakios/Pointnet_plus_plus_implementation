import json
import numpy as np
from pathlib import Path
import h5py
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

class Processor:

    def __init__(self):
        self.input = Path("/space/kkoukakis/RadarScenes_Clutter/RadarScenes/data")
        self.output_train = Path("/space/kkoukakis/RadarScenes_Clutter/train_datasets")
        self.output_eval = Path("/space/kkoukakis/RadarScenes_Clutter/test_datasets")

    def make_datasets(self, sequences = 158):
        with ProcessPoolExecutor(max_workers=8) as executor:
            executor.map(self.process_one_sequence, range(1, sequences + 1))

    def process_one_sequence(self, sequence):
        seq_points, seq_labels, train = self.load_sequence(sequence)
        self.save_sequence(sequence, seq_points, seq_labels, train)

    def save_sequence(self, sequence_id, points, labels, train):
        if train:
            seq_folder = self.output_train/ f"sequence_{sequence_id}"
        else:
            seq_folder = self.output_eval/ f"sequence_{sequence_id}"

        # create folder if it doesn't exist
        seq_folder.mkdir(parents=True, exist_ok=True)

        # save files
        np.save(seq_folder / "points.npy", points)
        np.save(seq_folder / "labels.npy", labels)

    def load_sequence(self, sequence):
        #load hdf5 radar data file into df for easy slicing
        radar = pd.DataFrame(np.array(h5py.File(self.input.joinpath("sequence_" + str(sequence),
                                    "radar_data.h5"), "r+")["radar_data"]))
        radar_data_df = radar[["x_cc", "y_cc", "vr_compensated"]].copy()
        radar_data_df["z_cc"] = 0
        radar_labels_df = radar[["label_id"]]
        #go back to numpy arrays
        radar_data = radar_data_df.to_numpy()
        radar_labels = radar_labels_df.to_numpy()

        #find scenes path
        scenes_path = self.input.joinpath("sequence_" + str(sequence), "scenes.json")

        #open it & turn to json
        with open(scenes_path, "r") as file:
            scenes_dict = json.load(file)

        train = scenes_dict["category"] == "train"

        #get the actual data
        scenes = scenes_dict["scenes"]
        scenes_num = len(scenes)
        sequence_data = np.empty((scenes_num, 512, 4))
        sequence_labels = np.empty((scenes_num, 512, 1))

        count = 0
        for timestamp in scenes:
            content = scenes[timestamp]
            indices = content["radar_indices"]
            points = radar_data[indices[0] : indices[1], :]
            labels = radar_labels[indices[0] : indices[1], :]

            if len(points) < 512:
                idx = np.random.choice(len(points), 512 - len(points), replace=True)
                points = np.concatenate([points, points[idx]], axis=0)
                labels = np.concatenate([labels, labels[idx]], axis = 0)
            elif len(points) > 512:
                idx = np.random.choice(len(points), 512, replace=False)
                points = points[idx]
                labels = labels[idx]

            sequence_data[count] = points
            sequence_labels[count] = labels
            count += 1

        return sequence_data, sequence_labels, train



if __name__ == "__main__":
    processor = Processor()
    processor.make_datasets()

