import json
import sys
import time
sys.path.append("./")
from radar_points import * 
import preprocess

# filenames
radcam_log = "data/radcam_log.json"
test1 = "data/Control_test1.json"
test2 = "data/Test2_tlv_data_log.json"

"""
visually view data per frame at specified interval for 
different loading functions
"""
def test_load(filename: str, func: str, interval=30) -> None:
    with open(filename, "r") as f:
        data = json.load(f)
    if func == "load_data_tlv":
        radar_data = preprocess.load_data_tlv(data)
    elif func == "load_data_mqtt":
        radar_data = preprocess.load_data_mqtt(data)
    elif func == "load_data_sensorhost":
        radar_data = preprocess.load_data_sensorhost(data)
    else:
        raise ValueError("Invalid function name")
    print(f"loading data with {func}...")
    print(radar_data)
    runs = 0
    while radar_data.has_data():
        radar_frame = radar_data.take_next_frame(interval)
        print(radar_frame)
        # time.sleep(0.1)
        runs += 1
    else:
        print("end of data")
        print(radar_data)
        print(f"ran {runs} frames")

def main():
    # test_load(filename="data/Test2_tlv_data_log.json", func="load_data_tlv")
    # test_load(filename="data/Control_test1.json", func="load_data_sensorhost")
    test_load(filename="data/radcam_log.json", func="load_data_mqtt")
if __name__ == "__main__":
    main()