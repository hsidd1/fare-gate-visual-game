import json
import sys
sys.path.append("./")
from radar_points import * 
# validation for test2 output
def test_t2load_tlv():
    import preprocess
    with open("data/Test2_tlv_data_log.json", "r") as f:
        data = json.load(f)
        radar_data = preprocess.load_data_tlv(data)
        print(radar_data)
    interval = 30
    while radar_data.has_data():
        radar_frame = radar_data.take_next_frame(interval)
        print(radar_frame)
    else:
        print("end of data")
        print(radar_data)

def main():
    test_t2load_tlv()

if __name__ == "__main__":
    main()