import sys
import rosbag
from tqdm import tqdm


def merge_chunks(file_in, chunk_size):
    outbag = rosbag.Bag(file_in[:-7] + ".bag", "w")
    for chunk in tqdm(range(chunk_size)):
        chunk_file = file_in[:-6] + "%02d.bag" % chunk
        bagfile = rosbag.Bag(chunk_file)
        msg_size = bagfile.get_message_count()
        for topic, msg_size, t in tqdm(
            bagfile.read_messages(), chunk_file, total=msg_size
        ):
            outbag.write(topic, msg_size, t)
    outbag.close()


if __name__ == "__main__":
    files = sys.argv[1:]

    for file in files:
        merge_chunks(file, 12)
