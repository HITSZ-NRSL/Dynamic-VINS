import math
import sys
import rosbag
from tqdm import tqdm


def extract_chunks(file_in, chunks):
    bagfile = rosbag.Bag(file_in)
    msg_size = bagfile.get_message_count()
    m_per_chunk = math.ceil((float(msg_size) / float(chunks)))
    chunk = 0
    m = 0
    outfile = file_in[:-4] + "_%02d.bag" % chunk
    outbag = rosbag.Bag(outfile, "w")
    with tqdm(desc=outfile, total=msg_size) as pbar:
        for topic, msg_size, t in bagfile.read_messages():
            m += 1
            if m % m_per_chunk == 0:
                outbag.close()
                chunk += 1
                outfile = file_in[:-4] + "_%02d.bag" % chunk
                pbar.desc = outfile
                outbag = rosbag.Bag(outfile, "w")
            outbag.write(topic, msg_size, t)
            pbar.update(1)
    outbag.close()


if __name__ == "__main__":
    files = sys.argv[1:]

    for file in files:
        extract_chunks(file, 12)
