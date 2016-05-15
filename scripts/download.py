import boto
import os
import threading
import multiprocessing
import sys

queuelock = threading.Lock()
count = 0
queue = []


class DownloadThread(threading.Thread):
    def __init__(self, id):
        threading.Thread.__init__(self)
        self.id = id

    def run(self):
        global count
        print '[%2d] Download thread starting' % self.id
        sys.stdout.flush()
        while len(queue) > 0:
            queuelock.acquire()
            key, path, filename = queue.pop()
            queuelock.release()

            print '[%2d] Downloading %s (%.0fMB)' % (self.id, filename, key.size / 1024**2)
            sys.stdout.flush()
            try:
                key.get_contents_to_filename(path)
            except:
                queuelock.acquire()
                queue.append((key, path, filename))
                queuelock.release()
        count -= 1

def download(data_folder='data/raw-mris/'):
    global count
    # Download MRI scan files from AWS S3 bucket
    conn = boto.connect_s3(anon=True)
    bucket = conn.get_bucket('fcp-indi')
    # We want to download from C-PAC pipeline
    for key in bucket.list('data/Projects/ABIDE_Initiative/Outputs/cpac/func_minimal/'):
        filename = os.path.basename(os.path.normpath(key.name))
        # C-PAC pipeline -> minimal pre-processing methodology
        if filename == 'func_minimal':
            continue

        path = os.path.join(data_folder, filename)

        if not os.path.isfile(path) or os.path.getsize(path) != key.size:
            # Add file to queue
            queue.append((key, path, filename))

    for i in xrange(multiprocessing.cpu_count()):
        t = DownloadThread(i)
        t.daemon = True
        t.start()

        count += 1

    while count > 0:
        pass

if __name__ == '__main__':
    download()
