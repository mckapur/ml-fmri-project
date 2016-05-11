import boto
import os

def download(data_folder='data/'):
    conn = boto.connect_s3(anon=True)
    bucket = conn.get_bucket('fcp-indi')
    for key in bucket.list('data/Projects/ABIDE_Initiative/Outputs/cpac/func_minimal/'):
        filename = os.path.basename(os.path.normpath(key.name))
        if filename == 'func_minimal':
            continue

        path = os.path.join(data_folder, filename)

        if not os.path.isfile(path) or os.path.getsize(path) != key.size:
            print 'Downloading %s (%.0fMB)' % (filename, key.size / 1024**2)
            key.get_contents_to_filename(path)

if __name__ == '__main__':
    download()
