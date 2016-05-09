import boto
import os

def download(data_folder='data/'):
    conn = boto.connect_s3(anon=True)
    bucket = conn.get_bucket('fcp-indi')
    for key in bucket.list('data/Projects/ABIDE_Initiative/Outputs/cpac/func_minimal/'):
        filename = os.path.basename(os.path.normpath(key.name))
        if filename == 'func_minimal':
            continue

        key.get_contents_to_filename(os.path.join(data_folder, filename))

if __name__ == '__main__':
    download()
