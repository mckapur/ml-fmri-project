import boto
import os

def download(data_folder='data/'):
	# Download MRI scan files from AWS S3 bucket
    conn = boto.connect_s3(anon=True)
    bucket = conn.get_bucket('fcp-indi')
    # We want to download from C-PAC pipeline
    for key in bucket.list('data/Projects/ABIDE_Initiative/Outputs/cpac/func_minimal/'):
        filename = os.path.basename(os.path.normpath(key.name))
        # C-PAC pipeline -> minimal pre-processing methodology
        if filename == 'func_minimal':
            continue

        # Save file to disk
        key.get_contents_to_filename(os.path.join(data_folder, filename))

if __name__ == '__main__':
    download()
