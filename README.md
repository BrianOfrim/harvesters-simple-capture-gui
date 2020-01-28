# Harvesters Simple Capture GUI

**This project is now deprecated, it's functionality has been added to https://github.com/BrianOfrim/boja**

A simple application for acquiring and saving images using the Harvesters image acquisition library
Harvesters Repository: https://github.com/genicam/harvesters

## Getting started 
### Installing dependencies
Harvesters is a consumer that requires a genTL (GenICam Transport Layer) producer to produce images for it to consume.

A good option for a GenTL producer is one supplied by Matrix Vision. To install it visit:  
http://static.matrix-vision.com/mvIMPACT_Acquire/2.29.0/
And download the following two packages:  
install_mvGenTL_Acquire.sh  
mvGenTL_Acquire-x86_64_ABI2-2.29.0.tgz  

Then run the following with your user (not sudo):
```
$ ./install_mvGenTL_Acquire.sh
```
This will install mvIMPACT_Acquire at the location /opt/mvIMPACT_Acquire  

The file that we are concerned about is the genTL producer which by default will be located at:  
/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti

Now install the pip dependencies (preferably in a virtual env):
```
$ pip install -r requirements.txt
```

### Running the application
To run the application navigate to the **hscg/** directory and run the **capture.py** file:  
```
$ cd hscg
$ python capture.py
```

#### Options
There are various command line options that can be seen by running:  
```
$ python capture.py --help
```
Example of these options include the location to store captured images, image file type, frame rate etc.  

#### AWS S3 Integration
The application can be configured to send images to an AWS S3 bucket in addition to saving them locally.
To have the application save images to a bucket you have access to run:  
```
$ python capture.py --s3_bucket_name <bucket_name>
```

You can also configure the directory within your s3 bucket to store the images at by supplying a value to the **--s3_image_dir** argument.
This will default to **data/images/**
