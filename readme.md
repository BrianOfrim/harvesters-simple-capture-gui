#Harvesters Simple Capture GUI
A simple application for acquiring and saving images using the Harvesters image acquisition library
Harvesters Repository: https://github.com/genicam/harvesters

##Getting started 
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

Now install the harvesters (preferably in a virtual env):
```
$ pip install harvesters
```


