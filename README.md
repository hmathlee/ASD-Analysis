# Autism Spectrum Disorder (ASD) and Eye-Gaze Analysis

This code serves to analyze eye-gaze in individuals with autism spectrum disorder (ASD) versus typical development (TD) individuals.

The analysis consists of two parts:
- Image segmentation with U-Net model
- Network analysis for ASD and TD eye-gaze

##U-Net

U-Net model training has been started. To continue the U-Net training, one should use the University of Waterloo MFCF student servers. Below are instructions to do so for Windows users:
- To begin, you'll need a VPN. If you are a UWaterloo student, see https://uwaterloo.atlassian.net/wiki/spaces/ISTKB/pages/262012949/How+to+install+and+connect+to+the+VPN+-+Windows+OS for instructions on how to download the Cisco AnyConnect Secure Mobility Client on Windows. Otherwise, you may need to see alternative options.

For UWaterloo students: once the Client is downloaded, you can connect to the VPN using your UWaterloo credentials and two-factor authentication:
- From _Start_, open _Remote Desktop_.
- Under "Computer", type _windows.student.math.uwaterloo.ca_. Click "Connect". If a window pops up saying "Do you trust this connection?" click "Connect" again.
- Click "Use another account" and enter _nexus\youruserid_ (replace _youruserid_ with your UWaterloo user ID) and your password. Click "OK". You should now be able to access the server.

**Note:** You will also need to import the PASCAL VOC 2012 image dataset into your workstation on the server:
- On _Remote Desktop_, when you have typed in the server name under "Computer", go to _Show Options_.
- Under the _Local Resources_ tab, under _Local drives and resources_, click "More".
- A new window will pop up. Under the _Drives_ section, check the box next to the drive from which you want to import your resources into your server workstation. Click "OK".

These instructions can also be found at: https://uwaterloo.ca/math-faculty-computing-facility/accessing-windows-student-server.

##Training the U-Net Model

Your workstation within the server comes with the Spyder Python IDE. However, it does not come with all of the necessary Python libraries to train the U-Net model.

It is beneficial to create your own environment for organizational purposes:
- From _Start_, launch _Anaconda Navigator_. In the _Environments_ tab, click _Create_.

In your new environment, you will need to install the missing libraries for U-Net model training. From _Anaconda Navigator_:
- Launch _CMD.exe Prompt_.

Type in each of the following commands (if there is a prompt that says "Proceed?" choose "y" for "yes"):
- _conda install tensorflow_ (should install Tensorflow version >= 2.0.0)
- _conda install matplotlib_
- _conda install scikit-image_
