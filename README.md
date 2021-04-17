# Autism Spectrum Disorder (ASD) and Eye-Gaze Analysis

This code serves to analyze eye-gaze in individuals with autism spectrum disorder (ASD) versus typical development (TD) individuals.

The analysis consists of two parts:
- Image segmentation with U-Net model
- Network analysis for ASD and TD eye-gaze

**U-Net**

U-Net model training has been started. To continue the U-Net training, one should use the University of Waterloo MFCF student servers. Below are instructions to do so for Windows users:
- To begin, you'll need a VPN. If you are a UWaterloo student, see https://uwaterloo.atlassian.net/wiki/spaces/ISTKB/pages/262012949/How+to+install+and+connect+to+the+VPN+-+Windows+OS for instructions on how to download the Cisco AnyConnect Secure Mobility Client on Windows. Otherwise, you may need to see alternative options.

For UWaterloo students: once the Client is downloaded, you can connect to the VPN using your UWaterloo credentials and two-factor authentication:
- From _Start_, open _Remote Desktop_.
- Under "Computer", type _windows.student.math.uwaterloo.ca_. Click "Connect". If a window pops up saying "Do you trust this connection?" click "Connect" again.
- Click "Use another account" and enter _nexus\youruserid_ (replace _youruserid_ with your UWaterloo user ID) and your password. You should now be able to access the server.

These instructions can also be found at: https://uwaterloo.ca/math-faculty-computing-facility/accessing-windows-student-server.
