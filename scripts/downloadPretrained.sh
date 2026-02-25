#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
cd ..


#Download from Google Drive for Max Speed..
FILEID="1DPSYH3_l2T_iaTAjpkMc9oh1VOwEJdpc"
FILENAME="model.zip"   # change to whatever you want
 
#curl -L -c cookies.txt "https://docs.google.com/uc?export=download&id=${FILEID}" -o page.html
#CONFIRM=$(sed -n 's/.*confirm=\([0-9A-Za-z_]*\).*/\1/p' /tmp/page.html | head -n 1)
#curl -L -b cookies.txt "https://docs.google.com/uc?export=download&confirm=${CONFIRM}&id=${FILEID}" -o "${OUT}"
#rm -f cookies.txt /tmp/page.html
# clean up
#rm -f cookies.txt page.html

#Use Google Docs
#wget "drive.google.com/u/3/uc?id=$FILEID&export=download&confirm=yes" -O "$FILENAME"

#Use trusty old ammar server..
wget  "http://ammar.gr/ymapnet/archive/YMapNET_ICRA26.zip" -O "$FILENAME"

unzip -o "$FILENAME"



exit 0
