#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
cd ..


#Download from Google Drive for Max Speed..
FILEID="1DPSYH3_l2T_iaTAjpkMc9oh1VOwEJdpc"
FILENAME="model.zip"   # change to whatever you want

# first request — get confirm token
wget --quiet --save-cookies cookies.txt --keep-session-cookies \
     "https://docs.google.com/uc?export=download&id=${FILEID}" -O- \
  | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt

# second request — use confirm token to download
wget --load-cookies cookies.txt \
     "https://docs.google.com/uc?export=download&confirm=$(<confirm.txt)&id=${FILEID}" \
     -O "${FILENAME}"

# clean up
rm -f cookies.txt confirm.txt

unzip -o "$FILENAME"



exit 0
