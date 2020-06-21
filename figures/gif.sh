ffmpeg -y -i $1/%02d.png -vf palettegen $1/palette.png
ffmpeg -y -i $1/%02d.png -i $1/palette.png -lavfi paletteuse $1/video.gif
