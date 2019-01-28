until python style_transfer.py; do
    echo "Program 'style transfer' crashed with exit code $?.  Restarting.." >&2
    sleep .5s
done