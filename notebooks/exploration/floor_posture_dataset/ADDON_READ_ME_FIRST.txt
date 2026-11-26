floor_posture dataset - add-on: exploration notebook
=====================================================

A SMALL add-on to the four data parts you already have. No new recordings.

Unzip into the SAME folder as the other parts, so explore_the_data.ipynb
sits next to data/, recorder/ and previews/.

  floor_posture_dataset/
  |- explore_the_data.ipynb   <- new
  |- README.md                <- updated (points at the notebook)
  |- data/ recorder/ previews/   (already yours)

Running it
----------
From the bundle folder:

  python -m venv venv
  venv\Scriptsctivate           (Windows)
  source venv/bin/activate        (macOS / Linux)

  python -m pip install -e recorder
  python -m pip install jupyter ipykernel
  python -m ipykernel install --user --name floor-posture --display-name "floor-posture"
  python -m jupyter lab

Then choose the "floor-posture" kernel (Kernel -> Change Kernel).

If you see "ModuleNotFoundError: No module named 'vt_radar'", Jupyter is running
a different Python than the one you pip installed into. The notebook detects
this, tries to load the reader straight out of recorder/, and if that is not
possible prints the exact command to fix it. Using "python -m ..." for every
step above avoids it in the first place.

It works with however many data parts you have unpacked - part 1 alone is fine.
